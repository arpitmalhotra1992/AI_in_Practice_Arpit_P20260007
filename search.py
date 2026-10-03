#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever, Retriever  # noqa: E402
from aip.retrieval import CrossEncoderReranker, LLMReranker  # noqa: E402
from aip.retrieval import Bm25Retriever, ChromaRetriever, DenseRetriever, HybridRetriever, Retriever  # noqa: E402

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """A1-A3: four strategies at 800, size sweep on the winner, markdown prefix on/off.
    A4: one failure example, printed for the report.
    """
    corpus = load_corpus()
    questions = load_questions()

    # --- A1: all four strategies at size=800 ---
    print("=" * 70)
    print("A1 — four strategies at size=800")
    print("=" * 70)
    rows_a1 = {}
    chunk_counts = {}
    for strat in ("fixed", "sliding", "recursive", "markdown"):
        t0 = time.perf_counter()
        chunks = build_chunks(corpus, strat, 800)
        r = DenseRetriever(chunks)
        build_ms = (time.perf_counter() - t0) * 1000
        m = evaluate(r, questions)
        m["_build_ms"] = build_ms
        m["_n_chunks"] = len(chunks)
        rows_a1[strat] = m
        chunk_counts[strat] = len(chunks)
    print(table(rows_a1))
    print("chunk counts:", chunk_counts)
    print()

    winner = max(rows_a1, key=lambda s: rows_a1[s]["ndcg@10"])
    print(f"winner at size=800: {winner} (ndcg@10={rows_a1[winner]['ndcg@10']:.4f})\n")

    # --- A2: size sweep on the winner ---
    print("=" * 70)
    print(f"A2 — size sweep on '{winner}'")
    print("=" * 70)
    rows_a2 = {}
    for size in (400, 800, 1600):
        chunks = build_chunks(corpus, winner, size)
        r = DenseRetriever(chunks)
        m = evaluate(r, questions)
        m["_n_chunks"] = len(chunks)
        rows_a2[f"{winner}-{size}"] = m
    print(table(rows_a2))
    print("chunk counts:",
          {k: v["_n_chunks"] for k, v in rows_a2.items()})
    print()

    # --- A3: markdown prefix on vs off (only meaningful if winner is markdown;
    #          run it regardless so you have the comparison either way) ---
    print("=" * 70)
    print("A3 — markdown heading-path prefix, on vs off")
    print("=" * 70)
    md_chunks = build_chunks(corpus, "markdown", 800)
    md_stripped = [
        Chunk(
            chunk_id=c.chunk_id,
            text=c.text.split("]\n", 1)[1] if c.text.startswith("[") and "]\n" in c.text else c.text,
            doc_id=c.doc_id,
            meta=dict(c.meta),
        )
        for c in md_chunks
    ]
    r_with = DenseRetriever(md_chunks)
    r_without = DenseRetriever(md_stripped)
    m_with = evaluate(r_with, questions)
    m_without = evaluate(r_without, questions)
    print(table({"markdown-800 WITH prefix": m_with,
                "markdown-800 WITHOUT prefix": m_without}))
    print()

    # --- A4: one failure example ---
    print("=" * 70)
    print("A4 — one question where chunking is the culprit")
    print("=" * 70)
    best_chunks = build_chunks(corpus, winner, 800)
    r = DenseRetriever(best_chunks)
    worst_q, worst_score = None, 2.0
    for q in questions:
        hits = r.search(q["question"], k=5)
        ranked = []
        seen = set()
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)
        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(5,))
        if m["hit_rate@5"] < worst_score:
            worst_score = m["hit_rate@5"]
            worst_q = q

    if worst_q:
        print(f"Question {worst_q['id']}: {worst_q['question']}")
        print(f"Relevant doc(s): {worst_q['relevant_docs']}")
        print(f"hit_rate@5: {worst_score}")
        print("\nTop 5 retrieved chunks:")
        hits = r.search(worst_q["question"], k=5)
        for h in hits:
            print(f"  [{h.doc_id}] {h.text[:150]!r}")
        print("\n-> Manually inspect data/corpus/{}.md for the span that".format(
            worst_q['relevant_docs'][0] if worst_q['relevant_docs'] else '?'))
        print("   SHOULD have matched, and compare it to what's printed above.")
    else:
        print("No clear single-question failure found at hit_rate@5 < 1.0 "
              "for this configuration — note this in the report as a strength, "
              "not a gap.")


def sweep_retrieval() -> None:
    """B1-B4.

    B1: dense / bm25 / hybrid on your best chunking.
    B2: print kind_table(m, col='mrr') for each, and pull out Q44 and Q41
        individually from metrics['_per_question_mrr'].

        USE MRR, NOT hit_rate@5. Every retriever here scores 0.93-0.98 on
        hit_rate@5, so it is saturated and shows you nothing -- which is why
        kind_table() and metrics['_per_question'] both default to it. That
        default is the trap, and noticing it is part of the lab.

    B3: RRF k in {10, 30, 60, 100} -- HybridRetriever(..., rrf_k=k).
    B4: unequal fusion weights -- HybridRetriever(..., weights=[2.0, 1.0]).
    """
    corpus = load_corpus()
    questions = load_questions()

    # Part A's winner: markdown-aware chunking at 400 characters.
    chunks = build_chunks(corpus, "markdown", 400)
    print(f"chunking fixed at markdown-400 ({len(chunks)} chunks) -- "
          f"carrying forward Part A's winner\n")

    # --- B1: dense / bm25 / hybrid ---
    print("=" * 70)
    print("B1 -- dense vs BM25 vs hybrid")
    print("=" * 70)
    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    rows_b1 = {}
    metrics_b1 = {}
    for name, r in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid)):
        m = evaluate(r, questions)
        rows_b1[name] = m
        metrics_b1[name] = m
    print(table(rows_b1))
    print()

    # --- B2: per-kind breakdown on MRR (not the saturated default) ---
    print("=" * 70)
    print("B2 -- per-kind breakdown, MRR (hit_rate@5 is saturated here)")
    print("=" * 70)
    for name, m in metrics_b1.items():
        print(f"--- {name} ---")
        print(kind_table(m, col="mrr"))
        print()

    print("Q44 (identifier 'AUR-HI-SIL-2026') and Q41 (paraphrase, no lexical overlap):")
    for name, m in metrics_b1.items():
        q44 = m["_per_question_mrr"].get("Q44")
        q41 = m["_per_question_mrr"].get("Q41")
        print(f"  {name:<8} Q44 mrr={q44}   Q41 mrr={q41}")
    print()

    # --- B3: RRF k sweep ---
    print("=" * 70)
    print("B3 -- RRF k sweep")
    print("=" * 70)
    rows_b3 = {}
    for k in (10, 30, 60, 100):
        hyb_k = HybridRetriever([dense, bm25], rrf_k=k)
        m = evaluate(hyb_k, questions)
        rows_b3[f"hybrid rrf_k={k}"] = m
    print(table(rows_b3))
    print("-> effect should be small; resist tuning k further, it's noise at n=42.\n")

    # --- B4: unequal fusion weights ---
    print("=" * 70)
    print("B4 -- unequal fusion weights")
    print("=" * 70)
    rows_b4 = {"hybrid 1:1": metrics_b1["hybrid"]}
    for w in ([2.0, 1.0], [1.0, 2.0], [3.0, 1.0]):
        hyb_w = HybridRetriever([dense, bm25], weights=w)
        m = evaluate(hyb_w, questions)
        rows_b4[f"hybrid weights={w}"] = m
    print(table(rows_b4))
    print("-> be honest about whether any of these beat 1:1 outside noise at n=42.\n")

    # --- B5: the headline finding, whichever way it lands ---
    print("=" * 70)
    print("B5 -- dense vs hybrid, the finding to report honestly")
    print("=" * 70)
    dense_ndcg = metrics_b1["dense"]["ndcg@10"]
    hybrid_ndcg = metrics_b1["hybrid"]["ndcg@10"]
    better = "hybrid" if hybrid_ndcg > dense_ndcg else "dense"
    print(f"dense ndcg@10={dense_ndcg:.4f}   hybrid ndcg@10={hybrid_ndcg:.4f}")
    print(f"-> {better} wins on this corpus. Report this as-is, even if it "
          f"contradicts T4 §4.3's general claim about hybrid.")

    # Count per-question wins to support the mechanism explanation.
    dense_mrr = metrics_b1["dense"]["_per_question_mrr"]
    bm25_mrr = metrics_b1["bm25"]["_per_question_mrr"]
    dense_wins = sum(1 for qid in dense_mrr
                     if dense_mrr[qid] != bm25_mrr.get(qid)
                     and dense_mrr[qid] > bm25_mrr.get(qid, 0))
    bm25_wins = sum(1 for qid in dense_mrr
                    if dense_mrr[qid] != bm25_mrr.get(qid)
                    and bm25_mrr.get(qid, 0) > dense_mrr[qid])
    print(f"   dense beats bm25 on {dense_wins} questions where they differ; "
          f"bm25 beats dense on {bm25_wins}.")


def sweep_rerank() -> None:
    corpus = load_corpus()
    questions = load_questions()

    # Carry forward Part A/B's winner: markdown-400 chunking, dense retrieval.
    chunks = build_chunks(corpus, "markdown", 400)
    dense = DenseRetriever(chunks)
    print(f"base retriever: dense on markdown-400 ({len(chunks)} chunks)\n")

    baseline_m = evaluate(dense, questions)

    print("=" * 70)
    print("C1 -- cross-encoder reranker (retrieve k=30, rerank to 5)")
    print("=" * 70)
    ce = CrossEncoderReranker()
    t0 = time.perf_counter()
    ce_m = evaluate(dense, questions, k=30, reranker=ce, final_k=5)
    ce_ms = (time.perf_counter() - t0) * 1000 / len(questions)
    print(table({"dense (no rerank)": baseline_m, "dense + cross-encoder": ce_m}))
    print(f"\nadded p95 latency: {ce_m['latency_p95_ms']:.1f}ms "
          f"(baseline was {baseline_m['latency_p95_ms']:.1f}ms); "
          f"avg per-query time with reranking: {ce_ms:.1f}ms")
    ce_delta_ndcg = ce_m["ndcg@10"] - baseline_m["ndcg@10"]
    print(f"delta ndcg@10: {ce_delta_ndcg:+.4f}  -> "
          f"{'reranker HELPS' if ce_delta_ndcg > 0 else 'reranker HURTS'} here\n")

    print("=" * 70)
    print("C2 -- LLM reranker (retrieve k=30, rerank to 5) -- slow, costs money")
    print("=" * 70)
    llm_rr = LLMReranker(tier="SMALL")
    t0 = time.perf_counter()
    llm_m = evaluate(dense, questions, k=30, reranker=llm_rr, final_k=5)
    llm_wall_s = time.perf_counter() - t0
    print(table({"dense (no rerank)": baseline_m, "dense + LLM rerank": llm_m}))
    print(f"\nwall time for all {len(questions)} questions: {llm_wall_s:.1f}s "
          f"({llm_wall_s / len(questions):.2f}s/question -- 30 sequential calls each)")
    llm_delta_ndcg = llm_m["ndcg@10"] - baseline_m["ndcg@10"]
    print(f"delta ndcg@10: {llm_delta_ndcg:+.4f}\n")

    print("=" * 70)
    print("C3 -- the decision table")
    print("=" * 70)
    decision_cols = ("ndcg@10", "hit_rate@1", "recall@5", "latency_p95_ms")
    print(table(
        {
            "dense, no rerank": baseline_m,
            "dense + cross-encoder": ce_m,
            "dense + LLM rerank": llm_m,
        },
        cols=decision_cols,
    ))
    print(
        "\nInteractive agent-facing search box: the LLM reranker's latency "
        f"(~{llm_wall_s / len(questions):.1f}s/query) is unusable at human "
        "typing-to-answer speed; the cross-encoder's ~tens-of-ms is tolerable, "
        "but here it doesn't earn its latency (see delta above) -- so for the "
        "interactive case, ship 'dense, no rerank' unless the cross-encoder's "
        "delta turns out positive.\n"
        "Overnight batch (e.g. regenerating all FAQ answers): latency is free, "
        "so take whichever of the three has the best ndcg@10 -- usually the "
        "LLM reranker -- and pay the per-call cost once, offline."
    )
    print()

    print("=" * 70)
    print("C4 -- a query reranking made worse")
    print("=" * 70)
    base_mrr = baseline_m["_per_question_mrr"]
    ce_mrr = ce_m["_per_question_mrr"]
    worst_qid, worst_drop = None, 0.0
    for qid, before in base_mrr.items():
        after = ce_mrr.get(qid, before)
        drop = before - after
        if drop > worst_drop:
            worst_drop = drop
            worst_qid = qid
    if worst_qid:
        q = next(q for q in questions if q["id"] == worst_qid)
        print(f"{worst_qid}: {q['question']!r}")
        print(f"MRR before rerank: {base_mrr[worst_qid]:.4f}  "
              f"after: {ce_mrr[worst_qid]:.4f}  (drop: {worst_drop:.4f})")
        print("-> likely cause: the cross-encoder (trained on web-search query/doc "
              "pairs) is out of domain on policy prose and re-scores a topically "
              "close but wrong chunk above the correct one that dense had ranked "
              "correctly.")
    else:
        print("No question got worse under reranking in this run -- note that as "
              "a (mild) finding rather than forcing one.")


def sweep_index() -> None:
    corpus = load_corpus()
    questions = load_questions()

    # Carry forward the winning config: markdown-400 chunking, dense retrieval.
    chunks = build_chunks(corpus, "markdown", 400)

    print("=" * 70)
    print("D1 -- ChromaRetriever (HNSW) vs DenseRetriever (exact) -- quality + latency")
    print("=" * 70)
    exact = DenseRetriever(chunks)
    chroma = ChromaRetriever(chunks, reset=True)

    m_exact = evaluate(exact, questions)
    m_chroma = evaluate(chroma, questions)
    print(table({"exact (DenseRetriever)": m_exact, "HNSW (ChromaRetriever)": m_chroma}))
    recall_gap = m_exact["recall@5"] - m_chroma["recall@5"]
    print(f"\nrecall@5 gap (exact - HNSW): {recall_gap:+.4f} "
          f"-- {'small, as expected at this scale' if abs(recall_gap) < 0.03 else 'larger than expected, worth a look'}")
    print(f"latency_p95: exact={m_exact['latency_p95_ms']:.4f}ms  "
          f"HNSW={m_chroma['latency_p95_ms']:.4f}ms "
          f"-- {'HNSW is SLOWER here (per-query graph traversal + Python overhead beats a single BLAS matmul at this tiny scale)' if m_chroma['latency_p95_ms'] > m_exact['latency_p95_ms'] else 'HNSW is faster'}\n")

    print("=" * 70)
    print("D2 -- timing exact vs HNSW as the corpus scales")
    print("=" * 70)
    print(f"~{len(chunks)} chunks (current corpus):")
    print(f"  exact p95:  {m_exact['latency_p95_ms']:.4f}ms")
    print(f"  HNSW  p95:  {m_chroma['latency_p95_ms']:.4f}ms")
    print()
    print("To get the 4k-chunk and 40k-chunk timings, run (once, separately):")
    print("  python scripts/expand_corpus.py --docs 4000")
    print("Then re-run this sweep -- build_chunks() will pick up the expanded")
    print("corpus automatically. The filler documents carry no golden answers,")
    print("so they only inflate the index, not your quality numbers.")
    print("Report all three timings (~160 / ~4k / ~40k chunks) and name the")
    print("crossover point where HNSW overtakes exact search.\n")

    print("=" * 70)
    print("D3 -- metadata filtering fixes the archived-document trap")
    print("=" * 70)
    trap_ids = {"Q29", "Q30", "Q31"}
    trap_qs = [q for q in questions if q["id"] in trap_ids]

    tagged_chunks = []
    for c in chunks:
        meta = dict(c.meta)
        meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"
        tagged_chunks.append(Chunk(chunk_id=c.chunk_id, text=c.text, doc_id=c.doc_id, meta=meta))

    chroma_tagged = ChromaRetriever(tagged_chunks, reset=True)

    def hit_at_1(retriever, qs, **search_kw):
        hits_ok = 0
        for q in qs:
            hits = retriever.search(q["question"], k=5, **search_kw)
            seen, ranked = set(), []
            for h in hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    ranked.append(h.doc_id)
            m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1,))
            hits_ok += m["hit_rate@1"]
        return hits_ok / len(qs)

    before = hit_at_1(chroma_tagged, trap_qs)
    after = hit_at_1(chroma_tagged, trap_qs, where={"status": "current"})
    print(f"Q29/Q30/Q31 hit_rate@1 before filter: {before:.4f}")
    print(f"Q29/Q30/Q31 hit_rate@1 after  filter: {after:.4f}")
    print(f"delta: {after - before:+.4f}\n")
    print("-> this required zero changes to the retriever itself -- it's a data/")
    print("   metadata fix, not a modelling fix. Keep the archived doc (you may")
    print("   need it for an audit); just stop it from winning the race at query time.")


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
