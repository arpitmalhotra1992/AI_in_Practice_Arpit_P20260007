#!/usr/bin/env python3
"""Lab 5 — the failure classifier (plus the fix).

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto
    python labs/lab5/diagnose.py --input reports/lab4.json --save reports/lab5_diagnosis.json
    python labs/lab5/diagnose.py --compare --save reports/lab5_before_after.json  # Part D

Implements the T4 §5 diagnostic tree. Everything decidable by code is decided
by code; mode 2 needs eyes and the script says so.

The v2 fix lives at the bottom of this file (Part C): one fix at a time,
measured each time. lab4/ is untouched -- it stays the v1 baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

_STOP = {
    "about", "after", "all", "also", "and", "answer", "are", "because", "been",
    "before", "being", "between", "both", "but", "claim", "claims", "corpus",
    "cover", "covered", "covers", "days", "does", "during", "each", "follow",
    "from", "have", "into", "more", "most", "must", "only", "other", "over",
    "plan", "plans", "policy", "question", "same", "should", "such", "than",
    "that", "their", "there", "these", "this", "those", "through", "under",
    "used", "using", "what", "when", "where", "which", "while", "will",
    "with", "within", "your",
}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test: is the answer in the corpus at all?

    Improved over the shipped stub in two ways. (1) Numbers gate: if the gold
    answer names numbers, at least one must appear (comma-normalised) in the
    relevant documents -- this stops paraphrase false-passes like "a month"
    matching nothing while still catching "30 days". (2) Stopword filtering so
    generic insurance vocabulary ("policy", "claim", "cover") cannot carry the
    overlap ratio alone. The REFUSE./PARTIAL REFUSE. label prefixes that Lab 4
    gold answers carry are stripped before testing -- they describe the
    expected behaviour, not corpus content.
    """
    gold = re.sub(r"^(PARTIAL\s+)?REFUSE[^.]*\.\s*", "", gold_answer.strip(),
                  flags=re.IGNORECASE)
    text = " ".join(corpus.get(d, "") for d in relevant_docs)
    if not text.strip():
        return False
    gold_nums = set(re.findall(r"\d[\d,]*", gold))
    if gold_nums:
        flat = text.replace(",", "")
        if not any(n.replace(",", "") in flat for n in gold_nums):
            return False
    tokens = [t.strip(".,;()\"'") for t in gold.lower().split()
              if len(t) > 4 and t.strip(".,;()\"'") not in _STOP]
    if not tokens:
        return True
    low = text.lower()
    hits = sum(1 for t in tokens if t in low)
    return hits / len(tokens) > 0.4


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             gold_chunk_in_final: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    Ordering is the tree's ordering, which is what makes modes exclusive.
    Note the inversion: gold_context_fixes_it == True means RETRIEVAL failed
    (the generator was capable and was starved), not generation.

    Rank signals are chunk-level: `gold_chunk_in_final` says the proxy gold
    chunk (highest gold-answer overlap among relevant-doc chunks) was in the
    final-k context; `in_top_30` says it was in the top-30 pool. Doc-level
    membership is evidence only, because Q44 shows a relevant doc can rank #2
    via the wrong chunk.
    """
    # Mode 7 first: right answer, wrong citation -- not a retrieval failure.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content (numbers/entities) not found in relevant docs"

    # Mode 6 branch: gold context does NOT fix it -> generation was always
    # going to fail, no retrieval change can help.
    if gold_context_fixes_it is False:
        return 6, "still wrong with gold context -- generator fails on perfect input"

    # Retrieval branch (gold fixes it): subdivide by chunk-rank signals.
    if gold_context_fixes_it is True:
        if dropped_by_reranker:
            return 5, "gold chunk in top-30 pool but reranker dropped it from final_k"
        if gold_chunk_in_final:
            # The exact span was in context and the model still failed on the
            # noisy retrieved context: distractor/conflict overload. Fixed
            # under the mode-6 menu (fewer distractors, archived filtering).
            return 6, ("gold chunk WAS in the final context; clean gold context "
                       "fixes it -- distractor/conflict overload")
        if in_top_30 is False:
            return 3, "gold chunk not in top 30 (verbatim check in evidence)"
        return 4, "gold chunk in top 30 but outside the final-k context"

    # Signals unavailable (e.g. --no-gold): fall through to human check.
    if in_top_30:
        if dropped_by_reranker:
            return 5, "gold chunk in top-30 pool but reranker dropped it from final_k"
        return 4, "gold chunk in top 30 but outside the final-k context"
    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# signal collection (the automatable checks)
# ---------------------------------------------------------------------------
def _dedup_doc_ids(hits) -> list[str]:
    seen, out = set(), []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            out.append(h.doc_id)
    return out


def retrieval_signals(question: dict, retriever, k_pool: int = 30,
                      final_k: int = 5) -> dict:
    """Chunk-level rank signals, doc-level evidence, and the verbatim probe.

    The proxy gold chunk is the relevant-doc chunk (same markdown-400
    chunking the index uses) with the highest word overlap to the gold
    answer. Its membership in the top-30 pool vs the final-k context is what
    separates modes 3/4 from mode 6-distractor (Q44: relevant doc ranked #2
    via the wrong chunk).
    """
    from aip.chunking import markdown_chunks

    relevant = question["relevant_docs"]
    gold_text = question["gold_answer"].lower()
    gold_words = set(re.findall(r"[a-z0-9]+", gold_text))
    gold_nums = set(re.findall(r"\d[\d,]*", gold_text))
    # Score chunks by number matches (weighted: numbers are the load-bearing
    # content of policy answers) plus word overlap. Pure word overlap picks
    # headers (Q29: m0 header outscored the m4 rule); numbers pick the rule.
    proxy, proxy_overlap = None, -1
    if relevant:
        for doc_id in relevant:
            for c in markdown_chunks(_CORPUS.get(doc_id, ""), doc_id, size=400):
                ct = c.text.lower().replace(",", "")
                num_hit = sum(1 for n in gold_nums if n.replace(",", "") in ct)
                tok_hit = len(gold_words & set(re.findall(r"[a-z0-9]+", ct)))
                score = 3 * num_hit + tok_hit
                if score > proxy_overlap:
                    proxy, proxy_overlap = c, score
    if proxy is None or proxy_overlap <= 0:
        proxy = None

    pool_hits = retriever.search(question["question"], k=k_pool)
    pool_chunk_ids = [h.chunk.chunk_id for h in pool_hits]
    top30_docs = _dedup_doc_ids(pool_hits)
    # final-k context, recomputed deterministically (search k=12, take 5 --
    # the v1 pipeline's exact rule)
    final_hits = retriever.search(question["question"], k=12)[:final_k]
    final_chunk_ids = [h.chunk.chunk_id for h in final_hits]

    in_top_30_chunks = bool(proxy) and proxy.chunk_id in pool_chunk_ids
    gold_chunk_in_final = bool(proxy) and proxy.chunk_id in final_chunk_ids

    # Verbatim probe for the mode 2/3 split: if the proxy chunk's own text
    # retrieves its document, the query is the problem (mode 3); if not even
    # that retrieves it, the chunk is unfindable as embedded (mode 2 -> eyes).
    verbatim_ok: bool | None = None
    verbatim_detail = ""
    if proxy is not None and not in_top_30_chunks:
        probe = " ".join(proxy.text.split()[:40])
        got = _dedup_doc_ids(retriever.search(probe, k=5))
        verbatim_ok = proxy.doc_id in got
        verbatim_detail = (f"proxy-gold-chunk {proxy.chunk_id} "
                           f"(overlap {proxy_overlap}); verbatim top-5={got}")

    return {"in_top_30": in_top_30_chunks,
            "gold_chunk_in_final": gold_chunk_in_final,
            "proxy_chunk": proxy.chunk_id if proxy else None,
            "in_final_docs": (bool(relevant) and
                            any(d in set(_dedup_doc_ids(final_hits))
                                for d in relevant)),
            "top30_docs": top30_docs,
            "final_docs": _dedup_doc_ids(final_hits),
            "verbatim_ok": verbatim_ok, "verbatim_detail": verbatim_detail}


_CORPUS: dict[str, str] = {}


def diagnose_failures(rows: list[dict], questions: dict, corpus: dict[str, str],
                      retriever, *, with_gold: bool = True,
                      verbose: bool = True) -> list[dict]:
    """Classify every failure. Returns per-question records with evidence."""
    global _CORPUS
    _CORPUS = corpus
    from aip.cost import Budget
    from labs.lab4.rag import answer_with_gold_context
    from labs.lab4.evaluate import judge_correctness

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    if verbose:
        print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    with Budget(limit_usd=1.00, label="lab5-diagnose") as b:
        for r in failures:
            q = questions[r["id"]]
            sig = retrieval_signals(q, retriever)
            gold_fixes: bool | None = None
            gold_detail = ""
            if with_gold and q["relevant_docs"]:
                g = answer_with_gold_context(
                    q["question"],
                    [corpus[d] for d in q["relevant_docs"] if d in corpus])
                gs = judge_correctness(q["question"], g.text, q["gold_answer"])
                base = r.get("correctness") or 0
                gold_fixes = (gs is not None) and (gs > base)
                from labs.lab4.rag import REFUSAL as _REF
                gold_detail = (f"gold_correctness={gs} vs retrieved={base}; "
                               f"gold_refused={g.text.strip() == _REF}; "
                               f"gold_ans={g.text[:160]!r}")
            elif with_gold:
                gold_detail = "no relevant docs -- gold test undefined"
            mode, evidence = classify(
                r, q, corpus, gold_context_fixes_it=gold_fixes,
                in_top_30=sig["in_top_30"], dropped_by_reranker=False,
                gold_chunk_in_final=sig["gold_chunk_in_final"])
            # attach the verbatim probe outcome to mode-3 evidence; a failed
            # probe converts the provisional mode 3 into a human mode-2 check
            if mode == 3 and sig["verbatim_detail"]:
                if sig["verbatim_ok"] is False:
                    mode, evidence = (2, "needs_human_check: " + sig["verbatim_detail"] +
                                      " -- even verbatim chunk text misses; inspect boundary")
                else:
                    evidence += "; " + sig["verbatim_detail"]
            evidence += (f" | proxy={sig['proxy_chunk']} "
                         f"in_final_docs={sig['in_final_docs']} "
                         f"final_docs={sig['final_docs']}")
            if gold_detail:
                evidence += " | " + gold_detail
            tally[mode] += 1
            rec = {"id": r["id"], "kind": q["kind"], "mode": mode,
                   "mode_name": MODES[mode], "evidence": evidence,
                   "question": q["question"], "answer": r["answer"][:300],
                   "in_top_30": sig["in_top_30"],
                   "gold_chunk_in_final": sig["gold_chunk_in_final"],
                   "in_final_docs": sig["in_final_docs"],
                   "verbatim_ok": sig["verbatim_ok"],
                   "gold_fixes": gold_fixes}
            out.append(rec)
            if verbose:
                print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence[:220]}")

    if verbose:
        print("\n" + pareto(tally))
        print("\n" + b.report())
        print("\nCases marked needs_human_check are Part A2. Open them.")
    return out


# ---------------------------------------------------------------------------
# Part C — the fix (one variable at a time)
# ---------------------------------------------------------------------------
# FIX V2 (candidate 1): archived-document exclusion at ingest.
# The corpus retains claims-timelines-2024-ARCHIVED "for reference only" with
# an explicit must-not-quote header, yet the v1 index embeds it on equal
# footing, so it lands in the top-5 (Q01 rank 3, Q30 rank 1, Q31 rank 3) and
# forces the generator into conflict-surfacing or refusal. Lab 3 D3 showed a
# metadata filter taking Q29-31 hit_rate@1 0.667 -> 1.000. DenseRetriever has
# no `where` clause, so the equivalent here is ingest-side exclusion.
ARCHIVED_DOCS = {"claims-timelines-2024-ARCHIVED"}

FIX_DESCRIPTION = (
    "v2 excludes claims-timelines-2024-ARCHIVED at ingest (same markdown-400 "
    "chunking, same exact dense retrieval, same k=12/final_k=5, same prompt). "
    "Rationale: the document is superseded by claims-timelines (effective "
    "1 April 2026) and carries a must-not-quote header; it is a data-layer "
    "fix (T4 §6.3 defence #2, cheapest), not a modelling change."
)


def build_v2_retriever():
    """v1 pipeline minus the superseded document. Everything else identical."""
    from aip.chunking import markdown_chunks
    from aip.retrieval import DenseRetriever
    from labs.lab3.search import load_corpus

    corpus = {d: t for d, t in load_corpus().items() if d not in ARCHIVED_DOCS}
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks, show_progress=False)


# ---------------------------------------------------------------------------
# FIX V3 (candidate 2, recorded negative): hybrid BM25+dense (RRF).
# Targets the ranking/embedding cluster (Q23 proxy rank 17->5 under hybrid in
# a retrieval-only probe; Q44 Lab-3 doc-level precedent). Lab 3 measured
# hybrid ndcg@10 0.7949 BELOW dense 0.8527 on this corpus, so the prior is
# net-negative; it is attempted second, alone, precisely to record the number.
# Shares v1 chunking (markdown-400, archived doc INCLUDED -- one variable).
def build_v3_hybrid():
    from aip.chunking import markdown_chunks
    from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever
    from labs.lab3.search import load_corpus

    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    dense = DenseRetriever(chunks, show_progress=False)
    bm25 = Bm25Retriever(chunks)
    return HybridRetriever([dense, bm25])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    ap.add_argument("--no-gold", action="store_true",
                    help="skip the LLM gold-context test (retrieval signals only)")
    ap.add_argument("--compare", action="store_true",
                    help="Part D: run v1 vs v2 full comparison")
    args = ap.parse_args()

    if args.compare:
        run_comparison(args.save if args.save != "reports/lab5_diagnosis.json"
                       else "reports/lab5_before_after.json")
        return

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()
    from labs.lab4.evaluate import build_retriever
    retriever = build_retriever()

    out = diagnose_failures(rows, questions, corpus, retriever,
                            with_gold=not args.no_gold)
    tally = Counter(r["mode"] for r in out)
    if args.pareto:
        print("\n" + pareto(tally))

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


def run_comparison(save: str) -> None:
    """Part D: v1 vs v2 on the same questions, every Lab 4 metric."""
    import statistics

    from aip.cost import Budget
    from aip.evals import retrieval_metrics
    from aip.retrieval import format_context
    from labs.lab4.evaluate import (build_retriever, judge_correctness,
                                    judge_faithfulness)
    from labs.lab4.rag import answer_question

    questions = load_questions(include_unanswerable=True)
    corpus = load_corpus()
    r1 = build_retriever()
    r2 = build_v2_retriever()

    def run_all(retriever, label):
        rows = []
        with Budget(limit_usd=1.00, label=label) as b:
            for q in questions:
                a = answer_question(q["question"], retriever)
                ctx = format_context(a.hits)
                unans = not q["relevant_docs"] or q["kind"] == "unanswerable"
                seen, ranked = set(), []
                for h in a.hits:
                    if h.doc_id not in seen:
                        seen.add(h.doc_id)
                        ranked.append(h.doc_id)
                rm = retrieval_metrics(ranked, q["relevant_docs"] or [], ks=(1, 3, 5, 10)) \
                    if q["relevant_docs"] else {}
                rows.append({
                    "id": q["id"], "kind": q["kind"], "unanswerable": unans,
                    "answer": a.text, "refused": a.refused,
                    "citations_valid": a.citations_valid,
                    "faithfulness": judge_faithfulness(a.text, ctx),
                    "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                    "retrieved": [h.doc_id for h in a.hits],
                    "relevant": q["relevant_docs"],
                    "retrieval": rm,
                })
        return rows, b

    rows1, b1 = run_all(r1, "lab5-v1-baseline")
    rows2, b2 = run_all(r2, "lab5-v2-archived-filter")

    def agg(rows):
        ans = [r for r in rows if not r["unanswerable"]]
        una = [r for r in rows if r["unanswerable"]]
        ref = [r for r in rows if r["refused"]]
        f = lambda xs: statistics.fmean([x for x in xs if x is not None]) \
            if any(x is not None for x in xs) else float("nan")
        m = {
            "citation_validity": f([r["citations_valid"] for r in rows]),
            "faithfulness": f([r["faithfulness"] for r in rows]),
            "correctness": f([r["correctness"] for r in ans]) / 2,
            "refusal_recall": sum(1 for r in una if r["refused"]) / len(una),
            "refusal_precision": (sum(1 for r in ref if r["unanswerable"]) / len(ref)
                                  if ref else 1.0),
            "n_refusals": len(ref),
        }
        for k in ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10"):
            vals = [r["retrieval"].get(k) for r in rows
                    if r["relevant"] and k in r["retrieval"]]
            m[k] = statistics.fmean(vals) if vals else float("nan")
        return m

    a1, a2 = agg(rows1), agg(rows2)
    print(f"\n{'metric':<20}{'v1':>10}{'v2':>10}{'Δ':>10}")
    for k in a1:
        d = a2[k] - a1[k]
        print(f"{k:<20}{a1[k]:>10.3f}{a2[k]:>10.3f}{d:>+10.3f}")
    print(f"\n{'cost/query $':<20}{b1.spent_usd/len(rows1):>10.4f}"
          f"{b2.spent_usd/len(rows2):>10.4f}"
          f"{b2.spent_usd/len(rows2)-b1.spent_usd/len(rows1):>+10.4f}")
    print(f"{'p95 latency ms':<20}{b1.percentile(95):>10.0f}"
          f"{b2.percentile(95):>10.0f}{b2.percentile(95)-b1.percentile(95):>+10.0f}")
    print("\n" + b1.report() + "\n" + b2.report())

    p = ROOT / save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "fix": FIX_DESCRIPTION,
        "v1": {"aggregate": a1, "budget": b1.as_dict(), "rows": rows1},
        "v2": {"aggregate": a2, "budget": b2.as_dict(), "rows": rows2},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
