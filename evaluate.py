#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import (  # noqa: E402
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    judge_agreement,
    llm_judge,
)
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"


def build_retriever():
    """Lab 3 winning configuration: markdown-aware chunking at 400 chars,
    exact dense retrieval (see labs/lab3/report.md, Part A/B)."""
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks)


# ---------------------------------------------------------------------------
# judges (yours)
# ---------------------------------------------------------------------------
IMPROVED_RUBRIC_FAITHFULNESS = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Rules:
- Judge support only. Do NOT judge whether the answer is helpful, well written,
  or matches your own knowledge.
- An answer is unsupported (score 0) if it states anything the context does not
  contain, even if that statement is true in the real world.
- Arithmetic derived from numbers IN the context (e.g. 160000 - 100000 = 60000)
  is SUPPORTED, not a hallucination. Additional correct, cited detail that does
  not contradict the context is SUPPORTED.
- Refusing to answer when the context is genuinely insufficient is SUPPORTED (1).
- Refusing to answer when the context DOES contain the answer is UNSUPPORTED (0):
  the refusal claim "I don't have enough information" is itself false.
- A partial answer that states the supported part and declines the rest is
  SUPPORTED, provided each stated fact is in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [..], "reason": "one sentence"}}
"""


def judge_faithfulness(answer_text: str, context: str) -> int | None:
    verdict = llm_judge(IMPROVED_RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier="LARGE", max_tokens=2048)
    if verdict.get("parse_error"):
        return None  # missing data, not a failing answer -- exclude, don't score 0
    return int(verdict.get("score", 0))


IMPROVED_RUBRIC_CORRECTNESS = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

Score 2 = same substantive content as the reference (wording may differ).
          The candidate may include ADDITIONAL correct, cited detail beyond
          what the reference states -- that is NOT a reason to lower the
          score. Only lower the score if the extra material is itself
          wrong, or actually CONTRADICTS the reference.
Score 1 = partially correct: some correct content, but OMITS something the
          reference states as necessary to answer the question, or adds
          something that genuinely CONTRADICTS the reference (not merely
          additional, non-conflicting detail).
Score 0 = wrong, or refuses when the reference answers.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def judge_correctness(question: str, candidate: str, reference: str) -> int | None:
    candidate_refused = candidate.strip() == REFUSAL
    # Gold answers use a label prefix: "REFUSE. ..." for fully unanswerable
    # questions, "PARTIAL REFUSE. ..." for the one (Q37) needing a partial
    # answer. Only the full case is mechanical -- a full refusal is either
    # exactly right or exactly wrong. The partial case needs judgement, so
    # it falls through to the LLM judge below.
    reference_says_refuse = reference.strip().upper().startswith("REFUSE.")

    if reference_says_refuse:
        return 2 if candidate_refused else 0
    if candidate_refused:
        return 0  # should have answered (even if only partially); a full
                  # refusal here is wrong, not a safe default

    verdict = llm_judge(IMPROVED_RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE", max_tokens=2048)
    if verdict.get("parse_error"):
        return None
    return int(verdict.get("score", 0))

# ---------------------------------------------------------------------------
def run_full(save: str = "") -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label="lab4-full") as b:
        for q in questions:
            a = answer_question(q["question"], retriever)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else float("nan")

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {_mean([r['citations_valid'] for r in rows]):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {_mean([r['faithfulness'] for r in rows]):.3f}"
          f"  (n_scored={sum(1 for r in rows if r['faithfulness'] is not None)}/{len(rows)};"
          " parse_error=None excluded, not scored 0)")
    print(f"correctness (0-2)   {_mean([r['correctness'] for r in ans]):.3f}"
          f"  normalised {_mean([r['correctness'] for r in ans]) / 2:.3f}"
          f"  (n_scored={sum(1 for r in ans if r['correctness'] is not None)}/{len(ans)})")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({sum(1 for r in refusals if r['unanswerable'])}/{len(refusals)} refusals total;"
          f" {len(refusals)} refusals total)")
    print("\n" + b.report())
    print(f"cost per query      ${b.spent_usd / len(rows):.4f}   (target <= $0.01)")

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r["correctness"] for r in ans if r["kind"] == kind and r["correctness"] is not None]
        print(f"  {kind:<16} {_mean(sub)/2:.3f}"
              f"  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_gold_context() -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            s = judge_correctness(q["question"], a.text, q["gold_answer"])
            if s is not None:
                retrieved_scores.append(s / 2)
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            s = judge_correctness(q["question"], g.text, q["gold_answer"])
            if s is not None:
                gold_scores.append(s / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\nn_gold={len(gold_scores)} n_retrieved={len(retrieved_scores)}"
          " (parse_error=None excluded)")
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        h = [r[f"human_{field}"] for r in human if r[f"human_{field}"] is not None]
        m = [machine[r["id"]][field] for r in human if r[f"human_{field}"] is not None]
        if not h:
            print(f"{field}: no human labels yet")
            continue
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.calibrate, a.kappa]):
        ap.print_help()
