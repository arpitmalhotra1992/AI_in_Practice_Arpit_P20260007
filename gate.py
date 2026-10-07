#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure() -> dict[str, float]:
    """TODO D1: run your golden set and return the metric dict.

    Keys must match thresholds.yml. Run under AIP_OFFLINE=1 so CI replays the
    committed cache and costs nothing. Uses the shipped v2 pipeline
    (archived-filter retriever + Lab 4 answering), identical prompts to the
    Lab 5 before/after run, so offline replay hits the committed cache.
    LAB7_GATE_FINAL_K env overrides final_k for the D3 breakage demo.
    """
    import os
    import statistics
    import time

    from aip.cost import Budget
    from aip.evals import retrieval_metrics
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import judge_correctness, judge_faithfulness
    from labs.lab4.rag import answer_question
    from labs.lab5.diagnose import build_v2_retriever

    final_k = int(os.getenv("LAB7_GATE_FINAL_K", "5"))
    questions = load_questions(include_unanswerable=True)
    retriever = build_v2_retriever()
    lat: list[float] = []

    from aip.retrieval import format_context
    rows = []
    with Budget(limit_usd=2.00, label="lab7-gate") as b:
        t_all = time.perf_counter()
        for q in questions:
            t0 = time.perf_counter()
            a = answer_question(q["question"], retriever, k=12,
                                final_k=final_k)
            ctx = format_context(a.hits)
            unans = not q["relevant_docs"] or q["kind"] == "unanswerable"
            seen, ranked = set(), []
            for h in a.hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    ranked.append(h.doc_id)
            rm = (retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
                  if q["relevant_docs"] else {})
            rows.append({
                "unanswerable": unans, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text,
                                                 q["gold_answer"]),
                "retrieval": rm, "relevant": q["relevant_docs"]})
            lat.append((time.perf_counter() - t0) * 1000)
        wall = (time.perf_counter() - t_all) * 1000

    def fmean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else float("nan")

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    ref = [r for r in rows if r["refused"]]
    s = sorted(lat)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    hr5 = [r["retrieval"].get("hit_rate@5") for r in rows
           if r["relevant"] and "hit_rate@5" in r["retrieval"]]
    return {
        "correctness": fmean([r["correctness"] for r in ans]) / 2,
        "faithfulness": fmean([r["faithfulness"] for r in rows]),
        "citation_validity": fmean([r["citations_valid"] for r in rows]),
        "refusal_recall": (sum(1 for r in una if r["refused"]) / len(una)),
        "refusal_precision": (sum(1 for r in ref if r["unanswerable"]) / len(ref)
                              if ref else 1.0),
        "hit_rate_at_5": statistics.fmean(hr5) if hr5 else float("nan"),
        "cost_per_query_usd": round(b.spent_usd / len(rows), 6),
        "p95_latency_ms": round(p95, 1),
        "wall_ms": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
