# Lab 5 — RAG v2: Diagnose, Fix, Prove — Report

## Part A — Failure tally (17 failures of 45) and Pareto

`python labs/lab5/diagnose.py --input reports/lab4.json` implements the T4 §5 tree: mode 7 → mode 1 → gold-context split (fixes = retrieval branch, still-wrong = 6) → chunk-rank subdivision (in final = 6-distractor; top-30-only = 4; verbatim probe → 3, else 2-eyes). Correctedness < 2 or invalid citations defines failure (17: nine corr-0 refusals, eight corr-1 partials; citation validity 1.00 so mode 7 = 0).

```
failure mode          n    share   cumulative
generation            13   76.5%    76.5%  ███████████████████████
ranking                3   17.6%    88.2%  █████
embedding_mismatch     1    5.9%   100.0%  ██
```

Modes 1/2/5/7 = 0 (no reranker; nothing missing from the corpus). Two modes cover 94% — concentrated, as expected. Split of the 13: 9 distractor/conflict overload (gold chunk WAS in final-5; clean gold context scores 2: Q19, Q22, Q28, Q29, Q31, Q32, Q35, Q41, Q43) and 4 pure generation (gold still fails: Q05, Q11, Q20 judge-capped partials; Q40 full-refusal-vs-partial). Ranking: Q04 (proxy r21), Q23 (r17), Q44 (r8). Embedding: Q37 only.

`answer_in_corpus` was improved (strip REFUSE./PARTIAL prefixes; numbers gate with comma normalisation; stopword filtering). Validation: True on all 40 answerable golds except Q33 (compact "30/60" style — noted limitation, Q33 is corr-2 so harmless) and correctly False on Q36/38/39 (no relevant docs).

**A2 human checks.** The proxy-chunk heuristic needed two overrides, both documented in `reports/lab5_diagnosis.json`: Q29 script-proxy m0 (doc header) → true m4 (45-day rule, rank 2, in final; motor-scope distractor caused it); Q31 script-proxy m3 (a *different* 15-day rule — number collision) → true m9 (15-working-days settlement, rank 1, in final; archived conflict caused it). Lesson: number matching without rule-identity verification misattributes; headers outscore rules on word overlap. Q37 proxy m4 verified correct (the outside-India sentence); verbatim probe retrieves `exclusions` at rank 1 while the question ranks it outside top-30 under dense AND hybrid — clean mode 3.

## Part B — Ranking by expected value

| Cluster | n | Fix | Est. recovery | Cost Δ | Latency Δ |
|---|---|---|---|---|---|
| G-distractor, archived-caused (Q31) | 1 certain | Exclude `claims-timelines-2024-ARCHIVED` at ingest | 1 (Q31 0→2) | 0 | 0 |
| G-distractor, scope-caused (Q29 motor, Q41 life) | 2 | Scope-guard prompt (ignore other product lines) | 1–2, uncertain | 0 | 0 |
| R-rank (Q04 r21, Q23 r17, Q44 r8) | 3 | final_k 5→10 | 1 (Q44 only) | ~+60% context tok | small |
| R-embed (Q37) + Q23 | 1–2 | Hybrid BM25+dense (RRF) | 1 (Q23 r5; Q37 unproven) | ~0 | +1 ms |
| G-pure (Q05, Q11, Q20, Q40) | 4 | Prompt completeness/partial rules | ≤1 (Q40 only; rest judge-capped: gold still scores 1) | 0 | 0 |

**Pick, one sentence:** Archived exclusion is the only candidate with a certain mechanism, zero cost/latency delta, Lab 3 D3 precedent (Q29–31 hit_rate@1 0.667→1.000), and a blast radius of exactly 3 questions (only Q01/Q30/Q31 ever retrieve the doc).

**Prediction (written before `--compare` ran):** *"Archived exclusion will recover 1 of the 17 failures (Q31: 0→2), with Q01 and Q30 unchanged at 2 and no other question changing score."*

## Part C/D — Before/after (`--compare`, same 45 questions, all Lab 4 metrics)

| Metric | v1 | v2 archived-filter | Δ |
|---|---|---|---|
| Correctness (norm) | 0.725 | **0.750** | **+0.025** (meets Lab 4 target) |
| Faithfulness | 0.978 | 1.000 | +0.022 (Q01 judge-0 resolved: extra archived detail gone) |
| Citation validity | 1.000 | 1.000 | 0 |
| Refusal recall / precision | 1.000 / 0.417 (5/12) | 1.000 / 0.455 (5/11) | 0 / +0.038 |
| hit_rate@1 / @5, recall@5, MRR, nDCG@10 | 0.786 / 0.976 / 0.891 / 0.877 / 0.831 | 0.810 / 0.976 / 0.891 / 0.889 / 0.839 | all ≥0 |
| Cost/query | $0.0000* | $0.0006 | +$0.0006 (≤2× ✓) |
| p95 latency | 0* / real ~3.5 s (Lab 4) | 2192 ms | within budget |

\*v1 fully cache-hit. Per-question diff: exactly Q01/Q30/Q31 changed — Q31 refusal→correct 15-working-days answer (0→2) ✓; Q01/Q30 stay 2 (Q30's answer got cleaner: conflict note gone, 72 h stated directly). **Prediction confirmed exactly: 1 recovery, 2 protected, 0 other moves.**

**D2 regression check (v2): nothing got worse.** No question dropped score; no kind regressed; recall held (filtering did not teach the system to answer unanswerables — the removed doc was never load-bearing for a correct refusal).

## The fix that did not work: hybrid BM25+dense (−0.037)

Second, alone, as the reflex-retrieval-fix control the handout warns about: correctness 0.725→**0.688 (−0.037)**, nDCG@10 −0.079, hit_rate@1 −0.119, precision 0.417→0.385 (+1 refusal), cost $0.0108/query (**2.1× baseline — violates the ≤2× discipline**). Improved 3 (Q28 0→2, Q31 0→2, Q32 0→1) but worsened 6 (Q20 1→0, Q21 2→0, Q22 1→0, Q26 2→1, Q33 2→1, Q45 2→0). Notably Q23 — the one case the retrieval-only probe predicted hybrid would save (proxy r17→r5) — did **not** recover: probe used the k=30 pool while the pipeline reranks a k=12 pool, so the probe overpromised. Mechanism: on this corpus dense alone already outranks BM25 almost everywhere (Lab 3 B5), so fusion imports BM25's losses more often than its wins. Fix and tally agreed (tally said generation/distractor, not lexical mismatch) — the tally was right.

## D3 — Re-classification of the 16 survivors

Shape unchanged, as predicted: generation 12 (4 pure + 8 distractor), ranking 3, embedding 1 — Q31 simply left the distractor cluster; no failure migrated modes (the fix removed a failure instead of revealing a masked one, since the archived chunk is gone from every context rather than replaced).

## D4 — Next fix

Prompt scope-guard + synthesis licence ("ignore motor/life/travel/group-corporate chunks unless asked; combining two chunks is synthesis, not guessing; partial answers earn credit") targeting the remaining wrongful refusals with in-context answers (Q28, Q32, Q41, Q43) and Q40's partial. Worth an estimated 2–3 recoveries at zero cost; Q05/Q11/Q20 are deliberately excluded (gold-capped — no prompt recovers them under the current judge).
