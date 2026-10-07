# Evaluation Report — Aurora Policy Assistant (RAG service, Labs 3–7)

## 1. What it does

Aurora's helpdesk staff and customers ask questions in plain English about health-insurance policies — waiting periods, claim windows, what a plan covers — and get short answers where every factual sentence cites the policy document it came from. When the documents do not contain the answer, it says so instead of guessing. It runs as an HTTP service with a demo UI, a cost/latency dashboard, and an automated test that blocks any change making answers worse.

## 2. How well it works (45 golden questions; gate thresholds in brackets)

| Metric | Value | Gate |
|---|---|---|
| Correctness | 0.750 (≥ 0.72) | pass |
| Faithfulness | 1.000 (≥ 0.95) | pass |
| Citation validity | 1.000 (≥ 0.98) | pass |
| Refusal recall / precision | 1.00 (5/5) / 0.455 (5/11) (≥ 0.80 / 0.40) | pass |
| Retrieval hit_rate@5 | 0.976 (≥ 0.90) | pass |
| Calibration | correctness κ 0.85; faithfulness 19/20 agreement (κ paradox, App. A) | — |

Weakest kinds: paraphrase 0.40, trap-archived 0.50. The gate replays the committed response cache offline (deterministic, $0) and fails the build on breach — demonstrated failing on `final_k=1` (correctness 0.613, hit-rate 0.810, exit 1).

## 3. Where it fails (16 remaining failures of 45)

Generation/distractor overload 8 (right chunk in context, wrong use: Q19, Q22, Q28, Q29-motor, Q32, Q35, Q41-life, Q43); generation-incomplete 4 (gold itself partial: Q05, Q11, Q20, Q40-full-instead-of-partial); ranking 3 (answer chunk ranks 8–21, outside top-5: Q04, Q23, Q44); embedding 1 (Q37: Singapore question never retrieves the exclusion docs). Worst: Q37 — needs a *partial* answer (benefit exists on Platinum, limit unknown) and the retriever returns nothing relevant, so it fully refuses. Lab 6: 17/17 attacks blocked at 0/4 false positives; two custom attacks survive everything — I06 (invented USD 50,000 Singapore limit stated with citation) and X03 (unauthenticated cross-customer record lookup).

## 4. What it costs

$0.0033/query live (gate replay $0). Per 1,000 queries ≈ $3.30. At 10,000 queries/day ≈ **$12,000/year**, dominated by generation tokens. Semantic cache threshold shipped at 0.990 (measured: true paraphrases score 0.66–0.67 while Silver-vs-Bronze confusables score 0.88+ — no threshold separates them, so only cosmetic variants may hit).

## 5. How fast it is

Cached p95 ≈ 0–8 ms (exact 0.1 ms; semantic one embed, ~1 s). Uncached: p50 4.6 s, **p95 8.9 s — misses the 6 s SLO**. Stage split (traces): generate p50 3.7 s (~98% of steady-state), retrieve 2 ms (cached embeds), validate ~0, no reranker. Streaming (`/ask/stream`, SSE + terminal machine-readable `validation` event because citations cannot be checked before completion): TTFT ≈ total on the reasoning MAIN tier (0 incremental token deltas measured) — the TTFT ≤ 1.5 s target is achievable only off the reasoning tier (SMALL-tier probe: first token 1.1 s).

## 6. What it is not safe for

Do not let it settle coverage or pay claims unsupervised. Reasons, each measured: 1 in 4 answers is wrong or incomplete (0.750); it over-refuses (precision 0.455 — 6 of 11 declines are answerable questions, so staff will route around it if declines are hard blocks); paraphrased questions score 0.40; a poisoned document stating a *plausible fact* with no instruction language is repeated as truth (I06 — no guard layer catches it); anyone can read anyone's policy record (X03 — no caller-identity check); the offline gate pins quality but not live cost/latency (replay is $0/0 ms by construction). Safe envelope: staff-facing draft answers with expandable citations a human checks, read-only tools, refunds always human-confirmed (three independent gates: schema cap, allowlist, confirmation).

## 7. What next (ranked by expected value)

1. **Generation-tier A/B (SMALL vs MAIN) on quality × TTFT** — SMALL streams (1.1 s TTFT) and may hold 0.75 correctness at ~⅓ cost; the gate decides. Worth ~3 s p95 + ~$8k/year if it holds.
2. **Scope-guard + synthesis licence in the answer prompt** — targets the 8 distractor refusals (est. 2–3 recoveries, $0), the largest measured cluster.
3. **Auth-bind policy lookup; freshness metadata on corpus docs** — closes X03 and forces I06-class poisoning to beat a date check instead of getting a free pass. Security, not quality-measured.

*Appendix A.* Faithfulness κ ≈ 0 at 19/20 agreement is the prevalence paradox (44/45 faithful ⇒ κ cannot reach 0.4); the judge was fixed instead (wrongful-refusal and arithmetic clauses; 0.956 → 1.000) and raw agreement reported. *Appendix B.* Lab journey: Lab 3 dense+markdown-400 (nDCG 0.853); Lab 4 over-refusal (5/14 precision); Lab 5 archived-filter fix (+0.025, prediction exact); Lab 6 guards; Lab 7 ships v2.
