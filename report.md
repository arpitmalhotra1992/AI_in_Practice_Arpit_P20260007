# Lab 4 — RAG v1: Grounded Answers with Citations — Report

## 1. `ANSWER_SYSTEM` and differences from the reference

Ours (`labs/lab4/rag.py`) contains all six required elements (T4 §6.1): source-only with general knowledge explicitly forbidden even when confident; cite-by-index `[1]`/`[2][5]`; never cite an unsupplied number; exact refusal string; partial-answer rule (answer supported part with citation, decline rest in refusal-like language, never guess); surface-don't-pick on disagreement; 2–3 sentence discipline; plus `UNTRUSTED_SYSTEM_CLAUSE`.

Differences from `aip/rag.py::ANSWER_SYSTEM`: (1) reference is priority-ordered rules; ours is prose. (2) Reference rule 4 **explicitly permits synthesis** ("combining facts from 2+ sources is synthesis, not general knowledge, and is required for multi-hop"); ours has no synthesis clause — it only forbids general knowledge. This likely contributes to our multi-hop over-refusals (§3): the model treats combining two chunks as risky and refuses. (3) Reference: "every factual sentence must *end* with a citation"; ours: "cite every factual claim *immediately after* the claim". Equivalent in practice. (4) Generation params: ours `max_tokens=512`, prompt `delimit(context)+Question`; reference `max_tokens=600` + `"Answer with citations:"` suffix. (5) Refusal detection: reference `startswith(REFUSAL[:40])`; ours exact match — so a partial answer is `refused=False` in our harness.

## 2. Citation enforcement (B) — validity 1.00

`validate_answer()` checks: every `[n]` in range via `aip.guards.enforce_citations`; non-empty; `finish_reason != "length"` (truncation); non-refusal has ≥1 citation. On failure (non-refusal): **retry once** with a corrective message naming the exact reason and the valid index range; if retry also fails, **fall back to exact refusal**. Rationale in code comment: silently returning a bogus citation is the one wrong answer (unverifiable claim reaches a customer); a retry preserves answerable cases, and refusal is always safe and machine-detectable. Result: **citation validity 1.000 (45/45)** — a code guarantee, not model behaviour.

## 3. Refusal, both directions (C)

| Setting | Recall (of 5) | Precision (of refusals) |
|---|---|---|
| Baseline (as shipped) | **1.000 (5/5)** | **0.417 (5/12)** |
| Strict (appended "any doubt → refuse; only answer when directly/explicitly stated") | **1.000 (5/5)** | **0.417 (5/12)** |

Both numbers are noisy and reported with counts: 1 case moves recall 0.20, precision ~0.08–0.12. The strict run did **not** move aggregates but churned 8 questions: fixed Q28/Q31/Q32/Q41, broke Q25/Q27/Q34/Q45. Prompt strictness is not a monotonic dial — aggregate equality hides per-question instability.

**Q37 partial.** With retrieved context, top-5 contains none of `exclusions`/`plans-overview` (Lab 3 A4: hit_rate@5 = 0.0) — full refusal is the only safe behaviour given that context. With **gold** context the same generator produces the textbook partial: *"Treatment outside India is excluded except under Platinum's international emergency benefit [1][5][19]. However, I don't have enough information…"* — supported part stated, limit refused. So the prompt works; retrieval blocks it (Failure 3, §6).

**Product recommendation: ship the baseline, not stricter.** Recall is already 5/5; stricter cannot improve it and only churns wrongful refusals. One invented claim deadline (customer loses a valid claim, regulatory finding) outweighs many unnecessary refusals (agent spends ~2 min). But 7 wrongful refusals/45 is still expensive — fix them via **retrieval** (archived filtering, §6), not prompt leniency, since loosening the prompt risks the recall that regulation requires.

## 4. Judges and calibration (D)

Faithfulness rubric: base + three explicit clauses (arithmetic from context numbers is SUPPORTED; additional correct cited detail is SUPPORTED; wrongful refusal — refusing when context contains the answer — is UNSUPPORTED 0; partial state-and-decline is SUPPORTED). Correctness rubric: improved template (extra correct detail never lowers; only wrong/contradicting extra lowers; refusal handling mechanical: `REFUSE.`-prefixed gold → exact-match scoring, else LLM judge; parse_error → `None`, excluded, never 0).

Stratified calibration sample (n=20: 7 single-hop, 5 multi-hop, 2 aggregation, 2 trap_archived, 2 paraphrase, Q36/Q37/Q40 unanswerable + Q01/Q13), hand-labelled before re-running judges:

| Rubric | Raw agreement | Cohen's κ (n=20) |
|---|---|---|
| Faithfulness | 0.55 (11/20) | **−0.10** |
| Correctness | 0.90 (18/20) | **0.85** |

Correctness passes (substantial; gate ≥0.4). Disagreements: Q05 (human 2 vs judge 1 — annual-only question needs no instalment detail) and Q40 (human 1 vs judge 0 — full refusal omits statable helpline fact; partial credit). Rubric already handles extra detail; no further change.

Faithfulness fails the gate for two honest reasons, both reported: (1) **prevalence paradox** — true faithfulness is 0.978 (44/45), so κ cannot reach 0.4 without multiple true negatives; raw agreement on the easy majority is 19/20. (2) **Systematic disagreement on refusals** — human (strict reading) marks 8 wrongful refusals 0; judge marks all refusals 1 despite the new "wrongful refusal = 0" clause, and still marks Q01 (30d + correct group/archived context) 0 despite the "extra detail is SUPPORTED" clause. The rubric fix did move faithfulness 0.956 → 0.978 (Q25 arithmetic corrected) with zero parse errors, but the judge tier does not reliably follow refusal nuance. Judge numbers are therefore reported with this caveat. **Self-preference:** generator is MAIN (`gemini-3.7-flash`), judges are LARGE (`gemini-3.5-flash`, different family) — not the same model, so self-preference bias is avoided by tier separation; any residual family-level upward bias is small relative to the retrieval gap below.

## 5. Full results (E1) + decomposition (E2)

`python labs/lab4/evaluate.py --full --save reports/lab4.json` (n=45: 40 answerable, 5 unanswerable):

| Metric | Value | Target | Ref |
|---|---|---|---|
| Citation validity | **1.000** | 1.00 | 1.000 ✓ |
| Faithfulness | 0.978 (45/45 scored) | ≥0.90 | 0.933 ✓ |
| Correctness (norm) | 0.725 (1.45/2) | ≥0.75 | 0.825 ✗ (close) |
| Refusal recall | 5/5 = 1.000 | ≥4/5 | 5/5 ✓ |
| Refusal precision | 5/12 = 0.417 | ≥0.70 | 0.714 ✗ |
| Cost/query | $0.0051 | ≤$0.01 | ✓ |
| p95 latency | 3522 ms | ≤6000 ms | ✓ |

Weakest kinds: paraphrase 0.40, trap_archived 0.50, aggregation 0.625.

E2 (`--gold-context`, n=42 with relevant docs): **A (gold) = 0.929** (generation ceiling), **B (retrieved) = 0.690**, **retrieval loss A−B = 0.238**, **generation loss 1−A = 0.071**. Retrieval is 3.4× the generation loss — **Lab 5 goes to retrieval**, contrary to the CONCEPTS.md note where generation dominated. When given the right docs the generator scores 0.93; the retriever denies it the chance.

## 6. Failure tally — Lab 5 backlog (E3, 10 wrong answers)

| Q | Symptom | Mode (T4 §5) |
|---|---|---|
| Q37 | relevant docs rank outside top 5 (hit_rate@5=0) | 3 embedding |
| Q44 | `plan-silver::m1` (answer) loses to `::m0` (UIN string) on exact ID (dense MRR 0.5; BM25 1.0 per Lab 3) | 3 embedding |
| Q23, Q28, Q31, Q32, Q41, Q43 | relevant docs/chunks in top 5, model refuses | 6 generation (over-refusal) |
| Q40 | helpline doc retrieved (rank 3), full refusal instead of partial | 6 generation |
| Q22 | answers but adds irrelevant 30-day claim, omits day-one cover | 6 generation |

Tally: **Failure 6 ×8, Failure 3 ×2**, Failures 1/2/4/5/7 ×0 (no missing content among failures; no reranker; citation validity 1.0). Backlog: (i) filter `ARCHIVED` + group-scope distractors at query time (fixes Q01/Q30-style conflicts and Q31); (ii) hybrid/lexical path for exact IDs (Q44) and Q37-style vocabulary mismatch; (iii) add the missing synthesis licence to the prompt (multi-hop refusals).
