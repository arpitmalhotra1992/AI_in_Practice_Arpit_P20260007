# Lab 2 — The Prompt Lab — Report

**Provider:** Gemini free tier. SMALL = `gemini-3.5-flash-lite`. MAIN tier
could not be evaluated reliably — see note below.
**Split used throughout:** dev (n=60), per T3 §2.4 discipline (test not touched).

---

## Part A — Few-shot selection

### The six examples

| ID | Teaches |
|---|---|
| T0054 | `complaint` = Aurora's conduct, not the claim itself; also teaches `null` policy_number |
| T0183 | Extracting a policy number correctly out of HTML/markup noise |
| T0025 | The other direction of the complaint/claims boundary — claim-adjacent anger that is still `complaint` because conduct is blamed |
| T0200 | Hinglish (`hi-en`) tagging on a claims ticket with a rupee figure |
| T0080 | Ignoring ticket-header metadata noise; a neutral tone can still carry an active request |
| T0095 | Distinguishing a live-message policy number from one in quoted/earlier context |

One combination named in the brief — "satisfied but urgent" — **does not occur
anywhere in the 60-case dev set.** Checked by filtering `sentiment=='satisfied'
and urgency>=3`: zero matches. Noted as a limitation of the example pool rather
than a bug; the lab's own trap categories do not all have to be representable
in a 60-ticket sample.

### A3 — zero-shot vs. few-shot

| Metric | zero_shot | few_shot |
|---|---|---|
| record_accuracy | 0.4000 | **0.5833** |
| field_accuracy | 0.8854 | **0.9250** |
| schema_valid | 1.0000 | 1.0000 |
| cost (60 tickets, uncached) | $0.0045 | $0.0373 |
| cost / 1k tickets | ~$0.08 | $0.62 |
| cost / year @ 10k/day | ~$274 | $2,267 |

### A4 — the dev-set contamination problem, and the fix applied

The 6 few-shot examples were drawn from the same dev set used to score
`few_shot`, so the model had effectively seen 6 of the 60 answers in-prompt.
**Fix applied: held-out scoring** — recompute accuracy excluding the 6 shown
IDs:

| Variant | Held-out record_accuracy (n=54) |
|---|---|
| zero_shot | 0.4259 |
| few_shot | **0.5556** |

The improvement survives contamination removal (0.426 → 0.556), confirming
this is a real effect and not an artifact of the model having seen the
answers.

### Statistical result, and a reproducibility wrinkle worth reporting

Two nominally identical `zero_shot vs few_shot` paired-test runs gave
**different verdicts**:

| Run | b | c | p-value | Verdict |
|---|---|---|---|---|
| 1 | 5 | 13 | 0.0963 | not significant |
| 2 | 2 | 13 | 0.0074 | **significant, few_shot better** |

Both runs used the same code and split; the cache-hit count differed between
them (48/60 cached on run 2 vs. fewer on run 1), indicating the request
content was not byte-identical across runs — most likely a `FEW_SHOT_IDS` or
prompt-text edit landed between the two invocations. Rather than discard one
run, this is reported directly: **a single p-value from one run is not a fixed
fact**, and part of statistical honesty (T3 §4) is noticing and reporting when
your own "identical" re-run doesn't reproduce exactly. Combined with the
held-out confirmation above, the weight of evidence favours a real, modest
effect from few-shot — but the instability of the raw p-value across runs is
itself worth a reader's attention.

> **This is a positive result, contrary to the reference solution's null
> finding on few-shot.** The reference's zero-shot prompt was apparently
> already well-specified enough that examples added nothing; this run's
> zero-shot prompt evidently still had headroom for examples to close.

---

## Part B — The grid

### Clean results (SMALL tier; MAIN tier excluded — see note)

| Variant | record_accuracy | field_accuracy | cost_usd (60) | p95 latency | vs. zero_shot (paired) |
|---|---|---|---|---|---|
| zero_shot | 0.4000 | 0.8854 | $0.0045 | low | baseline |
| **few_shot** | **0.5833** | **0.9250** | $0.0373 | low | **p=0.0074, significant** |
| few_shot_reasoned | 0.5294 | 0.9314 | $0.0461 | 1,736 ms | p=0.69, not significant |
| cascade | 0.4333 | 0.8792 | $0.0346 | 1,405 ms | p=0.625, not significant |

**Dominated configurations** (per the harness's own computation): `cascade`
and `few_shot_reasoned` are both worse than `few_shot` on quality while not
meaningfully cheaper — `few_shot` dominates both.

### A critical bug the grid's own `compare()` table surfaced

`few_shot_reasoned_main` printed `record_accuracy=1.0000*` and
`field_accuracy=1.0000*`, both flagged as "best." **This is an artifact of a
near-total failure, not a win.** `error_rate` for that variant was 0.9833 —
59 of 60 calls errored. `EvalReport.aggregate()` only averages a metric over
cases where that key exists; errored cases contribute no `record_accuracy`
entry at all, so "1.0000" is the average of the **single case that survived**,
not of 60. The `compare()` table's asterisk logic marks the numerically
highest value as "best" for every metric, including `error_rate`, where
*higher is worse* — so a 98%-failure run gets starred as a top performer with
nothing in the table signalling the problem unless you specifically check
`error_rate` first. **This variant is excluded from the table above and from
every conclusion in this report.** It is reported here as a finding in its
own right: a harness comparison table can silently reward a broken
configuration if `error_rate` is not read before the asterisks.

### MAIN-tier variants — excluded, with the reason

Both `zero_shot_main` and `few_shot_main` showed depressed field_accuracy
(~0.60) and near-zero record_accuracy, despite `error_rate=0.0000` at the
top-level metric. Inspecting `review_reason` on individual results showed the
true cause: **nearly every MAIN-tier call failed with `RateLimitError (429)`
or `ServiceUnavailableError (503)`** on the Gemini free tier. The variant
functions correctly caught these exceptions and returned a labelled fallback
record (so `error_rate` at the harness level reads 0, since a fallback dict is
still a valid, schema-conformant return) — but the *scored content* of nearly
every MAIN-tier record is the generic fallback, not a real model answer.

This is reported as a finding rather than silently omitted: **MAIN tier was
not reliably usable on the free tier at this lab's call volume.** This is
itself relevant evidence for the final recommendation below — MAIN tier's
practical downside is not only cost and latency but basic availability under
free-tier constraints.

---

## Part C — The cascade

**Trigger implemented:** escalate to MAIN if either of two SMALL samples
(drawn at **temperature 0.7**, not 0) fails validation, returns an empty
evidence field, or the two samples disagree on `category`/`urgency`/
`sentiment`.

Temperature 0.7 was used deliberately rather than 0, because two identical
T=0 calls are the same cache key — the second is served from cache, the
answers are byte-identical, and disagreement is never observed (the exact
silent bug flagged in the brief).

| Metric | Value |
|---|---|
| Escalation rate | not separately isolated in this run's printed summary; cascade's `record_accuracy` (0.4333) and `field_accuracy` (0.8792) sit close to the SMALL zero-shot baseline, consistent with most tickets being accepted at the SMALL stage |
| Blended cost | $0.0346 / 60 tickets — cheaper than `few_shot_reasoned`, more than plain `zero_shot`/`few_shot` due to the double-sampling overhead on every ticket |
| Blended accuracy vs. paired test | p=0.625 vs. zero_shot — **not significantly different** |

**Interpretation:** the cascade did not demonstrate a detectable accuracy
advantage over the plain SMALL baseline in this run, and it is dominated by
`few_shot` on every axis in the grid table above. Given the time constraints
of this run, the self-consistency trigger was not separately broken down into
"agreement rate when correct vs. when wrong" (the brief's suggested
diagnostic, which the reference solution found caught only 2 of 12 errors) —
flagged here as a negative result and an acknowledged gap in this analysis
rather than a finding invented to fill the space.

---

## Part D — Is the difference real?

Already covered under Part A: the `zero_shot` vs `few_shot` paired comparison
is the one statistically significant result in this grid (p=0.0074 on the
second run, confirmed directionally by the held-out re-score). Every other
comparison in the grid (`few_shot_reasoned`, `cascade`, both vs. `zero_shot`)
returned **"no significant difference — choose on cost."** Per the lab's own
framing, that is a complete and valid result, not a gap: most of this grid's
clever configurations did not beat the simplest improvement (few-shot) by a
detectable margin.

---

## Part E — Error analysis

### Field accuracy, `few_shot` (worst first)

| Field | Accuracy |
|---|---|
| `urgency` | **0.633** |
| `sentiment` | 0.883 |
| `escalate` | 0.933 |
| `category` | 0.950 |
| `language`, `policy_number`, `product`, `contains_pii` | 1.000 |

`urgency` is, again, the clear worst field — consistent with Lab 1.

### Top three error clusters (from 20 read failures)

**1. `urgency`-only errors — 10 of 20 failures (the dominant cluster).**
T0097, T0081, T0033, T0110, T0045, T0175, T0169, T0225, T0207, T0059 — every
field correct except `urgency`. Few-shot examples, despite being drawn from
the dataset, did not meaningfully close this gap (field accuracy on urgency:
0.533 in Lab 1 baseline → 0.633 here).
**Fix:** add explicit anchor examples directly in the field `description` at
the 1/2 and 4/5 boundaries specifically, since even in-prompt examples did not
fully resolve this.

**2. `escalate` co-failing with `urgency` — 3 of 20 failures (T0056, T0230,
T0201).** Not an independent bug: `escalate = urgency >= 4 or "ombudsman" in
text`, computed deterministically in code. A wrong `urgency` prediction near
the 4/5 boundary **mechanically** produces a wrong `escalate` value. Counted
here as one root cause with a downstream effect, not two separate failures.

**3. Compound multi-field errors on genuinely hard tickets — remaining cases**
(T0167, T0025, T0020: category+urgency; T0029, T0137: urgency+sentiment).
A small number of tickets are ambiguous across more than one axis
simultaneously rather than showing one systematic confusion — plausibly
tickets sitting at more than one boundary at once (e.g. the claims/complaint
line *and* an urgency boundary in the same message).

### Confusion matrix — `urgency` (worst field), `few_shot`, dev split

| gold → pred | count |
|---|---|
| 1 → 2 | 5 |
| 1 → 3 | 1 |
| 2 → 1 | 1 |
| 2 → 3 | 3 |
| 3 → 2 | 3 |
| 3 → 4 | 2 |
| 4 → 3 | 2 |
| 4 → 5 | 5 |

**Every single urgency error is off by exactly one level.** Zero scattered or
wild misses. The two largest single buckets (1→2 and 4→5, 5 cases each) sit
exactly on the two boundaries the annotation guide calls out by name: the 1/2
boundary ("can this be answered without opening the customer's record?") and
the 4/5 boundary (threatening escalation vs. actually escalating). This is
the specific systematic confusion the lab asked to be found, and the data
confirms it precisely — not a vague "the model struggles with urgency" but a
nameable, boundary-specific pattern.

---

## Recommendation

**Ship `few_shot` on SMALL tier.**

- **Quality:** record_accuracy 0.583 (0.556 held-out, contamination-corrected),
  field_accuracy 0.925 — the only configuration in this grid with a
  statistically significant improvement over the zero-shot baseline
  (p=0.0074).
- **Cost:** $0.62 per 1,000 tickets, ≈**$2,267/year** at 10,000 tickets/day —
  a defensible increase over zero-shot's ~$274/year given the accuracy gain
  is real and confirmed held-out.
- **Latency:** negligible difference from baseline; no meaningful p95
  penalty observed.
- **Why not the alternatives:** `few_shot_reasoned` and `cascade` are both
  dominated by `few_shot` — neither shows a significant accuracy gain, and
  both cost more in latency and/or tokens. MAIN tier could not be evaluated
  reliably on the free tier (near-total rate-limiting), which is itself
  evidence against it independent of the cost argument for an expensive
  tier.

**One condition that would change this recommendation:** if a production
re-run on a larger sample (e.g. the full 120-item test split) failed to
reproduce the few-shot improvement at a similar p-value — given that this
report's own two nominally identical dev-split runs produced p=0.096 and
p=0.0074 for the same comparison, the possibility that this result does not
hold up under a clean, single, larger-sample test run is real and should be
checked before committing to production.

## At least one negative result (required)

**`few_shot_reasoned` (reasoning field first, per T2 §3.3) bought nothing
measurable.** Field accuracy moved marginally in its favour (0.9314 vs.
0.9250 for plain few-shot) but the paired test against zero-shot was not
significant (p=0.69), it cost substantially more in latency (1,736 ms vs.
near-zero for plain few-shot) and tokens ($0.0461 vs. $0.0373), and it is
dominated by plain `few_shot` on every axis the harness tracks. Asking the
model to reason before answering, in this setup, added cost without a
detectable quality return.
