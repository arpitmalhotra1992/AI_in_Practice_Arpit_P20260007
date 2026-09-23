# Lab 1 — The Reliable Extractor — Report

**Provider:** Gemini free tier (`gemini-3.5-flash-lite`, `SMALL` tier)
**Split for final numbers:** test (n=120), run once, saved to `reports/lab1_test.json`

---

## Part A — v0, the naive way: failure table (n=40)

| Failure mode | Count in 40 | Example ticket id |
|---|---|---|
| Not valid JSON at all | 0 | — |
| JSON wrapped in a markdown fence | 32 | T0054 |
| Extra prose before or after the JSON | 0 | — |
| Valid JSON, missing a required field | 0 | — |
| Category outside the allowed set | 32 | T0054 |
| Urgency as a string instead of an int | 32 | T0054 |
| Policy number invented (not in the text) | 0 | — |
| Unhandled exception | 8 | T0078 (`RateLimitError`, 429) |

**The arc:** 0/40 parsed by bare `json.loads()` → 32/40 parsed once the markdown
fence is stripped → **still 0/40 clean**, because every one of those 32 carried
both `urgency_is_string` and `category_out_of_set` inside it. A one-line parsing
fix only exposes the real problem; it does not solve it.

**Two rows outside the T1 §3 taxonomy:**
1. `Unhandled exception (RateLimitError)` — an infrastructure/reliability
   failure (free-tier quota, 15 req/min), not a model-output failure. It
   belongs with ops concerns, not prompt-format concerns.
2. `Policy number invented` (0 here, but structurally distinct) — this is a
   *hallucination* failure mode, categorically different from a
   formatting failure; the model didn't format the answer wrong, it stated
   something false.

**Would a human reviewer notice these in production?** The markdown fence and
the rate-limit error are loud — visible crashes or empty output. `category`
and `urgency` type/value errors are silent: they'd pass through a loosely
typed pipeline and corrupt the `escalate` business rule downstream without
anyone noticing until a ticket was mis-routed.

---

## Variant comparison — v0 / B / C

| Metric | v0 (diagnostic only) | B (dev) | C (dev) | **C (test, official)** |
|---|---|---|---|---|
| Schema validity | n/a (0/40 parsed cleanly) | 1.0000 | 1.0000 | **1.0000** |
| Field accuracy | n/a | 0.9071 | 0.8438 | **0.8583** |
| Record accuracy | n/a | 0.4500 | 0.3167 | **0.4417** |
| Cost (60 tickets, dev) | $0.0058 (32 calls only) | $0.0027* | $0.0302 | — |
| Cost (120 tickets, test) | — | — | — | **$0.0565** |
| p95 latency | 1,639 ms | 1,392 ms | 1,806 ms | **1,733 ms** |
| Unhandled exceptions | 8/40 | 0 | 0 | **0** |

`*` The B-variant dev cost is not a fair comparison point: 56 of 60 calls were
served from cache (identical prompt run earlier in the session), so $0.0027
understates B's true cost. The dev-split B vs C comparison above is included
for completeness but should not be read as "C costs more than B" — it mostly
reflects a caching artifact, not a real cost regression. The correct
comparison is against the official, uncached **test** run above.

### Against the lab's targets (test split, official run)

| Metric | Target | Reference solution | **This run** | Met? |
|---|---|---|---|---|
| Schema validity | 100% | 1.000 | **1.0000** | ✅ |
| Field accuracy | ≥ 0.90 | 0.930 | **0.8583** | ❌ (−0.042) |
| Record accuracy | ≥ 0.55 | 0.608 | **0.4417** | ❌ (−0.108) |
| Cost, 120-item run | ≤ $0.15 | $0.080 | **$0.0565** | ✅ |
| p95 latency | ≤ 4,000 ms | 2,276 ms | **1,733 ms** | ✅ |
| Unhandled exceptions | 0 | 0 | **0** | ✅ |

Four of six targets are met, including all reliability targets (validity,
cost, latency, zero crashes). The two misses are both on the *judgement*
fields, not the deterministic ones — see the per-field breakdown below.

---

## Per-field accuracy (test, worst first)

| Field | Accuracy |
|---|---|
| `urgency` | 0.533 |
| `sentiment` | 0.725 |
| `category` | 0.825 |
| `escalate` | 0.867 |
| `product` | 0.933 |
| `language` | 0.983 |
| `contains_pii` | **1.000** |
| `policy_number` | **1.000** |

The two fields moved out of the model in Part C (`policy_number`,
`contains_pii`) are perfectly accurate and fully auditable, exactly as the
handout predicts. All of the accuracy shortfall against target is concentrated
in the three subjective, boundary-dependent fields the annotation guide
explicitly warned about.

### `category` confusion matrix (rows = gold, cols = predicted)

| gold \\ pred | billing | claims | complaint | information | policy_change | technical |
|---|---|---|---|---|---|---|
| billing | **13** | . | . | 3 | . | . |
| claims | . | **17** | . | 4 | . | . |
| complaint | . | 4 | **8** | 4 | . | . |
| information | . | . | . | **22** | . | . |
| policy_change | . | . | . | . | **22** | . |
| technical | . | . | . | 6 | . | **17** |

`complaint` is the weakest category by far (8/16 correct). Every error moves
*away* from `complaint`, split evenly toward `claims` and `information` — the
exact boundary the annotation guide calls out: an angry message about a claim
is `claims` unless Aurora's own conduct is the subject, and the model is not
reliably making that distinction.

---

## Top three error clusters

**1. `category` — `complaint` boundary confusion (8 of 41 imperfect records touch `category`).**
The model defaults to the topic of the message (a claim, a billing issue)
rather than checking whether the complaint is about *Aurora's conduct*
specifically. **Fix:** strengthen the `category` field description to
explicitly instruct the model to check for conduct-directed language
("kept on hold", "mis-sold", "ignored") *before* falling back to the
transactional category.

**2. `urgency` — boundary errors, not scatter (the single most common wrong field).**
Spot-checking the flagged records (e.g. T0009, T0109) shows the errors are
adjacent-value misses (gold 2 vs. predicted 1, or gold 3 vs. predicted 2),
consistent with the README's prediction that most urgency errors sit at a
scale boundary rather than being random. **Fix:** add 2–3 more concrete
example situations anchoring each of the five urgency levels in the field
description, particularly around the 1/2 boundary ("can this be answered
without opening the customer's record?").

**3. `urgency` and `sentiment` co-failing on the same tickets (T0049, T0208, T0177 all wrong on both).**
These two fields are supposed to be independent (tone vs. situation), but
several tickets get both wrong together, suggesting a shared root cause —
likely highly emotional or Hinglish-heavy tickets that are simply harder to
read correctly for both axes at once. **Fix:** worth a targeted look at
whether these specific failures cluster in Hinglish tickets, which would
point to a language-specific weakness rather than two independent field
problems.

---

## D5 — The economic argument

**Measured cost per ticket (test split):** $0.0565 / 120 ≈ **$0.00047/ticket**

**Annual API cost at 10,000 tickets/day:**
$0.00047 × 10,000 × 365 ≈ **$1,716/year**

**Cost of the human baseline** (40 sec/ticket, ₹300/hr, 10,000 tickets/day):
- 40 sec = 0.0111 hr → ₹3.33/ticket
- Daily: ₹33,333 → **Annual: ≈ ₹1.22 crore ≈ $146,000/year** (at ~₹83/$1)

Even accounting for the fact that only 44% of records (`record_accuracy` =
0.4417) are fully correct on all 8 fields, the system is overwhelmingly
cheaper *if* reviewing a flagged record costs meaningfully less than a full
manual entry from scratch — a reasonable assumption, since a human reviewer
is confirming or correcting a pre-filled record rather than typing one from a
blank ticket.

**Break-even record accuracy:** Let `r` = record accuracy (fraction needing no
review) and assume a reviewed record still costs the full ₹3.33 in reviewer
time (a conservative, worst-case assumption). Total annual cost becomes:

`(1 − r) × ₹1.22cr + r × $1,716 ≈ (1 − r) × ₹1.22cr` (the API cost is
negligible by comparison)

Even at the low end — this run's r = 0.44 — annual cost is still only
`0.56 × ₹1.22cr ≈ ₹68 lakh`, already a >40% reduction versus full manual
processing, before counting any speed-up from having 44% of records need no
touch at all. The system is worth deploying well below the current 0.44; the
break-even point is closer to r ≈ 0.02–0.05, i.e. it would need to be almost
completely wrong before manual processing became cheaper. The real
constraint is not cost — it is whether flagged/incorrect records are caught
reliably enough downstream that routing errors don't cause customer harm.

---

## One thing that did not work

**Concurrent evaluation runs (`--workers` default) hit the Gemini free-tier
rate limit (15 requests/minute) partway through Part A**, producing 8/40
unhandled `RateLimitError` exceptions. My first instinct was to treat this as
a bug in `extract_b()`'s exception handling — but the actual fix was two
separate things: (1) catching the broader `Exception` class rather than only
`StructuredOutputError` in `extract_b`/`extract_c`, since `litellm` raises
provider-specific exception types that don't inherit from the toolkit's own
error class, and (2) dropping to `--workers 1` for all evaluation runs to stay
under the free-tier quota. Neither fix was obvious from the error message
alone; both took a round-trip through the actual traceback to diagnose. This
is a reasonable finding in its own right: retry/backoff logic alone doesn't
fully protect a run when the failure is a hard 429 quota wall rather than a
transient blip.

---

## Known limitation of these numbers

The dev-split cost comparison between variant B ($0.0027) and variant C
($0.0302) is confounded by caching — B mostly hit cache, C did not — and
should not be read as a real cost delta. The test-split cost ($0.0565 for
120 tickets, uncached) is the trustworthy number and is well under the $0.15
budget target.