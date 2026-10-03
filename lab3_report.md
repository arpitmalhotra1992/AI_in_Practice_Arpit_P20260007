# Lab 3 — Semantic Search Report

**Corpus:** 30 documents (16 Aurora policy docs, 14 lexical distractors). **Golden set:** 45 questions; **3 excluded** (Q36, Q38, Q39 — no relevant document, so recall/nDCG are undefined for them), leaving **n = 42** for all retrieval metrics below. Five questions are of kind `unanswerable`; two of those five do have relevant documents and are included in n = 42 — the other three are the ones excluded above.

**Baseline** (sliding-800, dense, before any tuning): hit_rate@1 = 0.7857, recall@5 = 0.8452, MRR = 0.8451, ndcg@10 = 0.8053.

---

## Part A — Chunking

**A1 — four strategies at size = 800**

| Strategy | ndcg@10 | chunks |
|---|---|---|
| fixed | 0.7952 | 83 |
| sliding | 0.8053 | 91 |
| recursive | 0.8251 | 98 |
| **markdown** | **0.8458** | 164 |

Markdown-aware chunking wins clearly, carrying the heading path into each chunk's embedding.

**A2 — size sweep on markdown**

| Size | ndcg@10 | chunks |
|---|---|---|
| 400 | **0.8527** | 235 |
| 800 | 0.8458 | 164 |
| 1600 | 0.8075 | 150 |

**Not monotonic.** Going from 1600→800→400 improves quality, but it doesn't keep improving forever (it would at some point start splitting single rules across chunks and recall would fall). At 1600, a chunk embedding has to represent several unrelated rules at once — the vector sits *between* them, close to none (the dilution argument, T4 §2.2). 400 wins on this corpus; we did not probe below 400 to find where the curve turns back down.

**A3 — heading-path prefix, on vs off** (markdown-800)

| | hit_rate@1 | hit_rate@5 | MRR | ndcg@10 |
|---|---|---|---|---|
| WITH prefix | 0.7619 | 0.9762 | 0.8720 | 0.8458 |
| WITHOUT prefix | 0.6190 | **1.0000** | 0.7837 | 0.7915 |

The prefix **trades recall@5 for ranking quality**: it raises hit_rate@1, MRR and nDCG@10 substantially, but slightly lowers hit_rate@5. It tells the embedding what section the text belongs to, which sharpens *where in the list* the right chunk lands, without changing *whether* it's retrieved at all within 5. Report both halves — looking only at hit_rate@5 would make this look like a loss.

**A4 — a chunking failure**

Q37 ("Does Aurora cover treatment in Singapore, and up to what limit?") scored hit_rate@5 = 0.0 against relevant docs `['exclusions', 'plans-overview']` — none were retrieved in the top 5 even once. The top-5 hits were all topically adjacent but wrong (topup-and-super-topup, travel-insurance-exclusions, network-hospitals, senior-citizen-plan, plan-silver). This is a genuine retrieval gap, not an obvious chunking slice-up — flagged here as a limit of the current configuration rather than a fixed bug.

**Winning chunking config carried forward: markdown-aware, size 400.**

---

## Part B — Dense vs BM25 vs hybrid

**B1 — overall** (markdown-400)

| Config | hit_rate@1 | hit_rate@5 | recall@5 | MRR | ndcg@10 | p95 latency |
|---|---|---|---|---|---|---|
| dense | 0.7857 | 0.9762 | 0.9028 | 0.8800 | **0.8527** | 0.50 ms |
| bm25 | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | 0.48 ms |
| hybrid (RRF) | 0.6667 | 0.9762 | 0.8631 | 0.7976 | 0.7949 | 0.94 ms |

`hit_rate@5` is saturated (0.93–0.98 across all three) and reports almost nothing; MRR and ndcg@10 show the real spread.

**B2 — the mechanism (Q44 vs Q41)**

| | Q44 (exact identifier `AUR-HI-SIL-2026`) | Q41 (paraphrase, no lexical overlap) |
|---|---|---|
| dense MRR | 0.50 | 1.00 |
| bm25 MRR | 1.00 | 0.00 |
| hybrid MRR | 1.00 | 0.25 |

Dense embeds meaning, so it nails Q41 (*"skip paying on time"* ≈ *"grace period"*) but only half-ranks the rare literal token in Q44 — an unfamiliar ID string looks, to the embedding, much like any other ID string. BM25 is the reverse: term frequency × inverse document frequency makes a rare exact token score enormously in the one document containing it, but it finds zero lexical overlap on the paraphrase. **Hybrid rescues Q44 (0.50→1.00) but damages Q41 (1.00→0.25)** — fusing in BM25's opinion helps on the question BM25 is good at and hurts on the one it is bad at.

**B3 — RRF k sweep**

| k | ndcg@10 | hit_rate@5 |
|---|---|---|
| 10 | 0.8156 | 1.0000 |
| 30 | 0.7949 | 0.9762 |
| 60 | 0.7949 | 0.9762 |
| 100 | 0.7901 | 0.9762 |

k ≥ 30 is flat, as predicted — RRF's main design goal (not being sensitive to k) holds. k = 10 showed a mild edge over the rest; at n = 42 we did not chase this further, since a 0.02 spread here risks fitting noise rather than a real effect.

**B4 — unequal fusion weights:** weighting dense more heavily moved the fused score toward dense's own number but never matched dense run alone, and none of the weightings tested materially beat 1:1 by more than noise at this sample size.

**B5 — the headline finding: hybrid loses here.** Dense ndcg@10 = 0.8527 vs hybrid 0.7949. Dense beats BM25 on 20 of the questions where they differ; BM25 beats dense on only 6. T4 §4.3 calls hybrid "the strongest single change most RAG systems can make" — true on the corpora it was measured on, **false on this one**, because the embedding model here is strong enough on its own to handle most of what BM25 would otherwise rescue, so fusing in a substantially weaker retriever drags down more rankings than it saves. **Recommended retriever: dense alone.**

---

## Part C — Reranking

Base: dense, markdown-400, retrieve k=30 → rerank → final k=5.

| Config | ndcg@10 | hit_rate@1 | recall@5 | p95 latency |
|---|---|---|---|---|
| dense, no rerank | 0.8527 | 0.7857 | 0.9028 | 0.51 ms |
| dense + cross-encoder | 0.8174 | 0.7619 | 0.8889 | 106.6 ms |
| dense + LLM rerank | **0.8649** | **0.8810** | 0.8869 | 30,513 ms (≈29.4 s/query, 30 sequential calls) |

**C1 — cross-encoder hurts.** ndcg@10 drops (0.8527→0.8174) and hit_rate@1 drops too, for +106 ms. `ms-marco-MiniLM-L-6-v2` is trained on web-search query/document pairs and is out of domain on policy prose — it is actively mis-ranking, not just failing to help.

**C2 — LLM reranker is the best config measured.** ndcg@10 rises to 0.8649 and hit_rate@1 to 0.8810, but at ~29.4 s/query (30 sequential model calls per query, confirmed via wall time), it is unusable at interactive speed, and it is also the first configuration in the lab with a non-zero per-query dollar cost.

**C3 — decision, two different answers:**
- **Interactive agent-facing search box:** ship **dense, no rerank**. The cross-encoder actively *lowers* quality here, and the LLM reranker's ~29 s latency is far outside any usable response time — neither reranker earns its cost in this setting.
- **Overnight batch job** (e.g., regenerating stored FAQ answers): latency is free, so take the best-quality configuration regardless of speed — **dense + LLM rerank**, paying the per-call cost once, offline.

These answers differ because the two deployments have opposite constraints on the same resource (latency), and only one reranker configuration actually improves quality at any latency.

**C4 — a query reranking made worse.** Q32 ("Which plans have no co-payment?"): MRR under the cross-encoder dropped from 1.0000 to 0.2000. Diagnosis: the cross-encoder re-scores a topically-close-but-wrong chunk above the chunk dense had correctly ranked first — consistent with it being out of its training domain on insurance policy prose rather than web search content.

---

## Part D — Index and metadata

**D1 — exact vs HNSW** (markdown-400, dense, 235 chunks)

| | hit_rate@1 | recall@5 | ndcg@10 | p95 latency |
|---|---|---|---|---|
| exact (DenseRetriever) | 0.7857 | 0.9028 | 0.8527 | 0.51 ms |
| HNSW (ChromaRetriever) | 0.7857 | 0.9028 | 0.8527 | 2.55 ms |

**Identical quality**, confirming HNSW isn't sacrificing anything on this corpus — but HNSW is **~5× slower** at this scale. Per-query graph traversal plus Python-level overhead loses to a single BLAS matrix multiply when there are only hundreds of vectors to compare against. This is a benchmark-validity point, not a verdict against HNSW: at 235 chunks there is no scale for an approximate index to pay off against.

**D2 — scale test: not run.** `scripts/expand_corpus.py --docs 4000` (and the corresponding 40k-chunk timing) was skipped due to time constraints. We report only the ~235-chunk timing above and do not claim a crossover point; we expect, per T4 §3.2, that HNSW would overtake exact search somewhere in the low thousands of chunks, but this is not verified here.

**D3 — metadata filter on the archived-document trap.** Q29/Q30/Q31 each have a correct answer in `claims-timelines` and a wrong one in `claims-timelines-2024-ARCHIVED`. Tagging chunks with `status: current | archived` at ingest and filtering `where={"status": "current"}` at query time:

| | hit_rate@1 (Q29–Q31) |
|---|---|
| before filter | 0.6667 |
| after filter | **1.0000** |

This required **zero changes to the retriever itself** — it is a data/metadata fix, not a modelling fix, and it fully resolves the trap without deleting the archived document (which may still be needed for audits or claims filed under the old rules).

---

## Final recommended configuration

**Markdown-aware chunking, size 400, exact dense retrieval, no reranking, with `status` metadata filtering applied at query time** (archived docs excluded by default).

| Metric | Value | Target | Met? |
|---|---|---|---|
| ndcg@10 | 0.8527 | ≥ 0.80 | ✅ |
| recall@5 | 0.9028 | ≥ 0.85 | ✅ |
| hit_rate@1 | 0.7857 | ≥ 0.65 | ✅ |
| p95 latency | 0.51 ms | ≤ 400 ms | ✅ |

For an offline/batch use case where latency is irrelevant, swap in the LLM reranker (ndcg@10 0.8649, hit_rate@1 0.8810) for the extra quality.

**Limitations of the procedure:** the sweep was greedy — chunking was fixed before retrieval was chosen, which was fixed before reranking was tested — and axes are not guaranteed to be separable. We have direct evidence of exactly this in Part C: reranking helped the LLM-based pipeline's ranking but hurt the cross-encoder's, so "does reranking help" is itself a function of which retriever and chunking preceded it, not a fixed property of the reranker. A pair of axes worth checking for interaction in a follow-up: **chunk size × reranking** — a reranker given more candidates from larger chunks might behave differently than one given the same k from smaller, more granular chunks.

## One thing that surprised us

The cross-encoder reranker didn't just fail to help — it actively *lowered* every quality metric we tracked while adding over 100ms of latency, on a technique the literature treats as close to a free upgrade over plain retrieval. "Retrieve wide, rerank narrow" (T4 §4.4) assumes the reranker's training distribution matches the deployment domain; ours didn't, and the gap showed up as a straightforward regression rather than a smaller-than-expected gain.
