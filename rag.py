#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from aip.chunking import Chunk, markdown_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, enforce_citations  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

ANSWER_SYSTEM = f"""\
You are Aurora's policy-answering assistant. Answer ONLY using the numbered
sources provided below the question. Do not use outside or general
knowledge, even if you are confident it is correct — if it is not in the
sources, it does not go in the answer.

Cite every factual claim by the number(s) of the source(s) that support it,
in square brackets immediately after the claim, e.g. [1] or [2][5]. Never
cite a source number that was not supplied to you.

If the sources do not contain enough information to answer the question,
respond with exactly this sentence and nothing else:
"{REFUSAL}"

If some part of the question is supported and some is not, answer the
supported part with its citation and then state plainly, using language
close to the refusal sentence above, that the rest is not covered by the
sources — do not refuse the whole answer, and do not guess the missing part.

If two sources disagree, do not silently pick one. State that the sources
conflict, cite both, and summarise what each one says.

Keep the answer to two or three sentences unless the question genuinely
requires more detail.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def _unpack_chat(resp: dict) -> tuple[str, str | None]:
    """chat(..., return_full=True) returns a dict; the key names aren't
    documented in help(), so fall back across the likely spellings."""
    text = resp.get("text") or resp.get("content") or resp.get("message") or ""
    finish_reason = resp.get("finish_reason") or resp.get("stop_reason")
    return text, finish_reason




CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    truncated = finish_reason == "length"
    refused = text.strip() == REFUSAL
    _, invalid_idx = enforce_citations(text, n_sources)
    n_citations = len(CITATION_RE.findall(text))

    reasons = []
    if truncated:
        reasons.append("truncated (finish_reason == 'length')")
    if not text.strip():
        reasons.append("empty answer")
    if invalid_idx:
        reasons.append(f"invalid citation indices: {invalid_idx}")
    if not refused and n_citations == 0:
        reasons.append("non-refusal answer has no citations")

    return {
        "valid": not reasons,
        "refused": refused,
        "invalid_citations": invalid_idx,
        "n_citations": n_citations,
        "truncated": truncated,
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    hits = retriever.search(question, k=k)
    hits = reranker.rerank(question, hits, k=final_k) if reranker is not None else hits[:final_k]

    context = format_context(hits)
    user_msg = delimit_untrusted(context) + f"\n\nQuestion: {question}"

    resp = chat(user_msg, system=ANSWER_SYSTEM, tier=tier, max_tokens=512, return_full=True)
    text, finish_reason = _unpack_chat(resp)
    check = validate_answer(text, len(hits), finish_reason)

    # B3: on a bad (non-refusal) answer, retry once with a corrective
    # message naming exactly what failed. If the retry ALSO fails
    # validation, fall back to the exact refusal string -- we never return
    # citations_valid=False, refused=False from this function.
    if not check["valid"] and not check["refused"]:
        corrective = (
            user_msg + f"\n\nYour previous answer was rejected: {check['reason']}. "
            f"Rewrite it, citing only indices 1 to {len(hits)}, or respond with "
            f"the exact refusal sentence if the sources do not support an answer."
        )
        resp = chat(corrective, system=ANSWER_SYSTEM, tier=tier, max_tokens=512, return_full=True)
        text, finish_reason = _unpack_chat(resp)
        check = validate_answer(text, len(hits), finish_reason)

    if not check["valid"] and not check["refused"]:
        text = REFUSAL
        check = validate_answer(text, len(hits), None)

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=check["refused"],
        citations_valid=not check["invalid_citations"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    # E2: SAME generator, different context. Chunk the gold docs with the same
    # markdown-400 strategy the retriever uses, so the comparison measures
    # retrieval quality and not chunk-size artefacts. Repair/fallback path is
    # identical to answer_question for the same reason.
    hits = [
        Hit(chunk=c, score=1.0)
        for doc in gold_docs
        for c in markdown_chunks(doc, doc_id="gold", size=400)
    ]
    # Keep doc identity for citations without changing chunking behaviour.
    for i, h in enumerate(hits):
        h.chunk.doc_id = f"gold-{i}"
        h.chunk.chunk_id = f"gold-{i}"
    context = format_context(hits)
    user_msg = delimit_untrusted(context) + f"\n\nQuestion: {question}"

    resp = chat(user_msg, system=ANSWER_SYSTEM, tier=tier, max_tokens=512, return_full=True)
    text, finish_reason = _unpack_chat(resp)
    check = validate_answer(text, len(hits), finish_reason)

    if not check["valid"] and not check["refused"]:
        corrective = (
            user_msg + f"\n\nYour previous answer was rejected: {check['reason']}. "
            f"Rewrite it, citing only indices 1 to {len(hits)}, or respond with "
            f"the exact refusal sentence if the sources do not support an answer."
        )
        resp = chat(corrective, system=ANSWER_SYSTEM, tier=tier, max_tokens=512, return_full=True)
        text, finish_reason = _unpack_chat(resp)
        check = validate_answer(text, len(hits), finish_reason)

    if not check["valid"] and not check["refused"]:
        text = REFUSAL
        check = validate_answer(text, len(hits), None)

    return Answer(
        question=question, text=text, hits=hits, refused=check["refused"],
        citations_valid=not check["invalid_citations"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"], truncated=check["truncated"],
    )
