#!/usr/bin/env python3
"""Lab 2 — the configurations under test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecordC, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples. FILL THESE IN with real dev-set IDs.
# ---------------------------------------------------------------------------
FEW_SHOT_IDS: list[str] = [
    "T0054",   # teaches: complaint = Aurora's conduct, not the claim itself; null policy_number
    "T0183",   # teaches: extract policy_number correctly despite HTML/markup noise
    "T0025",   # teaches: the other side of the complaint/claims boundary (claim-adjacent anger is still complaint if conduct is blamed)
    "T0200",   # teaches: hi-en tagging on a claims ticket with a rupee figure
    "T0080",   # teaches: ignore ticket-header metadata noise; neutral tone can still carry an active request
    "T0095",   # teaches: distinguish a live-message policy number from one in quoted/prior context
]


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str]) -> str:
    """Render examples in the EXACT JSON shape TicketRecordC asks for."""
    if not ids:
        return ""
    examples = load_examples(ids)
    parts = ["Worked examples. Match this exact JSON shape for every field:"]
    for ex in examples:
        gold = {k: v for k, v in (ex.get("expected") or {}).items()
                if k in TicketRecordC.model_fields}
        parts.append(
            f"Ticket:\n{ex['input']}\n\nOutput:\n"
            f"{json.dumps(gold, ensure_ascii=False)}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------
def _fallback(schema) -> dict:
    return schema(
        evidence="", category="information", urgency=1, sentiment="neutral",
        product="unknown", language="en", needs_human_review=True,
        review_reason="fallback",
    ).model_dump()


def _safe_structured(ticket: str, schema, system: str, tier: str,
                     temperature: float | None = None) -> dict:
    try:
        kwargs = {} if temperature is None else {"temperature": temperature}
        rec = structured(ticket, schema=schema, system=system, tier=tier, **kwargs)
        return rec.model_dump()
    except Exception as e:
        d = _fallback(schema)
        d["review_reason"] = f"{type(e).__name__}: {e}"[:200]
        return d


def _finish(fields: dict, ticket: str) -> dict:
    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Lab 1 Part C, no examples. The baseline."""
    fields = _safe_structured(ticket, TicketRecordC, SYSTEM_PROMPT, tier)
    return _finish(fields, ticket)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """zero_shot + the few-shot block."""
    system = SYSTEM_PROMPT + "\n\n" + few_shot_block(FEW_SHOT_IDS)
    fields = _safe_structured(ticket, TicketRecordC, system, tier)
    return _finish(fields, ticket)


class TicketRecordReasoned(TicketRecordC.__base__):
    pass


# Defined fresh (NOT a subclass of TicketRecordC) so `reasoning` is the FIRST
# field in declaration order -- Pydantic preserves declaration order in the
# JSON Schema, and field order influences generation order (T2 §3.3).
# Subclassing TicketRecordC would put inherited fields first and `reasoning`
# last, which conditions nothing -- it would just rationalise an answer
# already committed to. This is a deliberate deviation from the stub, which
# subclassed TicketRecord; doing so cannot satisfy its own "reasoning first"
# requirement, so the schema is rebuilt directly instead.
from typing import Literal  # noqa: E402
from labs.lab1.extract import CATEGORIES  # noqa: E402


class TicketRecordReasoned(TicketRecordC):  # type: ignore[no-redef]
    pass


def _build_reasoned_schema():
    from pydantic import BaseModel

    class _Reasoned(BaseModel):
        reasoning: str = Field(
            description="Think step by step: what does the ticket say, which "
                        "category/urgency rule applies, and why -- BEFORE "
                        "committing to an answer. 2-3 sentences max."
        )
        evidence: str = Field(
            max_length=200,
            description=TicketRecordC.model_fields["evidence"].description,
        )
        category: CATEGORIES = Field(
            description=TicketRecordC.model_fields["category"].description,
        )
        urgency: int = Field(
            ge=1, le=5,
            description=TicketRecordC.model_fields["urgency"].description,
        )
        sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
            description=TicketRecordC.model_fields["sentiment"].description,
        )
        product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
            description=TicketRecordC.model_fields["product"].description,
        )
        language: Literal["en", "hi-en"] = Field(
            description=TicketRecordC.model_fields["language"].description,
        )
        needs_human_review: bool = False
        review_reason: str = ""

    return _Reasoned


TicketRecordReasoned = _build_reasoned_schema()


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """few_shot with TicketRecordReasoned -- reasoning conditions the answer."""
    system = SYSTEM_PROMPT + "\n\n" + few_shot_block(FEW_SHOT_IDS)
    fields = _safe_structured(ticket, TicketRecordReasoned, system, tier)
    fields.pop("reasoning", None)  # not a graded field; drop before scoring
    return _finish(fields, ticket)


# ---------------------------------------------------------------------------
# Part C — the cascade
# ---------------------------------------------------------------------------
def _samples_agree(a: dict, b: dict) -> bool:
    keys = ("category", "urgency", "sentiment")
    return all(a.get(k) == b.get(k) for k in keys)


def cascade(ticket: str) -> dict:
    """SMALL first, twice at T=0.7; escalate to MAIN if they disagree or
    either fails validation / returns empty evidence.

    Two T=0.7 samples (not T=0, per the lab's own warning) so the second
    call is NOT served from cache -- identical T=0 calls are the same cache
    key, disagreement is never observed, and escalation silently reads 0%.
    """
    d1 = _safe_structured(ticket, TicketRecordC, SYSTEM_PROMPT, "SMALL", temperature=0.7)
    d2 = _safe_structured(ticket, TicketRecordC, SYSTEM_PROMPT, "SMALL", temperature=0.7)

    trigger = (
        d1.get("needs_human_review") or d2.get("needs_human_review")
        or not d1.get("evidence") or not d2.get("evidence")
        or not _samples_agree(d1, d2)
    )

    if not trigger:
        fields, path = d1, "small"
    else:
        fields = _safe_structured(ticket, TicketRecordC, SYSTEM_PROMPT, "MAIN")
        path = "large"

    fields = _finish(fields, ticket)
    fields["_path"] = path
    return fields


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
