#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else."""

    # B1a: evidence goes FIRST. Forces the model to quote the deciding span
    # before committing to category/urgency, so the label is grounded in
    # text it already produced rather than justified after the fact.
    evidence: str = Field(
        max_length=200,
        description="The exact span of the ticket text that determined the "
                    "category. Quote it verbatim, at most one sentence."
    )

    category: CATEGORIES = Field(
        description="billing = money in: premium, debits, refunds, invoices, "
                    "tax certificate, instalments. claims = an actual or "
                    "intended claim: cashless, reimbursement, settlement, "
                    "rejection. policy_change = altering the contract: add/"
                    "remove a member, upgrade, port, change contact details. "
                    "technical = the app, portal, OTP, locator, or upload is "
                    "broken. complaint = Aurora's CONDUCT is the subject "
                    "(mis-selling, being kept on hold, an ignored grievance) "
                    "-- not just an angry tone. information = a question with "
                    "no pending transaction."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="1 = answerable from general knowledge, no lookup needed. "
                    "2 = requires looking up or acting on this customer's "
                    "account, or a transaction is in flight. 3 = something has "
                    "already gone wrong or is stuck and the customer is "
                    "waiting. 4 = repeated failure to resolve, money/access at "
                    "risk now, or an explicit escalation threat. 5 = an "
                    "emergency in progress, a formal denial, or the customer "
                    "states they ARE escalating to the Ombudsman (not merely "
                    "threatening to). Add 1 (capped at 5) if a same-day or "
                    "next-morning deadline is stated."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Tone only, independent of urgency. angry = hostile, "
                    "shouting, threatening. frustrated = unhappy, civil, and "
                    "references a PRIOR failure (repeat attempt, delay, no "
                    "response). neutral = matter-of-fact, including a terse "
                    "first-time request. satisfied = thanks or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The plan name only if explicitly NAMED in the message. "
                    "Never infer it from sum insured or context. Use "
                    "'unknown' if no plan name appears."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words are mixed into the English, "
                    "including transliterated Hindi in Latin script (kripya, "
                    "jaldi, bahut, turant). 'en' otherwise."
    )

    policy_number: str | None = Field(
        default=None,
        description="Format AUR- followed by exactly 7 digits, copied "
                    "character for character from the LIVE message only "
                    "(not a quoted reply or signature block). Return null if "
                    "no such string appears. Never invent or reformat one."
    )
    contains_pii: bool = Field(
        default=False,
        description="True if the text contains a phone number, or an email "
                    "address that is not support@aurorahealth.example or "
                    "grievance@aurorahealth.example. A name alone does not "
                    "count."
    )

    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if v == "" or v.lower() in {"null", "none", "n/a"}:
            return None
        if not re.fullmatch(r"AUR-\d{7}", v):
            raise ValueError(f"policy_number {v!r} is not AUR-<7 digits>")
        return v


SYSTEM_PROMPT = """\
You extract structured data from customer support tickets for Aurora Health \
Insurance.

Read the ticket and populate every field of the schema. Base every field only \
on what the ticket actually says -- never infer, assume, or invent a value \
that is not supported by the text.

Quote the deciding evidence verbatim before assigning category or urgency.

If the ticket is a forwarded thread, lines starting with '>' are quoted \
history, not the live message -- do not extract a policy number from them.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL")
    except Exception as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            needs_human_review=True,
            review_reason=f"{type(e).__name__}: {e}"[:200],
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """Return {'policy_number', 'contains_pii'} without a model call.

    policy_number: only searched in the LIVE portion of the message (before
    the first quoted '>' line), per data/README.md -- a quoted reply or
    signature block can carry a stale/different number.

    contains_pii: per data/README.md's narrow definition for this dataset --
    True only for a phone number, or an email address that is NOT one of
    Aurora's own published addresses. Other PII categories in _PII_PATTERNS
    (Aadhaar, PAN, card, IP) are out of scope for this label.
    """
    match = QUOTE_MARKER.search(ticket)
    live_text = ticket[: match.start()] if match else ticket

    m = POLICY_RE.search(live_text)
    policy_number = m.group(0) if m else None

    aurora_addresses = {"support@aurorahealth.example", "grievance@aurorahealth.example"}

    has_phone = bool(_PII_PATTERNS["PHONE_IN"].search(ticket))
    emails = _PII_PATTERNS["EMAIL"].findall(ticket)
    has_non_aurora_email = any(e.lower() not in aurora_addresses for e in emails)

    contains_pii = has_phone or has_non_aurora_email

    return {"policy_number": policy_number, "contains_pii": contains_pii}


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """escalate = urgency >= 4 or 'ombudsman' appears in the ticket.

    A business rule, computed in code -- readable by a compliance officer,
    changeable without touching a prompt, and unit-testable.
    """
    rec_fields["escalate"] = (
        rec_fields.get("urgency", 0) >= 4 or "ombudsman" in ticket.lower()
    )
    return rec_fields


class TicketRecordC(BaseModel):
    """The reduced schema the model sees in Part C.

    Same as TicketRecord, minus policy_number and contains_pii -- those are
    now computed deterministically in extract_deterministic().
    """

    evidence: str = Field(
        max_length=200,
        description="The exact span of the ticket text that determined the "
                    "category. Quote it verbatim, at most one sentence."
    )
    category: CATEGORIES = Field(
        description="billing = money in: premium, debits, refunds, invoices, "
                    "tax certificate, instalments. claims = an actual or "
                    "intended claim: cashless, reimbursement, settlement, "
                    "rejection. policy_change = altering the contract: add/"
                    "remove a member, upgrade, port, change contact details. "
                    "technical = the app, portal, OTP, locator, or upload is "
                    "broken. complaint = Aurora's CONDUCT is the subject "
                    "(mis-selling, being kept on hold, an ignored grievance) "
                    "-- not just an angry tone. information = a question with "
                    "no pending transaction."
    )
    urgency: int = Field(
        ge=1, le=5,
        description="1 = answerable from general knowledge, no lookup needed. "
                    "2 = requires looking up or acting on this customer's "
                    "account, or a transaction is in flight. 3 = something has "
                    "already gone wrong or is stuck and the customer is "
                    "waiting. 4 = repeated failure to resolve, money/access at "
                    "risk now, or an explicit escalation threat. 5 = an "
                    "emergency in progress, a formal denial, or the customer "
                    "states they ARE escalating to the Ombudsman (not merely "
                    "threatening to). Add 1 (capped at 5) if a same-day or "
                    "next-morning deadline is stated."
    )
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Tone only, independent of urgency. angry = hostile, "
                    "shouting, threatening. frustrated = unhappy, civil, and "
                    "references a PRIOR failure (repeat attempt, delay, no "
                    "response). neutral = matter-of-fact, including a terse "
                    "first-time request. satisfied = thanks or praise."
    )
    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The plan name only if explicitly NAMED in the message. "
                    "Never infer it from sum insured or context. Use "
                    "'unknown' if no plan name appears."
    )
    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words are mixed into the English, "
                    "including transliterated Hindi in Latin script (kripya, "
                    "jaldi, bahut, turant). 'en' otherwise."
    )

    needs_human_review: bool = False
    review_reason: str = ""


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier="SMALL")
        fields = rec.model_dump()
    except Exception as e:
        fields = TicketRecordC(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            needs_human_review=True,
            review_reason=f"{type(e).__name__}: {e}"[:200],
        ).model_dump()

    fields.update(extract_deterministic(ticket))
    fields = apply_business_rules(fields, ticket)
    return fields