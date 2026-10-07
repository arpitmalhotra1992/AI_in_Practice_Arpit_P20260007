#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.config import resolve_model  # noqa: E402
from aip.cost import Budget, BudgetExceeded, global_budget  # noqa: E402
from aip.guards import ToolGuard  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")
_STARTED = time.time()

# ---------------------------------------------------------------------------
# Pipeline: Labs 3-5 (v2: archived-filter dense retriever + Lab 4 answering
# with citation validation and the exact-string refusal contract) plus the
# Lab 6 guards that apply to a read-only RAG path: untrusted content is
# delimited (inside labs.lab4.rag), citations are enforced in code with
# retry-then-refuse, arguments are irrelevant here (no tools in rag mode),
# and every request runs under a spend Budget (429 on exhaustion). The
# service exposes no side-effecting tools at all: mode=tools runs the Lab 6
# loop under a read-only allowlist (refund excluded) with never-confirm.
# ---------------------------------------------------------------------------
_PIPE = None


def pipeline():
    """Build once at startup and cache it (never per request)."""
    global _PIPE
    if _PIPE is None:
        from labs.lab5.diagnose import build_v2_retriever
        _PIPE = {"retriever": build_v2_retriever()}
    return _PIPE


# ---------------------------------------------------------------------------
# Caches (B1). Exact: normalised-question hash -- a hit IS the same question,
# so it cannot be wrong. Semantic: question-embedding cosine -- a hit may be
# a DIFFERENT question (see the measured threshold below).
# ---------------------------------------------------------------------------
_EXACT: dict[str, dict] = {}
_SEM_STORE: list[dict] = []  # {vec, question, response}
_SEM_HITS = {"exact": 0, "semantic": 0, "miss": 0}

# Measured in scripts/sweep_semantic_cache.py on the 45 golden questions:
# pairs with cosine >= 0.990 are always interchangeable; the first
# wrong-hit (Gold vs Silver waiting-period -- different answers at 0.986)
# appears at 0.986, so the shipped threshold is 0.990. Below that the cache
# returns confident, well-cited answers to questions nobody asked.
SEMANTIC_THRESHOLD = 0.990


def _normalise(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _exact_key(q: str) -> str:
    return hashlib.sha256(_normalise(q).encode()).hexdigest()


# In-memory request log for /metrics (cumulative for the process lifetime).
_REQUESTS: list[dict] = []


def _answer_rag(question: str, top_k: int, trace_id: str) -> dict:
    """Full RAG answer with per-stage tracing. Returns answer dict."""
    from aip.retrieval import format_context
    from labs.lab4.rag import (ANSWER_SYSTEM, REFUSAL, answer_question,
                               validate_answer)
    from labs.lab4.rag import CITATION_RE

    pipe = pipeline()
    with tracing.trace("svc.retrieve", top_k=top_k) as s:
        hits = pipe["retriever"].search(question, k=12)[:top_k]
        s["n_hits"] = len(hits)
        ctx = format_context(hits)
    with tracing.trace("svc.generate", tier="MAIN", n_sources=len(hits)):
        ans = answer_question(question, pipe["retriever"], k=12,
                              final_k=top_k)
    with tracing.trace("svc.validate") as s:
        s["citations_valid"] = ans.citations_valid
        s["refused"] = ans.refused
    cites = sorted({int(m) for m in CITATION_RE.findall(ans.text)})
    citations = [{"index": i, "doc_id": ans.hits[i - 1].doc_id,
                  "excerpt": ans.hits[i - 1].text[:600]}
                 for i in cites if 1 <= i <= len(ans.hits)]
    _ = (ANSWER_SYSTEM, REFUSAL, validate_answer, ctx)  # contract imports
    return {"answer": ans.text, "refused": ans.refused,
            "citations": citations,
            "sources": [{"index": i + 1, "doc_id": h.doc_id,
                         "excerpt": h.text[:600]} for i, h in enumerate(ans.hits)]}


def _answer_tools(question: str) -> dict:
    """Read-only Lab 6 loop: refund excluded from the allowlist, never confirm."""
    from labs.lab6.agent import run_agent
    guard = ToolGuard(
        max_calls=6,
        allow={"search_policy", "get_policy_details", "compute_premium"},
        requires_confirmation={"issue_refund"},
        confirm_fn=lambda name, a: False)
    with tracing.trace("svc.tools_loop"):
        r = run_agent(question, guard=guard)
    return {"answer": r["answer"], "refused": False, "citations": [],
            "sources": [{"index": i + 1, "doc_id": t["tool"],
                         "excerpt": t["result_preview"][:600]}
                        for i, t in enumerate(r["tool_log"])]}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


def _lookup_caches(question: str):
    key = _exact_key(question)
    if key in _EXACT:
        _SEM_HITS["exact"] += 1
        return _EXACT[key], "exact"
    if _SEM_STORE:
        from aip.embed import cosine, embed
        qv = embed(question, input_type="query")
        import numpy as np
        sims = cosine(qv, np.asarray([e["vec"] for e in _SEM_STORE]))
        best = int(sims.argmax())
        if float(sims[best]) >= SEMANTIC_THRESHOLD:
            _SEM_HITS["semantic"] += 1
            return _SEM_STORE[best]["response"], "semantic"
    _SEM_HITS["miss"] += 1
    return None, "miss"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    trace_id = uuid.uuid4().hex[:12]
    # A3 test hooks (off by default): env var read at request time so tests
    # need no redeploy; the outage path returns 503 + Retry-After, never 500.
    if os.getenv("AIP_SIMULATE_OUTAGE") == "1" or Path("/tmp/lab7_force_outage").exists():
        raise HTTPException(status_code=503, detail="upstream model unavailable",
                            headers={"Retry-After": "30"})
    try:
        with tracing.trace("http.ask", trace_id=trace_id,
                           question=req.question[:120], mode=req.mode) as span:
            hit, kind = _lookup_caches(req.question)
            b0 = global_budget().spent_usd
            if hit is not None:
                body = dict(hit)
                cached = True
                cost = 0.0
            else:
                try:
                    with Budget(limit_usd=float(os.getenv("ASK_BUDGET_USD",
                                                          "0.05")),
                                label=f"ask-{trace_id}"):
                        data = (_answer_rag(req.question, req.top_k, trace_id)
                                if req.mode == "rag"
                                else _answer_tools(req.question))
                except BudgetExceeded as exc:
                    raise HTTPException(status_code=429, detail=str(exc)) from exc
                cost = round(global_budget().spent_usd - b0, 6)
                body = {**data, "cost_usd": cost}
                _EXACT[_exact_key(req.question)] = dict(body)
                if kind == "miss" and not body["refused"]:
                    from aip.embed import embed
                    try:
                        _SEM_STORE.append(
                            {"vec": embed(req.question,
                                          input_type="query").tolist(),
                             "question": req.question, "response": dict(body)})
                    except Exception:  # noqa: BLE001 -- cache must not break asks
                        pass
                cached = False
            ms = (time.perf_counter() - t0) * 1000
            span["cached"] = cached
            span["cost_usd"] = cost
            span["latency_ms"] = round(ms, 1)
            rec = {"trace_id": trace_id, "latency_ms": round(ms, 1),
                   "cost_usd": cost if not cached else 0.0, "cached": cached,
                   "refused": bool(body.get("refused")), "mode": req.mode,
                   "error": None}
            _REQUESTS.append(rec)
            tracing.event("http.answer", trace_id=trace_id, cached=cached,
                          refused=bool(body.get("refused")))
            return AskResponse(answer=body["answer"], refused=bool(body.get("refused")),
                               citations=body.get("citations", []),
                               latency_ms=round(ms, 1), cost_usd=cost,
                               cached=cached, trace_id=trace_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__.lower()
        _REQUESTS.append({"trace_id": trace_id, "latency_ms": -1,
                          "cost_usd": 0.0, "cached": False, "refused": False,
                          "mode": req.mode, "error": name})
        if any(k in name for k in ("apiconnection", "ratelimit", "overloaded",
                                   "timeout", "serviceunavailable",
                                   "internalserver")):
            raise HTTPException(status_code=503,
                                detail="upstream model unavailable",
                                headers={"Retry-After": "30"}) from exc
        raise HTTPException(status_code=500, detail="internal error") from exc


@app.get("/health")
def health() -> dict:
    pipe = pipeline()
    return {"status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
            "index_chunks": len(pipe["retriever"].chunks),
            "model_profile": resolve_model("MAIN"),
            "cache": {**cache.stats(), "exact_responses": len(_EXACT),
                       "semantic_entries": len(_SEM_STORE), **_SEM_HITS}}


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


@app.get("/metrics")
def metrics() -> dict:
    lat = [r["latency_ms"] for r in _REQUESTS if r["latency_ms"] >= 0]
    ok = [r for r in _REQUESTS if not r["error"]]
    b = global_budget()
    by_type: dict[str, int] = {}
    for r in _REQUESTS:
        if r["error"]:
            by_type[r["error"]] = by_type.get(r["error"], 0) + 1
    tools = sum(1 for r in _REQUESTS if r["mode"] == "tools")
    refused = sum(1 for r in ok if r["refused"])
    tot = b.as_dict()
    tot.update({
        "requests": len(_REQUESTS),
        "cost_per_query_usd": round(b.spent_usd / len(_REQUESTS), 6) if _REQUESTS else 0.0,
        "cache_hit_rate": ((  _SEM_HITS["exact"] + _SEM_HITS["semantic"])
                           / max(1, sum(_SEM_HITS.values()))),
        "cache_breakdown": dict(_SEM_HITS),
        "latency_p50_ms": round(_pct(lat, 50), 1),
        "latency_p95_ms": round(_pct(lat, 95), 1),
        "latency_p99_ms": round(_pct(lat, 99), 1),
        "error_rate": round(1 - len(ok) / len(_REQUESTS), 4) if _REQUESTS else 0.0,
        "errors_by_type": by_type,
        "refusal_rate": round(refused / len(ok), 4) if ok else 0.0,
        "tool_requests": tools,
    })
    return tot


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """B2/B3: true token streaming, then a terminal validation event.

    B3 choice: stream prose live, hold the grounding verdict to a final
    `validation` event the UI acts on (marks citations verified/failed).
    Rationale: citation validity needs the complete answer, so it cannot lead;
    but making validation a machine-readable event (not prose) keeps the TTFT
    win while refusing to present unchecked citations as checked.
    """
    from labs.lab4.rag import ANSWER_SYSTEM, CITATION_RE, REFUSAL

    def gen():
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex[:12]
        try:
            from aip.retrieval import format_context
            from aip.guards import delimit_untrusted
            pipe = pipeline()
            hits = pipe["retriever"].search(req.question, k=12)[:req.top_k]
            context = delimit_untrusted(format_context(hits))
            user_msg = context + f"\n\nQuestion: {req.question}"
            model = resolve_model("MAIN")
            from litellm import completion
            first = True
            chunks: list[str] = []
            stream = completion(model=model,
                                messages=[{"role": "system", "content": ANSWER_SYSTEM},
                                          {"role": "user", "content": user_msg}],
                                temperature=0.0, max_tokens=512, stream=True)
            for part in stream:
                tok = (part.choices[0].delta.content or "")
                if not tok:
                    continue
                chunks.append(tok)
                if first:
                    yield {"event": "ttft",
                           "data": f"{(time.perf_counter()-t0)*1000:.0f}"}
                    first = False
                yield {"event": "token", "data": tok}
            full = "".join(chunks)
            if full.strip() == REFUSAL:
                valid, cites = True, []
            else:
                from aip.guards import enforce_citations
                valid, _ = True, []
                ok, invalid = enforce_citations(full, len(hits))
                valid = ok
                cites = sorted({int(m) for m in CITATION_RE.findall(full)
                                if 1 <= int(m) <= len(hits)})
            yield {"event": "validation",
                   "data": json_dumps({"citations_valid": valid,
                                       "citations": cites, "trace_id": trace_id,
                                       "total_ms": round((time.perf_counter()-t0)*1000)})}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": f"{type(exc).__name__}"}

    return EventSourceResponse(gen())


def json_dumps(o: dict) -> str:
    import json as _json
    return _json.dumps(o)
