"""Build the exact JSON shapes the v3 design consumes from a live Context.

These run on the pipeline worker thread (inside the event sink), where
``ctx.collection`` is stable, so there is no cross-thread dict iteration.
Shapes mirror ``web/live/static/jsx/data.jsx``:

  * accepted paper  -> {id, title, authors, year, venue, score, depth, cites, addedAt, source}
  * network         -> {nodes:[{id, paperId, seed, year, r, cites}], edges:[{a,b}]}  (a,b are indices)
  * metrics         -> dashboard metric strip + rejection/cost bar-lists
"""

from __future__ import annotations

import math
import time
from typing import Any

_MAX_GRAPH_NODES = 700  # keep the canvas responsive


def _authors_str(authors: list[dict] | None) -> str:
    if not authors:
        return ""
    names = [a.get("name", "") for a in authors if a.get("name")]
    if not names:
        return ""
    if len(names) <= 2:
        return " & ".join(names)
    return f"{names[0]} et al."


def _score(cites: int | None) -> float:
    c = max(0, int(cites or 0))
    return round(min(0.99, 0.45 + math.log10(1 + c) * 0.11), 2)


def _radius(cites: int | None, seed: bool) -> float:
    if seed:
        return 8.0
    c = max(0, int(cites or 0))
    return round(3.5 + min(4.5, math.log10(1 + c) * 1.2), 2)


def paper_dict(p, *, seed_ids: set[str]) -> dict[str, Any]:
    """One accepted-paper row for the stream."""
    cites = p.citation_count or 0
    return {
        "id": p.paper_id,
        "title": p.title or p.paper_id,
        "authors": _authors_str(p.authors),
        "year": p.year or 0,
        "venue": p.venue or "",
        "score": _score(cites),
        "depth": int(p.depth or 0),
        "cites": int(cites),
        "source": p.source or "",
        "addedAt": int(time.time() * 1000),
    }


def build_graph(ctx) -> dict[str, Any]:
    """Real citation graph from ctx.collection references ∩ collection.

    Nodes are capped at the most-cited ``_MAX_GRAPH_NODES`` so a big run
    stays smooth; edges use array indices (the canvas contract).
    """
    seed_ids = ctx.seed_ids
    papers = list(ctx.collection.values())
    # prefer seeds + most-cited when capping
    papers.sort(key=lambda p: (p.paper_id in seed_ids, p.citation_count or 0), reverse=True)
    papers = papers[:_MAX_GRAPH_NODES]

    idx = {p.paper_id: i for i, p in enumerate(papers)}
    nodes = []
    for p in papers:
        is_seed = p.paper_id in seed_ids or p.source == "seed"
        nodes.append({
            "id": p.paper_id,
            "paperId": p.paper_id,
            "seed": bool(is_seed),
            "year": p.year or 2022,
            "r": _radius(p.citation_count, is_seed),
            "cites": int(p.citation_count or 0),
        })

    seen: set[tuple[int, int]] = set()
    edges = []
    for p in papers:
        a = idx[p.paper_id]
        neighbors = list(p.references or []) + list(p.supporting_papers or [])
        for ref in neighbors:
            b = idx.get(ref)
            if b is None or b == a:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": a, "b": b})
    return {"nodes": nodes, "edges": edges}


def build_metrics(ctx) -> dict[str, Any]:
    """Dashboard metrics + rejection/cost bar-lists from ctx + budget."""
    budget = ctx.budget
    accepted = len(ctx.collection)
    rej_counts = ctx.rejection_counts
    rejected = sum(rej_counts.values())
    rej_reasons = [
        {"reason": r, "count": int(c)}
        for r, c in rej_counts.most_common(6)
    ]

    s2_req = budget.s2_requests
    s2_hits = budget.s2_cache_hits
    s2_total = s2_req + s2_hits
    s2_cache_pct = round(100 * s2_hits / s2_total) if s2_total else 0

    cost_by_model = budget.cost_by_model() or {}
    cost_by_source = [
        {"source": name, "cost": round(float(info.get("cost", 0.0)), 4)}
        for name, info in cost_by_model.items()
    ]
    s2_cost = round(0.0, 4)  # S2 is free; kept for parity with the design bar-list
    cost_by_source.append({"source": "Semantic Scholar API", "cost": s2_cost})

    elapsed = 0.0
    if ctx.pipeline_started_at is not None:
        elapsed = max(0.0, time.monotonic() - ctx.pipeline_started_at)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "rejectionReasons": rej_reasons,
        "llmTokensIn": budget.llm_input_tokens,
        "llmTokensOut": budget.llm_output_tokens,
        "llmReasoningTokens": budget.llm_reasoning_tokens,
        "llmCalls": budget.llm_calls,
        "llmCacheHits": budget.llm_cache_hits,
        "cost": round(budget.total_cost_usd(), 4),
        "costBySource": cost_by_source,
        "s2Requests": s2_req,
        "s2CacheHits": s2_hits,
        "s2CacheHitPct": s2_cache_pct,
        "elapsedSec": round(elapsed, 1),
    }
