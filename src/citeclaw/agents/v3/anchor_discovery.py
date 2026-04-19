"""Pre-worker anchor-paper discovery for V3.

Runs between ``set_strategy`` and worker dispatch for each sub-topic.
An agent writes a precise S2 query using only terms it's 100%
confident about — no speculative synonyms — then inspects the top
~15 papers by citation count and marks each on-topic / off-topic /
uncertain. Confirmed on-topic papers become the sub-topic's
:class:`~citeclaw.agents.v3.state.AnchorPaper` set, seeding the
worker's ``propose_first`` with real domain vocabulary and giving
``check_anchor_coverage`` something concrete to grade the query on.

The bias risk is bounded: anchors can only be as biased as the
precise query that found them, which is in turn bounded by the
sub-topic description the supervisor wrote. Worker-supplied anchors
are a different story and not part of this module.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from citeclaw.agents.v3.state import AnchorPaper
from citeclaw.prompts.search_agent_v3 import (
    ANCHOR_DISCOVERY_CONFIRM,
    ANCHOR_DISCOVERY_QUERY,
    ANCHOR_DISCOVERY_SYSTEM,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.clients.s2.api import SemanticScholarClient


log = logging.getLogger("citeclaw.agents.v3.anchor_discovery")

_TOP_N_CANDIDATES = 15
_TARGET_ANCHORS_MIN = 3
_TARGET_ANCHORS_MAX = 15


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}
    try:
        return json.loads(s[start:end])
    except json.JSONDecodeError:
        return {}


def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 3].rstrip() + "..."


def _format_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "  (no candidates)"
    lines: list[str] = []
    for i, p in enumerate(candidates, start=1):
        cc = p.get("citationCount") or 0
        title = (p.get("title") or "").strip() or "(no title)"
        abstract = _truncate(p.get("abstract") or "(no abstract)", 400)
        lines.append(f"  [{i}] {cc}c  {title}")
        lines.append(f"       {abstract}")
    return "\n".join(lines)


def _fetch_top_by_citation(
    s2_client: "SemanticScholarClient",
    query_lucene: str,
    limit: int,
) -> list[dict]:
    """Pull a handful of candidates, enrich them, rank by citation.

    We intentionally fetch a few more than ``limit`` so citation-sorting
    has signal even if the top results by S2 relevance are rank-inflated
    reviews / methods papers."""
    try:
        resp = s2_client.search_bulk(query=query_lucene, limit=max(limit * 3, 50))
    except Exception as exc:  # noqa: BLE001
        log.warning("anchor_discovery: search_bulk failed: %s", exc)
        return []
    rows = resp.get("data") or []
    paper_ids = [r.get("paperId") for r in rows if r.get("paperId")]
    if not paper_ids:
        return []
    try:
        records = s2_client.enrich_batch(
            [{"paper_id": pid} for pid in paper_ids[: max(limit * 3, 50)]]
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("anchor_discovery: enrich_batch failed: %s", exc)
        return []
    enriched: list[dict] = []
    for r in records:
        enriched.append({
            "paperId": r.paper_id,
            "title": r.title or "",
            "abstract": r.abstract or "",
            "citationCount": r.citation_count or 0,
        })
    enriched.sort(key=lambda p: -(p.get("citationCount") or 0))
    return enriched[:limit]


def discover_anchors(
    *,
    description: str,
    s2_client: "SemanticScholarClient",
    llm_client: "LLMClient",
    logger: "SearchLogger",
    spec_id: str,
) -> list[AnchorPaper]:
    """Run the 2-step anchor-discovery loop and return confirmed anchors.

    Returns an empty list if the LLM fails, the query returns nothing,
    or none of the candidates are confirmed — the worker still
    proceeds without anchors in that case.
    """
    system = ANCHOR_DISCOVERY_SYSTEM
    user_query = ANCHOR_DISCOVERY_QUERY.format(description=description)
    try:
        resp = llm_client.call(system, user_query, category="v3_anchor_query")
        text = (resp.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("anchor_discovery: query LLM failed: %s", exc)
        return []
    parsed = _extract_json(text)
    query_nl = str(parsed.get("query") or "").strip()
    if not query_nl:
        log.info("anchor_discovery: LLM returned no query for %s", spec_id)
        return []

    from citeclaw.agents.v3.query_translate import to_lucene

    query_lucene = to_lucene(query_nl)
    logger.log_tool_call(
        scope=f"v3_anchor::{spec_id}",
        turn=0,
        tool_name="precise_query",
        args={"query_nl": query_nl},
        result={"query_lucene": query_lucene},
    )

    candidates = _fetch_top_by_citation(s2_client, query_lucene, _TOP_N_CANDIDATES)
    if not candidates:
        logger.log_tool_call(
            scope=f"v3_anchor::{spec_id}",
            turn=0,
            tool_name="fetch_candidates",
            args={"query_lucene": query_lucene},
            result={"n": 0},
        )
        return []

    confirm_user = ANCHOR_DISCOVERY_CONFIRM.format(
        description=description,
        candidates=_format_candidates(candidates),
    )
    try:
        resp = llm_client.call(system, confirm_user, category="v3_anchor_confirm")
        text = (resp.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("anchor_discovery: confirm LLM failed: %s", exc)
        return []
    parsed = _extract_json(text)
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    decisions = decisions if isinstance(decisions, list) else []

    # decisions are indexed 1..N to match the [i] label in the prompt.
    keep_idx: set[int] = set()
    for d in decisions:
        if not isinstance(d, dict):
            continue
        verdict = str(d.get("verdict") or "").strip().lower()
        if verdict != "on_topic":
            continue
        try:
            idx = int(d.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(candidates):
            keep_idx.add(idx)

    anchors: list[AnchorPaper] = []
    for i, p in enumerate(candidates, start=1):
        if i not in keep_idx:
            continue
        anchors.append(AnchorPaper(
            paper_id=str(p.get("paperId") or ""),
            title=str(p.get("title") or ""),
            abstract=str(p.get("abstract") or ""),
            citation_count=int(p.get("citationCount") or 0),
        ))

    logger.log_tool_call(
        scope=f"v3_anchor::{spec_id}",
        turn=1,
        tool_name="confirm_candidates",
        args={
            "n_candidates": len(candidates),
            "min": _TARGET_ANCHORS_MIN,
            "max": _TARGET_ANCHORS_MAX,
        },
        result={"n_anchors": len(anchors), "anchor_titles": [a.title for a in anchors]},
    )
    return anchors


def render_anchors(anchors: list[AnchorPaper]) -> str:
    """Worker-facing rendering of the anchor set — full title + abstract
    so the worker can mine real author vocabulary."""
    if not anchors:
        return "  (no anchor papers — worker proposes from the description alone)"
    lines: list[str] = []
    for i, a in enumerate(anchors, start=1):
        abstract = _truncate(a.abstract, 500) or "(no abstract)"
        lines.append(f"  [{i}] [{a.citation_count}c] {a.title}")
        lines.append(f"       {abstract}")
    return "\n".join(lines)
