"""V3 worker — tutorial-style 6-phase loop.

One long conversation per worker. Each phase emits a single user
message asking one focused question; the LLM replies with JSON. The
phases are:

  propose  →  check_total  →  check_clusters  →  check_top100  →
  diagnose_plan (tool-loop) →  write_next  →  (back to fetch)

Runs for ``max_iter`` iterations regardless of observed saturation —
no self-stop tool.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from citeclaw.agents.v3.analysis import (
    build_query_tree,
    diff_vs_prev,
    render_clusters,
    render_query_tree,
    topic_model,
)
from citeclaw.agents.v3.state import (
    AgentConfigV3,
    QueryIterationV3,
    SubTopicResultV3,
    SubTopicSpecV3,
    TopicCluster,
    WorkerStateV3,
)
from citeclaw.prompts.search_agent_v3 import (
    WORKER_CHECK_CLUSTERS,
    WORKER_CHECK_TOP100,
    WORKER_CHECK_TOTAL,
    WORKER_DIAGNOSE_PLAN,
    WORKER_FINAL_SUMMARY,
    WORKER_PROPOSE_FIRST,
    WORKER_SYSTEM,
    WORKER_WRITE_NEXT,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.clients.s2.api import SemanticScholarClient

log = logging.getLogger("citeclaw.agents.v3.worker")


_ENRICH_CAP = 2000  # max papers to hydrate per iteration for topic model + top-cited

_MAX_DIAGNOSE_TOOL_CALLS = 3


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response. Tolerant to
    code fences and stray prose."""
    if not text:
        return {}
    # Strip code fences
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Find first { ... } balanced block
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
    blob = s[start:end]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


def _call_llm(
    *,
    llm_client: "LLMClient",
    system: str,
    messages: list[dict[str, str]],
    user_msg: str,
    category: str,
) -> tuple[str, dict]:
    """Append user_msg, call LLM, append assistant reply, return (text, parsed)."""
    messages.append({"role": "user", "content": user_msg})
    # Single-turn call shape: build a composite user message from all
    # prior messages excluding system. Keeps it simple — conversations
    # stay a single thread.
    joined = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content']}" for m in messages
    )
    try:
        resp = llm_client.call(
            system,
            joined,
            category=category,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("V3 worker LLM call failed (%s): %s", category, exc)
        messages.append({"role": "assistant", "content": "{}"})
        return "", {}
    text = (resp.text or "").strip()
    messages.append({"role": "assistant", "content": text})
    parsed = _extract_json(text)
    return text, parsed


# ---------------------------------------------------------------------------
# S2 fetch + analyse one query
# ---------------------------------------------------------------------------


def _fetch_and_analyse(
    *,
    query: str,
    iter_idx: int,
    s2_client: "SemanticScholarClient",
    prior_iters: list[QueryIterationV3],
    config: AgentConfigV3,
) -> QueryIterationV3:
    paper_ids: list[str] = []
    total = 0
    token: str | None = None
    per_call_limit = 500
    while len(paper_ids) < config.max_papers_per_query:
        try:
            resp = s2_client.search_bulk(query=query, limit=per_call_limit, token=token)
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 search_bulk failed: %s", exc)
            break
        data = resp.get("data") or []
        if iter_idx == 0 and len(paper_ids) == 0:
            total = int(resp.get("total") or 0)
        elif token is None:
            total = int(resp.get("total") or 0)
        for row in data:
            pid = row.get("paperId")
            if pid:
                paper_ids.append(pid)
        token = resp.get("token")
        if not token or not data:
            break
    paper_ids = paper_ids[: config.max_papers_per_query]

    # Enrich the first slice so we can compute top-cited + topic model
    enrich_ids = paper_ids[:_ENRICH_CAP]
    enriched: list[dict] = []
    if enrich_ids:
        try:
            records = s2_client.enrich_batch([{"paper_id": pid} for pid in enrich_ids])
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 enrich_batch failed: %s", exc)
            records = []
        by_id = {r.paper_id: r for r in records}
        for pid in enrich_ids:
            r = by_id.get(pid)
            if r is None:
                continue
            enriched.append({
                "paperId": pid,
                "title": r.title or "",
                "abstract": r.abstract or "",
                "citationCount": r.citation_count or 0,
            })

    # Top-100 by citation count
    enriched_sorted = sorted(enriched, key=lambda p: -(p.get("citationCount") or 0))
    top_titles = [p["title"] for p in enriched_sorted[:100] if p.get("title")]

    # Topic clusters on the enriched subset (faster + more meaningful
    # than on 10K shallow records).
    clusters = topic_model(enriched, k=config.topic_k)

    # Query tree via count-only sub-queries
    query_tree = build_query_tree(query, full_total=total, s2_client=s2_client)

    # Diff vs prior iters
    prior_ids_set: set[str] = set()
    for it in prior_iters:
        prior_ids_set.update(it.paper_ids)
    diff_new, diff_seen = diff_vs_prev(paper_ids, prior_ids_set)

    return QueryIterationV3(
        iter_idx=iter_idx,
        query=query,
        total_count=total,
        fetched_count=len(paper_ids),
        paper_ids=paper_ids,
        query_tree=query_tree,
        clusters=clusters,
        top_titles_100=top_titles,
        diff_new=diff_new,
        diff_seen=diff_seen,
    )


# ---------------------------------------------------------------------------
# Tools during diagnose_plan phase
# ---------------------------------------------------------------------------


def _inspect_topic(
    iteration: QueryIterationV3,
    cluster_id: int,
    *,
    enriched_by_id: dict[str, dict],
) -> str:
    cluster = next((c for c in iteration.clusters if c.cluster_id == cluster_id), None)
    if cluster is None:
        return f"(no cluster {cluster_id}; valid ids: {[c.cluster_id for c in iteration.clusters]})"
    # Representative titles are already stored; fetch abstracts for them
    lines = [f"Cluster {cluster_id} — {cluster.count} papers. Keywords: {', '.join(cluster.keywords[:8])}"]
    for title in cluster.representative_titles[:5]:
        # Find the paper by title in enriched pool
        match = next(
            (p for p in enriched_by_id.values() if (p.get("title") or "").strip() == title.strip()),
            None,
        )
        if match:
            abs_text = (match.get("abstract") or "").strip()
            abs_short = abs_text if len(abs_text) <= 400 else abs_text[:400] + "..."
            lines.append(f"  · {title}")
            lines.append(f"    {abs_short}")
        else:
            lines.append(f"  · {title}")
    return "\n".join(lines)


def _inspect_paper(
    title: str,
    *,
    s2_client: "SemanticScholarClient",
    iteration: QueryIterationV3,
) -> str:
    try:
        matched = s2_client.search_match(title)
    except Exception as exc:  # noqa: BLE001
        return f"(search_match failed: {exc})"
    if not matched:
        return f"(S2 found no paper matching title: {title!r})"
    matched_id = matched.get("paperId", "")
    matched_title = matched.get("title", "")
    abstract = (matched.get("abstract") or "").strip()
    abs_short = abstract if len(abstract) <= 400 else abstract[:400] + "..."
    in_results = "YES" if matched_id in set(iteration.paper_ids) else "NO"
    return (
        f"Matched paper: {matched_title}\n"
        f"Paper ID: {matched_id}\n"
        f"In current iter's fetched set: {in_results}\n"
        f"Abstract: {abs_short or '(no abstract)'}"
    )


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------


def run_v3_worker(
    *,
    spec: SubTopicSpecV3,
    config: AgentConfigV3,
    s2_client: "SemanticScholarClient",
    llm_client: "LLMClient",
    logger: "SearchLogger",
) -> SubTopicResultV3:
    system_prompt = WORKER_SYSTEM.format(description=spec.description)
    state = WorkerStateV3(sub_topic_id=spec.id, description=spec.description)
    enriched_pool: dict[str, dict] = {}

    last_diagnosis = ""
    last_intended_change = ""

    for iter_idx in range(config.max_iter):
        # Phase 1: PROPOSE
        if iter_idx == 0:
            user_msg = WORKER_PROPOSE_FIRST.format(description=spec.description)
        else:
            user_msg = WORKER_WRITE_NEXT.format(
                diagnosis=last_diagnosis or "(none recorded)",
                intended_change=last_intended_change or "(none recorded)",
            )
        _, parsed = _call_llm(
            llm_client=llm_client,
            system=system_prompt,
            messages=state.messages,
            user_msg=user_msg,
            category="v3_worker_propose",
        )
        query = (parsed.get("query") or "").strip()
        if not query:
            log.warning("V3 worker %s: iter %d propose returned no query", spec.id, iter_idx)
            break

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="propose_query",
            args={"iter": iter_idx},
            result={"query": query},
        )

        # Phase 2: FETCH + ANALYSE (system-side, no LLM)
        iteration = _fetch_and_analyse(
            query=query,
            iter_idx=iter_idx,
            s2_client=s2_client,
            prior_iters=state.iterations,
            config=config,
        )
        state.iterations.append(iteration)
        # Add iter's enriched records to pool for inspect_topic
        for pid in iteration.paper_ids[:_ENRICH_CAP]:
            # We already populated enriched list in _fetch_and_analyse but
            # didn't return it. Enrich on demand here is wasteful — just
            # re-hydrate from cache (cheap).
            if pid in enriched_pool:
                continue
            try:
                recs = s2_client.enrich_batch([{"paper_id": pid}])
            except Exception:  # noqa: BLE001
                continue
            for r in recs:
                enriched_pool[r.paper_id] = {
                    "paperId": r.paper_id,
                    "title": r.title or "",
                    "abstract": r.abstract or "",
                    "citationCount": r.citation_count or 0,
                }

        # Phase 3: CHECK_TOTAL
        _call_llm(
            llm_client=llm_client,
            system=system_prompt,
            messages=state.messages,
            user_msg=WORKER_CHECK_TOTAL.format(
                query=query,
                total=iteration.total_count,
                diff_new=iteration.diff_new,
                diff_seen=iteration.diff_seen,
                query_tree=render_query_tree(iteration.query_tree),
            ),
            category="v3_worker_check_total",
        )

        # Phase 4: CHECK_CLUSTERS
        _call_llm(
            llm_client=llm_client,
            system=system_prompt,
            messages=state.messages,
            user_msg=WORKER_CHECK_CLUSTERS.format(
                query=query,
                description=spec.description,
                clusters=render_clusters(iteration.clusters),
            ),
            category="v3_worker_check_clusters",
        )

        # Phase 5: CHECK_TOP100
        titles_block = "\n".join(
            f"  {i+1}. {t}" for i, t in enumerate(iteration.top_titles_100)
        ) or "  (no papers fetched)"
        _call_llm(
            llm_client=llm_client,
            system=system_prompt,
            messages=state.messages,
            user_msg=WORKER_CHECK_TOP100.format(
                query=query,
                titles=titles_block,
            ),
            category="v3_worker_check_top100",
        )

        # Phase 6: DIAGNOSE + PLAN (with optional tool calls)
        diagnosis = ""
        intended_change = ""
        for tool_round in range(_MAX_DIAGNOSE_TOOL_CALLS + 1):
            _, parsed = _call_llm(
                llm_client=llm_client,
                system=system_prompt,
                messages=state.messages,
                user_msg=WORKER_DIAGNOSE_PLAN,
                category="v3_worker_diagnose_plan",
            )
            tool = (parsed.get("tool") or "").strip()
            if tool == "plan":
                diagnosis = str(parsed.get("diagnosis") or "")
                intended_change = str(parsed.get("intended_change") or "")
                break
            if tool_round >= _MAX_DIAGNOSE_TOOL_CALLS:
                # Force plan on last round
                state.messages.append({
                    "role": "user",
                    "content": (
                        "You've used your tool-call budget for this iteration. "
                        "Commit to a plan now. "
                        'Respond with: {"tool": "plan", "diagnosis": "...", "intended_change": "..."}'
                    ),
                })
                _, forced = _call_llm(
                    llm_client=llm_client,
                    system=system_prompt,
                    messages=state.messages,
                    user_msg="",
                    category="v3_worker_diagnose_plan_forced",
                )
                # Don't double-append user_msg
                state.messages.pop(-2) if state.messages[-2]["content"] == "" else None
                diagnosis = str(forced.get("diagnosis") or "")
                intended_change = str(forced.get("intended_change") or "")
                break
            if tool == "inspect_topic":
                cid = parsed.get("cluster_id")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    cid_int = -1
                result = _inspect_topic(iteration, cid_int, enriched_by_id=enriched_pool)
                state.messages.append({
                    "role": "user",
                    "content": f"[tool result: inspect_topic({cid_int})]\n{result}",
                })
            elif tool == "inspect_paper":
                title = str(parsed.get("title") or "")
                if not title:
                    state.messages.append({
                        "role": "user",
                        "content": "[tool error] inspect_paper needs a 'title' string.",
                    })
                else:
                    result = _inspect_paper(title, s2_client=s2_client, iteration=iteration)
                    state.messages.append({
                        "role": "user",
                        "content": f"[tool result: inspect_paper]\n{result}",
                    })
            else:
                state.messages.append({
                    "role": "user",
                    "content": (
                        "Unknown tool. Respond with one of: "
                        '{"tool": "inspect_topic", "cluster_id": N}, '
                        '{"tool": "inspect_paper", "title": "..."}, '
                        '{"tool": "plan", "diagnosis": "...", "intended_change": "..."}'
                    ),
                })
        last_diagnosis = diagnosis
        last_intended_change = intended_change

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="iter_summary",
            args={"iter": iter_idx, "query": query},
            result={
                "total_count": iteration.total_count,
                "fetched": iteration.fetched_count,
                "new": iteration.diff_new,
                "seen": iteration.diff_seen,
                "n_clusters": len(iteration.clusters),
                "diagnosis": diagnosis[:200],
                "intended_change": intended_change[:200],
            },
        )

    # Final aggregation
    all_ids = state.aggregate_paper_ids
    # Hydrate union for final topic model + top-cited
    union_papers: list[dict] = []
    for pid in all_ids[:_ENRICH_CAP]:
        rec = enriched_pool.get(pid)
        if rec is not None:
            union_papers.append(rec)
    final_sorted = sorted(union_papers, key=lambda p: -(p.get("citationCount") or 0))
    top_titles_final = [p["title"] for p in final_sorted[:30] if p.get("title")]
    clusters_final = topic_model(union_papers, k=config.topic_k) if union_papers else []

    # Final LLM summary (best-effort; not fatal if fails)
    summary_text = ""
    try:
        _, parsed = _call_llm(
            llm_client=llm_client,
            system=system_prompt,
            messages=state.messages,
            user_msg=WORKER_FINAL_SUMMARY.format(
                n_unique=len(all_ids),
                clusters=render_clusters(clusters_final),
            ),
            category="v3_worker_final",
        )
        summary_text = str(parsed.get("summary") or "")
    except Exception:  # noqa: BLE001
        summary_text = ""

    return SubTopicResultV3(
        spec_id=spec.id,
        description=spec.description,
        status="success" if state.iterations else "failed",
        paper_ids=all_ids,
        iterations=state.iterations,
        top_titles_final=top_titles_final,
        clusters_final=clusters_final,
        failure_reason="" if state.iterations else "no iterations completed",
    )
