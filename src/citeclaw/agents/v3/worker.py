"""V3 worker — tutorial-style 6-phase loop with ISOLATED context.

Per iteration each phase issues a fresh LLM call (system + a
phase-specific user message) rather than accumulating one long
conversation. The heavy data dump (top-10 cited + random-10 +
topic clusters + query tree) is shown ONLY in the diagnose_plan
phase, paired with one-sentence recaps of the earlier diagnostic
answers.

The worker writes queries in natural-language (AND / OR / NOT);
the translator converts to Lucene before hitting S2, and translates
back when the query is shown in the write-next history block.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, TYPE_CHECKING

from citeclaw.agents.v3.analysis import (
    build_query_tree,
    diff_vs_prev,
    render_clusters,
    render_query_tree,
    topic_model,
)
from citeclaw.agents.v3.query_translate import to_lucene, to_natural
from citeclaw.agents.v3.state import (
    AgentConfigV3,
    QueryIterationV3,
    SubTopicResultV3,
    SubTopicSpecV3,
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


_ENRICH_CAP = 2000
_MAX_DIAGNOSE_TOOL_CALLS = 3
_RANDOM_SAMPLE_SIZE = 10
_TOP_ABSTRACTS_SIZE = 10


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


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


def _single_call(
    *,
    llm_client: "LLMClient",
    system: str,
    user: str,
    category: str,
) -> tuple[str, dict]:
    """Fresh LLM call — no history accumulation. Returns (text, parsed_json)."""
    try:
        resp = llm_client.call(system, user, category=category)
    except Exception as exc:  # noqa: BLE001
        log.warning("V3 LLM call failed (%s): %s", category, exc)
        return "", {}
    text = (resp.text or "").strip()
    return text, _extract_json(text)


# ---------------------------------------------------------------------------
# S2 fetch + analyse one query
# ---------------------------------------------------------------------------


def _fetch_and_analyse(
    *,
    query_lucene: str,
    iter_idx: int,
    s2_client: "SemanticScholarClient",
    prior_iters: list[QueryIterationV3],
    config: AgentConfigV3,
) -> tuple[list[str], int, list, list[dict]]:
    """Fetch paper_ids for the query (up to max_papers_per_query),
    enrich the first _ENRICH_CAP, compute query tree and return.

    Returns (paper_ids, total_count, query_tree_nodes, enriched_dicts).
    """
    paper_ids: list[str] = []
    total = 0
    token: str | None = None
    per_call_limit = 500
    while len(paper_ids) < config.max_papers_per_query:
        try:
            resp = s2_client.search_bulk(query=query_lucene, limit=per_call_limit, token=token)
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 search_bulk failed: %s", exc)
            break
        data = resp.get("data") or []
        if token is None:
            total = int(resp.get("total") or 0)
        for row in data:
            pid = row.get("paperId")
            if pid:
                paper_ids.append(pid)
        token = resp.get("token")
        if not token or not data:
            break
    paper_ids = paper_ids[: config.max_papers_per_query]

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

    query_tree = build_query_tree(query_lucene, full_total=total, s2_client=s2_client)
    return paper_ids, total, query_tree, enriched


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _render_top10_abstracts(enriched_sorted: list[dict]) -> str:
    if not enriched_sorted:
        return "  (no enriched papers)"
    lines: list[str] = []
    for i, p in enumerate(enriched_sorted[:_TOP_ABSTRACTS_SIZE], start=1):
        cc = p.get("citationCount") or 0
        title = (p.get("title") or "").strip() or "(no title)"
        abstract = _truncate(p.get("abstract") or "(no abstract)", 400)
        lines.append(f"  {i}. [{cc}c] {title}")
        lines.append(f"     {abstract}")
    return "\n".join(lines)


def _render_random10(enriched: list[dict], seed: int) -> str:
    if not enriched:
        return "  (no enriched papers)"
    rng = random.Random(seed)
    n = len(enriched)
    k = min(_RANDOM_SAMPLE_SIZE, n)
    sample_idx = rng.sample(range(n), k)
    lines: list[str] = []
    for i, idx in enumerate(sample_idx, start=1):
        p = enriched[idx]
        cc = p.get("citationCount") or 0
        title = (p.get("title") or "").strip() or "(no title)"
        abstract = _truncate(p.get("abstract") or "(no abstract)", 400)
        lines.append(f"  {i}. [{cc}c] {title}")
        lines.append(f"     {abstract}")
    return "\n".join(lines)


def _render_top100_titles(enriched_sorted: list[dict]) -> str:
    if not enriched_sorted:
        return "  (no enriched papers)"
    lines: list[str] = []
    for i, p in enumerate(enriched_sorted[:100], start=1):
        title = (p.get("title") or "").strip() or "(no title)"
        lines.append(f"  {i}. {title}")
    return "\n".join(lines)


def _render_history(iters: list[QueryIterationV3]) -> str:
    """Prior-iter history for the write-next prompt — includes the
    natural-language query (translated back), total_count, the query
    tree, and the iter's one-sentence diagnosis. Keeps the worker from
    regressing to queries it already tried and from re-discovering
    problems it already flagged."""
    if not iters:
        return "  (this is iteration 0 — no prior iterations)"
    blocks: list[str] = []
    for it in iters:
        q_nl = to_natural(it.query_lucene) or it.query
        problems = " | ".join(
            p for p in (
                it.reasoning_total,
                it.reasoning_clusters,
                it.reasoning_top100,
            ) if p
        ) or "(no issues recorded)"
        tree = render_query_tree(it.query_tree)
        block = (
            f"iter {it.iter_idx}:\n"
            f"  query: {q_nl}\n"
            f"  total: {it.total_count} (new +{it.diff_new}, seen {it.diff_seen})\n"
            f"  query tree:\n{tree}\n"
            f"  issues found: {problems}\n"
            f"  diagnosis: {it.diagnosis or '(no plan)'}\n"
            f"  intended change: {it.intended_change or '(n/a)'}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Diagnose-plan tool helpers
# ---------------------------------------------------------------------------


def _inspect_topic(
    iteration: QueryIterationV3,
    cluster_id: int,
    *,
    enriched_by_id: dict[str, dict],
) -> str:
    cluster = next((c for c in iteration.clusters if c.cluster_id == cluster_id), None)
    if cluster is None:
        valid = [c.cluster_id for c in iteration.clusters]
        return f"(no cluster {cluster_id}; valid ids: {valid})"
    lines = [
        f"Cluster {cluster_id} — {cluster.count} papers.",
        f"Keywords: {', '.join(cluster.keywords[:8])}",
    ]
    for title in cluster.representative_titles[:5]:
        match = next(
            (p for p in enriched_by_id.values() if (p.get("title") or "").strip() == title.strip()),
            None,
        )
        if match:
            abs_text = _truncate(match.get("abstract") or "", 400)
            lines.append(f"  · {title}")
            lines.append(f"    {abs_text}")
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
    abstract = _truncate(matched.get("abstract") or "", 400)
    in_results = "YES" if matched_id in set(iteration.paper_ids) else "NO"
    return (
        f"Matched paper: {matched_title}\n"
        f"Paper ID: {matched_id}\n"
        f"In current iter's fetched set: {in_results}\n"
        f"Abstract: {abstract or '(no abstract)'}"
    )


# ---------------------------------------------------------------------------
# Main worker loop
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

    for iter_idx in range(config.max_iter):
        # -- Phase 1: PROPOSE ------------------------------------------------
        if iter_idx == 0:
            user_msg = WORKER_PROPOSE_FIRST.format(description=spec.description)
        else:
            history_block = _render_history(state.iterations)
            user_msg = WORKER_WRITE_NEXT.format(
                history_block=history_block,
                diagnosis=state.iterations[-1].diagnosis or "(none)",
                intended_change=state.iterations[-1].intended_change or "(none)",
            )
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=user_msg,
            category="v3_worker_propose",
        )
        query_nl = (parsed.get("query") or "").strip()
        if not query_nl:
            log.warning(
                "V3 worker %s: iter %d propose returned no query",
                spec.id, iter_idx,
            )
            break
        query_lucene = to_lucene(query_nl)

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="propose_query",
            args={"iter": iter_idx},
            result={"query_nl": query_nl, "query_lucene": query_lucene},
        )

        # -- Phase 2: FETCH + ANALYSE (system-side, no LLM) -----------------
        paper_ids, total, query_tree, enriched = _fetch_and_analyse(
            query_lucene=query_lucene,
            iter_idx=iter_idx,
            s2_client=s2_client,
            prior_iters=state.iterations,
            config=config,
        )
        for p in enriched:
            enriched_pool.setdefault(p["paperId"], p)

        enriched_sorted = sorted(enriched, key=lambda p: -(p.get("citationCount") or 0))
        top_titles = [p["title"] for p in enriched_sorted[:100] if p.get("title")]
        clusters = topic_model(enriched, s2_client=s2_client, k=config.topic_k)

        prior_ids_set: set[str] = set()
        for it in state.iterations:
            prior_ids_set.update(it.paper_ids)
        diff_new, diff_seen = diff_vs_prev(paper_ids, prior_ids_set)

        iteration = QueryIterationV3(
            iter_idx=iter_idx,
            query=query_nl,
            query_lucene=query_lucene,
            total_count=total,
            fetched_count=len(paper_ids),
            paper_ids=paper_ids,
            query_tree=query_tree,
            clusters=clusters,
            top_titles_100=top_titles,
            diff_new=diff_new,
            diff_seen=diff_seen,
        )

        # -- Phase 3: CHECK_TOTAL (fresh, isolated context) ----------------
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_CHECK_TOTAL.format(
                query=query_nl,
                total=total,
                diff_new=diff_new,
                diff_seen=diff_seen,
                query_tree=render_query_tree(query_tree),
            ),
            category="v3_worker_check_total",
        )
        iteration.reasoning_total = str(parsed.get("reasoning") or "").strip()

        # -- Phase 4: CHECK_CLUSTERS ---------------------------------------
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_CHECK_CLUSTERS.format(
                query=query_nl,
                description=spec.description,
                clusters=render_clusters(clusters),
            ),
            category="v3_worker_check_clusters",
        )
        ans_clusters_reasoning = str(parsed.get("reasoning") or "").strip()
        unrelated = parsed.get("unrelated_clusters") or []
        missing = parsed.get("missing_areas") or []
        extras: list[str] = []
        if unrelated:
            extras.append(f"unrelated_clusters={unrelated}")
        if missing:
            extras.append(f"missing_areas={missing}")
        iteration.reasoning_clusters = (
            ans_clusters_reasoning + ((" (" + "; ".join(extras) + ")") if extras else "")
        ).strip()

        # -- Phase 5: CHECK_TOP100 -----------------------------------------
        titles_block = _render_top100_titles(enriched_sorted)
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_CHECK_TOP100.format(
                query=query_nl,
                titles=titles_block,
            ),
            category="v3_worker_check_top100",
        )
        ans_top100_reasoning = str(parsed.get("reasoning") or "").strip()
        off_topic = parsed.get("off_topic_titles") or []
        if off_topic:
            ans_top100_reasoning += f" (off_topic_titles={off_topic[:5]})"
        iteration.reasoning_top100 = ans_top100_reasoning.strip()

        # -- Phase 6: DIAGNOSE + PLAN (fresh; loop with tool calls) --------
        top10_block = _render_top10_abstracts(enriched_sorted)
        random10_block = _render_random10(enriched, seed=iter_idx)
        clusters_block = render_clusters(clusters)

        base_diagnose_msg = WORKER_DIAGNOSE_PLAN.format(
            query=query_nl,
            ans_total=iteration.reasoning_total or "(no answer)",
            ans_clusters=iteration.reasoning_clusters or "(no answer)",
            ans_top100=iteration.reasoning_top100 or "(no answer)",
            total=total,
            query_tree=render_query_tree(query_tree),
            top10_block=top10_block,
            random10_block=random10_block,
            clusters=clusters_block,
        )
        tool_trail: list[str] = []
        diagnosis = ""
        intended_change = ""
        for tool_round in range(_MAX_DIAGNOSE_TOOL_CALLS + 1):
            user_msg = base_diagnose_msg
            if tool_trail:
                user_msg += "\n\n# Tool results so far this phase\n" + "\n\n".join(tool_trail)
            _, parsed = _single_call(
                llm_client=llm_client,
                system=system_prompt,
                user=user_msg,
                category="v3_worker_diagnose_plan",
            )
            tool = (parsed.get("tool") or "").strip()
            if tool == "plan":
                diagnosis = str(parsed.get("diagnosis") or "").strip()
                intended_change = str(parsed.get("intended_change") or "").strip()
                break
            if tool_round >= _MAX_DIAGNOSE_TOOL_CALLS:
                tool_trail.append(
                    "[system] You've used your tool-call budget. Commit a plan now: "
                    '{"tool": "plan", "diagnosis": "...", "intended_change": "..."}'
                )
                _, forced = _single_call(
                    llm_client=llm_client,
                    system=system_prompt,
                    user=base_diagnose_msg + "\n\n" + "\n\n".join(tool_trail),
                    category="v3_worker_diagnose_plan_forced",
                )
                diagnosis = str(forced.get("diagnosis") or "").strip()
                intended_change = str(forced.get("intended_change") or "").strip()
                break
            if tool == "inspect_topic":
                cid = parsed.get("cluster_id")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    cid_int = -1
                result = _inspect_topic(iteration, cid_int, enriched_by_id=enriched_pool)
                tool_trail.append(f"[inspect_topic({cid_int})]\n{result}")
            elif tool == "inspect_paper":
                title = str(parsed.get("title") or "")
                if not title:
                    tool_trail.append("[error] inspect_paper needs a 'title' string.")
                else:
                    result = _inspect_paper(title, s2_client=s2_client, iteration=iteration)
                    tool_trail.append(f"[inspect_paper({title!r})]\n{result}")
            else:
                tool_trail.append(
                    "[error] Unknown tool. Respond with inspect_topic / inspect_paper / plan."
                )
        iteration.diagnosis = diagnosis
        iteration.intended_change = intended_change

        state.iterations.append(iteration)

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="iter_summary",
            args={"iter": iter_idx, "query_nl": query_nl, "query_lucene": query_lucene},
            result={
                "total_count": total,
                "fetched": len(paper_ids),
                "new": diff_new,
                "seen": diff_seen,
                "n_clusters": len(clusters),
                "reasoning_total": iteration.reasoning_total[:200],
                "reasoning_clusters": iteration.reasoning_clusters[:200],
                "reasoning_top100": iteration.reasoning_top100[:200],
                "diagnosis": diagnosis[:200],
                "intended_change": intended_change[:200],
            },
        )

    # -- Final summary -----------------------------------------------------
    all_ids = state.aggregate_paper_ids
    union_papers = [enriched_pool[pid] for pid in all_ids[:_ENRICH_CAP] if pid in enriched_pool]
    final_sorted = sorted(union_papers, key=lambda p: -(p.get("citationCount") or 0))
    top_titles_final = [p["title"] for p in final_sorted[:30] if p.get("title")]
    clusters_final = (
        topic_model(union_papers, s2_client=s2_client, k=config.topic_k)
        if union_papers else []
    )

    summary_text = ""
    try:
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_FINAL_SUMMARY.format(
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
