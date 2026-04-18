"""V3 supervisor — plan-only agent, no query writing.

Supervisor input: parent ``topic_description`` only (no seeds, no
filter summary). Output: ``set_strategy`` with 3-to-max_subtopics
sub-topics, each ``{id, description}``. Then dispatches one worker per
sub-topic; after each worker returns, the supervisor sees:

- pairwise overlap matrix (Jaccard % between workers' paper_id sets)
- each dispatched sub-topic's id + description (so it remembers what
  worker 3 was supposed to cover N turns ago)
- the returned worker's top clusters + top-cited titles

The supervisor cannot see individual paper abstracts or the worker's
per-iter query log — only aggregate-level summaries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from citeclaw.agents.v3.state import (
    AgentConfigV3,
    StrategyV3,
    SubTopicResultV3,
    SubTopicSpecV3,
    SupervisorStateV3,
)
from citeclaw.agents.v3.worker import run_v3_worker
from citeclaw.agents.v3.analysis import render_clusters
from citeclaw.prompts.search_agent_v3 import (
    SUPERVISOR_SYSTEM,
    SUPERVISOR_USER_CONTINUE,
    SUPERVISOR_USER_FIRST,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.clients.s2.api import SemanticScholarClient

log = logging.getLogger("citeclaw.agents.v3.supervisor")


# ---------------------------------------------------------------------------
# JSON extraction (same tolerant shape as worker)
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


# ---------------------------------------------------------------------------
# Overlap + state rendering
# ---------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _render_overlap_matrix(results: list[SubTopicResultV3]) -> str:
    if len(results) < 2:
        return "  (need at least 2 dispatched workers to compute overlaps)"
    id_sets = [set(r.paper_ids) for r in results]
    lines: list[str] = []
    for i in range(1, len(results)):
        row_parts = []
        for j in range(i):
            jac = _jaccard(id_sets[i], id_sets[j]) * 100
            row_parts.append(f"vs {results[j].spec_id}: {jac:.0f}%")
        lines.append(f"  {results[i].spec_id}  " + "  ".join(row_parts))
    return "\n".join(lines)


def _render_dispatched(results: list[SubTopicResultV3]) -> str:
    if not results:
        return "  (none)"
    lines: list[str] = []
    for r in results:
        desc = r.description.strip().replace("\n", " ")
        if len(desc) > 100:
            desc = desc[:100].rstrip() + "..."
        lines.append(
            f"  - {r.spec_id} [{r.status}, {len(r.paper_ids)} papers]: {desc}"
        )
        # Cluster keywords summary
        if r.clusters_final:
            kw_summary = "; ".join(
                f"{c.count}:{','.join(c.keywords[:4])}"
                for c in r.clusters_final[:5]
            )
            lines.append(f"      clusters: {kw_summary}")
        # Top-cited (first 5)
        if r.top_titles_final:
            lines.append(
                "      top-cited: "
                + " | ".join(t[:60] for t in r.top_titles_final[:5])
            )
    return "\n".join(lines)


def _render_remaining(
    strategy: StrategyV3 | None,
    dispatched_ids: set[str],
) -> str:
    if strategy is None:
        return "  (none)"
    remaining = [s for s in strategy.sub_topics if s.id not in dispatched_ids]
    if not remaining:
        return "  (none)"
    lines: list[str] = []
    for s in remaining:
        desc = s.description.strip().replace("\n", " ")
        if len(desc) > 100:
            desc = desc[:100].rstrip() + "..."
        lines.append(f"  - {s.id}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class _SupervisorError(Exception):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


def _validate_sub_topic_entry(raw: Any, *, idx: int, existing: set[str]) -> SubTopicSpecV3:
    if not isinstance(raw, dict):
        raise _SupervisorError(f"sub_topics[{idx}] must be an object", "{id, description}")
    sid = str(raw.get("id") or "").strip()
    desc = str(raw.get("description") or "").strip()
    if not sid:
        raise _SupervisorError(f"sub_topics[{idx}].id missing", "unique slug")
    if sid in existing:
        raise _SupervisorError(
            f"sub_topics[{idx}].id {sid!r} duplicate", "each id must be unique"
        )
    if not desc:
        raise _SupervisorError(
            f"sub_topics[{idx}].description missing",
            "1-2 sentence English description",
        )
    return SubTopicSpecV3(id=sid, description=desc)


def _handle_set_strategy(args: dict, state: SupervisorStateV3, max_subtopics: int) -> dict:
    if state.strategy is not None:
        raise _SupervisorError(
            "set_strategy already called",
            "strategy is locked; use add_sub_topics or dispatch",
        )
    raw = args.get("sub_topics")
    if not isinstance(raw, list) or not raw:
        raise _SupervisorError(
            "'sub_topics' must be a non-empty list of {id, description}",
            f"3 to {max_subtopics} entries",
        )
    if len(raw) > max_subtopics:
        raise _SupervisorError(
            f"too many sub-topics ({len(raw)})",
            f"max {max_subtopics}; consolidate similar ones",
        )
    seen: set[str] = set()
    specs: list[SubTopicSpecV3] = []
    for idx, item in enumerate(raw):
        spec = _validate_sub_topic_entry(item, idx=idx, existing=seen)
        seen.add(spec.id)
        specs.append(spec)
    state.strategy = StrategyV3(sub_topics=tuple(specs))
    return {
        "acknowledged": True,
        "n_sub_topics": len(specs),
        "sub_topic_ids": [s.id for s in specs],
    }


def _handle_add_sub_topics(args: dict, state: SupervisorStateV3, max_subtopics: int) -> dict:
    if state.strategy is None:
        raise _SupervisorError("call set_strategy first", "")
    raw = args.get("sub_topics")
    if not isinstance(raw, list) or not raw:
        raise _SupervisorError(
            "'sub_topics' must be a non-empty list of {id, description}", ""
        )
    existing_ids = {s.id for s in state.strategy.sub_topics}
    new_specs: list[SubTopicSpecV3] = []
    for idx, item in enumerate(raw):
        spec = _validate_sub_topic_entry(item, idx=idx, existing=existing_ids)
        existing_ids.add(spec.id)
        new_specs.append(spec)
    combined = len(state.strategy.sub_topics) + len(new_specs)
    if combined > max_subtopics:
        raise _SupervisorError(
            f"combined strategy would have {combined} sub-topics",
            f"cap is {max_subtopics}",
        )
    state.strategy = StrategyV3(
        sub_topics=tuple(list(state.strategy.sub_topics) + new_specs),
    )
    return {
        "acknowledged": True,
        "added": [s.id for s in new_specs],
        "n_sub_topics_total": len(state.strategy.sub_topics),
    }


def _handle_dispatch(
    args: dict,
    state: SupervisorStateV3,
    *,
    config: AgentConfigV3,
    s2_client: Any,
    llm_client: Any,
    logger: Any,
) -> dict:
    spec_id = args.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        raise _SupervisorError("missing 'spec_id'", "")
    if state.strategy is None:
        raise _SupervisorError("call set_strategy first", "")
    spec = next((s for s in state.strategy.sub_topics if s.id == spec_id), None)
    if spec is None:
        raise _SupervisorError(
            f"spec_id {spec_id!r} not in strategy",
            f"valid ids: {[s.id for s in state.strategy.sub_topics]}",
        )
    if any(r.spec_id == spec_id for r in state.sub_topic_results):
        raise _SupervisorError(
            f"spec_id {spec_id!r} already dispatched",
            "each sub-topic runs at most once",
        )
    log.info("V3 supervisor dispatching worker for %s", spec_id)
    result = run_v3_worker(
        spec=spec,
        config=config,
        s2_client=s2_client,
        llm_client=llm_client,
        logger=logger,
    )
    state.sub_topic_results.append(result)
    return {
        "spec_id": result.spec_id,
        "status": result.status,
        "n_papers": len(result.paper_ids),
        "n_iterations": len(result.iterations),
    }


def _handle_done(args: dict) -> dict:
    summary = str(args.get("summary") or "")
    return {"acknowledged": True, "summary": summary, "closed": True}


# ---------------------------------------------------------------------------
# Supervisor main loop
# ---------------------------------------------------------------------------


def run_v3_supervisor(
    *,
    topic_description: str,
    config: AgentConfigV3,
    s2_client: "SemanticScholarClient",
    llm_client: "LLMClient",
    logger: "SearchLogger",
    supervisor_max_turns: int = 20,
) -> tuple[SupervisorStateV3, list[str]]:
    state = SupervisorStateV3()
    system_prompt = SUPERVISOR_SYSTEM.format(max_subtopics=config.max_subtopics)

    messages: list[dict[str, str]] = []
    last_tool_name = ""
    last_tool_result: dict[str, Any] = {}

    for turn in range(1, supervisor_max_turns + 1):
        state.turn_index = turn
        if turn == 1:
            user_msg = SUPERVISOR_USER_FIRST.format(
                topic_description=topic_description,
                max_subtopics=config.max_subtopics,
                supervisor_max_turns=supervisor_max_turns,
                max_iter=config.max_iter,
            )
        else:
            dispatched_ids = {r.spec_id for r in state.sub_topic_results}
            tool_res_str = (
                f"Previous tool `{last_tool_name}` result:\n{json.dumps(last_tool_result, indent=2)}"
                if last_tool_name
                else ""
            )
            user_msg = SUPERVISOR_USER_CONTINUE.format(
                tool_results=tool_res_str,
                dispatched_block=_render_dispatched(state.sub_topic_results),
                remaining_block=_render_remaining(state.strategy, dispatched_ids),
                overlap_matrix=_render_overlap_matrix(state.sub_topic_results),
            )

        messages.append({"role": "user", "content": user_msg})
        joined = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        try:
            resp = llm_client.call(
                system_prompt,
                joined,
                category="v3_supervisor",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 supervisor LLM call failed: %s", exc)
            break
        text = (resp.text or "").strip()
        messages.append({"role": "assistant", "content": text})
        parsed = _extract_json(text)
        tool_name = str(parsed.get("tool_name") or "")
        tool_args = parsed.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        try:
            if tool_name == "set_strategy":
                result = _handle_set_strategy(tool_args, state, config.max_subtopics)
            elif tool_name == "add_sub_topics":
                result = _handle_add_sub_topics(tool_args, state, config.max_subtopics)
            elif tool_name == "dispatch_sub_topic_worker":
                result = _handle_dispatch(
                    tool_args, state,
                    config=config,
                    s2_client=s2_client,
                    llm_client=llm_client,
                    logger=logger,
                )
            elif tool_name == "done":
                result = _handle_done(tool_args)
            else:
                result = {
                    "error": f"unknown tool {tool_name!r}",
                    "hint": "one of: set_strategy, add_sub_topics, dispatch_sub_topic_worker, done",
                }
        except _SupervisorError as exc:
            result = {"error": str(exc), "hint": exc.hint}

        logger.log_tool_call(
            scope="v3_supervisor",
            turn=turn,
            tool_name=tool_name,
            args=tool_args,
            result=result,
        )
        last_tool_name = tool_name
        last_tool_result = result

        if tool_name == "done" and result.get("closed"):
            break

    aggregate = state.aggregate_paper_ids()
    return state, aggregate
