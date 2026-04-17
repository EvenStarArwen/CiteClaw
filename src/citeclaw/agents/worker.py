"""Sub-topic worker loop for v2 ExpandBySearch.

Given a :class:`~citeclaw.agents.state.SubTopicSpec`, runs a
capped-iteration LLM loop where the model issues structured-output
tool calls that flow through
:class:`~citeclaw.agents.tool_dispatch.WorkerDispatcher`. The loop
terminates when the model calls ``done`` successfully, the budget is
exhausted, or ``worker_max_turns`` is reached.

Never called directly by users — the supervisor dispatches it per
sub-topic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from citeclaw.agents.search_tools import register_worker_tools
from citeclaw.agents.state import (
    AgentConfig,
    QueryAngleResult,
    StructuralPriors,
    SubTopicResult,
    SubTopicSpec,
    WorkerState,
)
from citeclaw.agents.tool_dispatch import WorkerDispatcher, is_error
from citeclaw.prompts.search_agent_worker import (
    RESPONSE_SCHEMA,
    SYSTEM,
    USER_TEMPLATE_CONTINUE,
    USER_TEMPLATE_FIRST,
    render_seed_block,
    render_structural_priors,
)

if TYPE_CHECKING:
    from citeclaw.agents.dataframe_store import DataFrameStore
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.context import Context

log = logging.getLogger("citeclaw.agents.worker")


def run_sub_topic_worker(
    *,
    worker_id: str,
    spec: SubTopicSpec,
    priors: StructuralPriors,
    topic_description: str,
    filter_summary: str,
    seed_papers: list[dict[str, Any]],
    llm_client: "LLMClient",
    ctx: "Context",
    dataframe_store: "DataFrameStore",
    agent_config: AgentConfig,
    logger: "SearchLogger",
) -> SubTopicResult:
    """Run one sub-topic worker end-to-end.

    Returns a :class:`SubTopicResult`. Does NOT mutate any shared
    state other than (a) the DataFrameStore (via the dispatcher,
    cleared at end) and (b) ctx.budget (via LLM + S2 calls).
    """
    state = WorkerState(sub_topic_id=spec.id)
    dispatcher = WorkerDispatcher(
        worker_state=state,
        dataframe_store=dataframe_store,
        agent_config=agent_config,
        ctx=ctx,
        worker_id=worker_id,
    )
    register_worker_tools(dispatcher)

    logger.log_worker_started(
        worker_id=worker_id,
        spec_id=spec.id,
        description=spec.description,
        initial_query_sketch=spec.initial_query_sketch,
        reference_papers=list(spec.reference_papers),
    )

    priors_block = render_structural_priors({
        "year_min": priors.year_min,
        "year_max": priors.year_max,
        "required_keywords": priors.required_keywords,
        "excluded_keywords": priors.excluded_keywords,
        "fields_of_study": priors.fields_of_study,
        "venue_filters": priors.venue_filters,
    })
    seed_block = render_seed_block(seed_papers if agent_config.share_seeds_with_agents else [])
    reference_str = list(spec.reference_papers) or "(none)"

    initial_user = USER_TEMPLATE_FIRST.format(
        sub_topic_id=spec.id,
        sub_topic_description=spec.description,
        initial_query_sketch=spec.initial_query_sketch,
        reference_papers=reference_str,
        topic_description=topic_description,
        filter_summary=filter_summary,
        seed_block=seed_block,
        structural_priors=priors_block,
        max_turns=agent_config.worker_max_turns,
        max_angles_per_worker=agent_config.max_angles_per_worker,
        max_refinement_per_angle=agent_config.max_refinement_per_angle,
    )

    last_tool_result: dict[str, Any] | None = None
    last_tool_name: str = ""
    last_active_fp: str | None = None
    turn = 0
    failure_reason = ""

    while turn < agent_config.worker_max_turns:
        turn += 1
        # Render user message for this turn.
        if turn == 1:
            user_msg = initial_user
        else:
            user_msg = USER_TEMPLATE_CONTINUE.format(
                tool_results=_render_tool_result(last_tool_name, last_tool_result),
                n_angles=len(state.angles),
                max_angles_per_worker=agent_config.max_angles_per_worker,
                active_angle=(state.active_fingerprint or "(none)"),
                n_cumulative=len(state.cumulative_paper_ids),
                turn=turn,
                max_turns=agent_config.worker_max_turns,
            )

        # LLM call with structured output.
        tokens_before = ctx.budget.llm_total_tokens
        try:
            resp = llm_client.call(
                SYSTEM,
                user_msg,
                category=f"expand_by_search_worker:{spec.id}",
                response_schema=RESPONSE_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 — LLM outage -> fail this worker
            log.warning("worker %s LLM call failed: %s", worker_id, exc)
            failure_reason = f"llm_call_failed: {exc}"
            break
        tokens_spent = ctx.budget.llm_total_tokens - tokens_before
        logger.log_worker_turn(
            worker_id=worker_id,
            turn=turn,
            system=SYSTEM,
            user=user_msg,
            response_text=resp.text or "",
            reasoning=resp.reasoning_content or "",
            tokens_in=None,  # budget tracks tokens per-category, not per-call
            tokens_out=tokens_spent,
        )

        # Parse the structured response.
        try:
            decoded = _parse_tool_call(resp.text or "")
        except _ParseError as exc:
            # Feed the parse error back to the model as a "tool result".
            last_tool_name = "(parse_error)"
            last_tool_result = {
                "error": "could not parse your response as JSON",
                "hint": str(exc),
            }
            continue

        tool_name = decoded.get("tool_name", "")
        tool_args = decoded.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        # Track angle transitions (before dispatch, so the log sees the
        # "from" state).
        prev_fp = last_active_fp
        # Dispatch.
        result = dispatcher.dispatch(tool_name, tool_args)
        logger.log_tool_call(
            scope=f"worker:{worker_id}",
            turn=turn,
            tool_name=tool_name,
            args=tool_args,
            result=result,
        )
        last_tool_name = tool_name
        last_tool_result = result
        # Log angle transition if the active fingerprint changed.
        new_fp = state.active_fingerprint
        if new_fp is not None and new_fp != prev_fp:
            angle = state.angles.get(new_fp)
            logger.log_angle_transition(
                worker_id=worker_id,
                from_fingerprint=prev_fp,
                to_fingerprint=new_fp,
                query=(angle.query if angle else ""),
            )
            last_active_fp = new_fp

        if dispatcher.done_called:
            break
        # Budget check — if the pipeline's shared budget is exhausted,
        # abort this worker cleanly.
        if ctx.budget.is_exhausted(ctx.config):
            failure_reason = "budget_exhausted"
            break

    # Clean up the DataFrame store for this worker.
    dispatcher.store.drop_all_for_worker(worker_id)

    # Build the SubTopicResult.
    done_result = dispatcher.done_result
    if done_result:
        assessment = done_result.get("coverage_assessment")
        paper_ids = list(done_result.get("paper_ids") or [])
        summary = done_result.get("summary") or ""
        status = "success"
    else:
        assessment = None
        paper_ids = sorted(state.cumulative_paper_ids)
        if failure_reason == "budget_exhausted":
            status = "budget_exhausted"
            summary = "Worker aborted: shared budget exhausted."
        elif failure_reason:
            status = "failed"
            summary = f"Worker failed: {failure_reason}"
        else:
            status = "failed"
            summary = "Worker hit max_turns without calling done()."
            failure_reason = "max_turns_without_done"

    # Assemble per-angle QueryAngleResults.
    angle_results: list[QueryAngleResult] = []
    for fp, angle in state.angles.items():
        angle_results.append(QueryAngleResult(
            query=angle.query,
            filters=dict(angle.filters),
            fingerprint=fp,
            n_fetched=angle.n_fetched or 0,
            total_in_corpus=angle.total_in_corpus or 0,
            papers_added_to_cumulative=(angle.n_fetched or 0),
            refinement_count=angle.refinement_count,
            topic_model_ran=angle.checked_topic_model,
            inspection_notes=angle.inspection_notes,
        ))

    logger.log_worker_finished(
        worker_id=worker_id,
        spec_id=spec.id,
        status=status,
        n_paper_ids=len(paper_ids),
        coverage_assessment=assessment,
        summary=summary,
        turns_used=turn,
        failure_reason=failure_reason,
    )

    return SubTopicResult(
        spec_id=spec.id,
        status=status,
        paper_ids=paper_ids,
        coverage_assessment=assessment,
        summary=summary,
        turns_used=turn,
        query_angles=angle_results,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _parse_tool_call(text: str) -> dict[str, Any]:
    """Parse a worker response into ``{tool_name, tool_args, reasoning}``.

    Accepts:
    - Bare JSON object.
    - ```json fenced``` block.
    - Object with leading whitespace or prose (strips to first ``{``).
    """
    if not text:
        raise _ParseError("empty response")
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Strip any leading prose before the first '{'.
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    try:
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise _ParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _ParseError("response must be a JSON object")
    if "tool_name" not in data:
        raise _ParseError(
            "missing 'tool_name' — every response must be "
            "{reasoning, tool_name, tool_args}"
        )
    return data


def _render_tool_result(tool_name: str, result: dict[str, Any] | None) -> str:
    """Render the previous turn's tool result for the next user message."""
    if result is None:
        return "(no prior tool call)"
    if is_error(result):
        return (
            f"**Previous call**: `{tool_name}`\n"
            f"**Result**: ❌ ERROR — {result.get('error', '')}\n"
            f"**Hint**: {result.get('hint', '')}"
        )
    # Short non-error results rendered as compact JSON; long ones summarised.
    try:
        blob = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        blob = str(result)
    if len(blob) > 6000:
        # Truncate the middle so head + tail both survive.
        head = blob[:3000]
        tail = blob[-1500:]
        blob = f"{head}\n... [TRUNCATED {len(blob) - 4500} chars] ...\n{tail}"
    return (
        f"**Previous call**: `{tool_name}`\n"
        f"**Result**:\n```json\n{blob}\n```"
    )


__all__ = ["run_sub_topic_worker"]
