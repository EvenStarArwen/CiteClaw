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
    state = WorkerState(sub_topic_id=spec.id, structural_priors=priors)
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
    auto_closed = False  # set True iff the penultimate-turn rescuer ran and succeeded

    while turn < agent_config.worker_max_turns:
        turn += 1
        # Render user message for this turn.
        if turn == 1:
            user_msg = initial_user
        else:
            active_query = "(none)"
            if state.active_angle is not None:
                aa = state.active_angle
                active_query = f"{aa.query!r}"
                if aa.filters:
                    active_query += f" with filters={aa.filters}"
            valid_next = _compute_valid_next_tools(
                state, agent_config, last_tool_name, turn=turn,
            )
            user_msg = USER_TEMPLATE_CONTINUE.format(
                sub_topic_id=spec.id,
                tool_results=_render_tool_result(last_tool_name, last_tool_result),
                n_angles=len(state.angles),
                max_angles_per_worker=agent_config.max_angles_per_worker,
                active_angle=(state.active_fingerprint or "(none)"),
                active_query=active_query,
                n_cumulative=len(state.cumulative_paper_ids),
                turn=turn,
                max_turns=agent_config.worker_max_turns,
                valid_next_tools=valid_next,
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
        # Auto-close safety net: on the penultimate turn, if the worker
        # has produced any cumulative papers, force a clean close. We
        # auto-abandon any angle whose checklist is incomplete (so the
        # done hook's precondition passes) and synthetically call
        # ``done`` with coverage_assessment=limited. Rationale: a
        # worker that fetched real papers but couldn't orchestrate the
        # verification cycle in 20 turns is still more useful than one
        # we reject for max_turns. Only kicks in on turn
        # ``worker_max_turns - 1`` so it never pre-empts a healthy run.
        if turn == agent_config.worker_max_turns - 1:
            before_done = dispatcher.done_called
            _attempt_auto_close(
                dispatcher, state, spec=spec, logger=logger,
                worker_id=worker_id, turn=turn,
            )
            if dispatcher.done_called and not before_done:
                # Flag it for the SubTopicResult so the supervisor can
                # tell an auto-rescue from a natural close.
                auto_closed = True
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
        auto_closed=auto_closed,
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


def _auto_close_fallback_fetch(dispatcher, spec: SubTopicSpec) -> None:
    """Last-chance fetch using the supervisor's ``initial_query_sketch``.

    Dispatched through the real tool layer so the query still goes
    through the S2 lint. Best-effort: any failure is swallowed and
    the caller will see the empty cumulative set.
    """
    sketch = spec.initial_query_sketch
    if not sketch:
        return
    sz = dispatcher.dispatch("check_query_size", {"query": sketch})
    if not isinstance(sz, dict) or "error" in sz:
        return
    if (sz.get("total") or 0) == 0:
        return
    fr = dispatcher.dispatch("fetch_results", {"query": sketch})
    if not isinstance(fr, dict) or "error" in fr:
        return
    dispatcher.dispatch("inspect_angle", {})


def _attempt_auto_close(
    dispatcher,
    state: WorkerState,
    *,
    spec: SubTopicSpec,
    logger: "SearchLogger",
    worker_id: str,
    turn: int,
) -> None:
    """Force a clean close on the penultimate turn.

    Steps:
      1. If the worker has zero fetched papers, skip — there's nothing
         to close and the natural "failed" status is correct.
      2. Abandon every angle whose inspection checklist is incomplete
         so the done hook doesn't reject on per-angle grounds.
      3. If no verification cycle has run, synthetically run one
         against the first reference paper (or any first_3_titles
         we can see in the log). Records a diagnose_miss with
         action_taken="accept_gap" for misses.
      4. Call ``done`` via the dispatcher with coverage_assessment
         inferred from remaining state.

    Every step is logged via the dispatcher so the transcript shows
    exactly what the auto-closer did. Best-effort: if anything fails
    we leave the worker in its current state and the normal
    max_turns failure path handles it.
    """
    had_cumulative_before = bool(state.cumulative_paper_ids)
    had_contains_before = any(e.get("tool") == "contains" for e in state.call_log)
    if not state.cumulative_paper_ids:
        # Last-chance fallback: run the supervisor's initial_query_sketch
        # through size + fetch + inspect. If that produces zero papers
        # too, the sub-topic is genuinely hard and we return cleanly as
        # a failure. If it produces papers, auto-close succeeds.
        _auto_close_fallback_fetch(dispatcher, spec)
        if not state.cumulative_paper_ids:
            return
    # Step 2: abandon any angle with an incomplete checklist.
    incomplete = [
        a for a in list(state.angles.values())
        if a.df_id is not None and not a.is_checklist_complete()
    ]
    for angle in incomplete:
        # Point active_angle at the incomplete angle, then abandon via
        # the dispatcher so the effect is consistent with a real
        # abandon call (drops df, removes papers, logs).
        state.active_fingerprint = angle.fingerprint
        dispatcher.dispatch("abandon_angle", {})
    # Emit a transparency event so postmortem tooling can tell
    # auto-closed workers from naturally-closed ones — the result
    # looks the same from the outside but the agent's coverage
    # judgement was bypassed. ``SubTopicResult.auto_closed=True`` is
    # set on the worker-loop side after the dispatch returns.
    try:
        logger.log_auto_close_invoked(
            worker_id=worker_id,
            spec_id=spec.id,
            turn=turn,
            had_cumulative=had_cumulative_before,
            had_verification=had_contains_before,
            angles_abandoned=len(incomplete),
        )
    except Exception:  # noqa: BLE001 — best-effort observability
        pass
    # Step 3: if no verification cycle has run yet, attempt one.
    has_contains = any(e.get("tool") == "contains" for e in state.call_log)
    if not has_contains:
        # Pick the first reference paper we have, or give up on verify.
        anchor_title = (
            spec.reference_papers[0] if spec.reference_papers else None
        )
        if anchor_title:
            sm = dispatcher.dispatch("search_match", {"title": anchor_title})
            if isinstance(sm, dict) and "error" not in sm and sm.get("match"):
                pid = sm["match"].get("paper_id")
                if isinstance(pid, str):
                    c = dispatcher.dispatch("contains", {"paper_id": pid})
                    if isinstance(c, dict) and c.get("contains") is False:
                        dispatcher.dispatch("diagnose_miss", {
                            "target_title": anchor_title,
                            "hypotheses": ["auto-closer: not reached within worker budget"],
                            "action_taken": "accept_gap",
                            "query_angles_used": [
                                a.query for a in state.angles.values() if a.query
                            ],
                        })
    # Step 4: force done.
    summary = (
        f"auto-closed on turn {turn} of {dispatcher.config.worker_max_turns} "
        f"(reached turn budget); {len(state.cumulative_paper_ids)} papers "
        f"fetched across {len(state.angles)} live angle(s)"
    )
    dispatcher.dispatch("done", {
        "paper_ids": sorted(state.cumulative_paper_ids),
        "coverage_assessment": "limited",
        "summary": summary,
    })


def _compute_valid_next_tools(
    state: WorkerState,
    cfg: "AgentConfig",
    last_tool_name: str,
    *,
    turn: int,
) -> str:
    """Derive the narrow set of tools that make sense from the current
    worker state. Rendered as a bullet list in the continuation user
    message so weaker models follow the checklist lane by default.

    The set is advisory: the dispatcher still accepts any registered
    tool, but the prompt makes the intended next action obvious.

    When the worker is near its turn budget (>= 75% of max_turns),
    ``done`` is strongly emphasised and exploration tools are
    demoted. When the angle cap is hit, only ``done`` and
    ``abandon_angle`` are recommended.
    """
    active = state.active_angle
    verification_done = any(
        e.get("tool") == "contains" for e in state.call_log
    )
    unresolved_miss = False
    for e in reversed(state.call_log):
        if e.get("tool") == "diagnose_miss":
            result = e.get("result") or {}
            if "error" in result:
                continue
            break
        if e.get("tool") == "contains":
            if (e.get("result") or {}).get("contains") is False:
                unresolved_miss = True
            break

    angle_cap_hit = len(state.angles) >= cfg.max_angles_per_worker
    time_pressure = turn >= int(cfg.worker_max_turns * 0.75)

    candidates: list[str] = []
    if unresolved_miss:
        candidates = ["diagnose_miss"]
    elif angle_cap_hit and active is None:
        # All slots used and nothing active → just close.
        candidates = ["done"]
    elif active is None:
        # No active angle: either open one or close.
        has_usable = any(a.is_checklist_complete() and a.df_id for a in state.angles.values())
        if time_pressure and has_usable:
            # Strongly prefer closing over opening another angle under time pressure.
            if verification_done:
                candidates = ["done"]
            else:
                candidates = ["search_match", "done"]
        elif angle_cap_hit:
            candidates = ["done"]
        else:
            candidates = ["check_query_size"]
            if has_usable:
                if verification_done:
                    candidates.append("done")
                else:
                    candidates.insert(0, "search_match")
    else:
        # An angle is active. Drive the checklist.
        if active.df_id is None:
            candidates = ["fetch_results", "abandon_angle"]
        elif not (active.checked_top_cited and active.checked_random and active.checked_years):
            # inspect_angle does all three in one call — strongly preferred.
            candidates = ["inspect_angle", "abandon_angle"]
        elif active.requires_topic_model and not active.checked_topic_model:
            # inspect_angle also covers topic_model. The agent likely
            # already called inspect_angle and it skipped topic_model
            # for some reason; fall back to the primitive.
            candidates = ["topic_model", "abandon_angle"]
        else:
            # Checklist on active complete.
            if not verification_done:
                candidates = ["search_match"]
                # Allow opening a new angle only if not under time pressure
                # AND we haven't hit the cap.
                if not angle_cap_hit and not time_pressure:
                    candidates.append("check_query_size")
                candidates.append("abandon_angle")
            else:
                # Verification done — closing is strongly preferred.
                if time_pressure or angle_cap_hit:
                    candidates = ["done"]
                else:
                    candidates = ["done", "check_query_size", "abandon_angle"]

    always_available = ["get_paper", "search_within_df"]
    pressure_note = ""
    if time_pressure:
        pressure_note = (
            f"\n\n⚠️  TIME PRESSURE: turn {turn} of {cfg.worker_max_turns} "
            f"— strongly prefer `done` over opening new angles. If the per-angle "
            f"checklist on the active angle is complete and a verification cycle "
            f"(search_match → contains) has run, call `done` now."
        )
    if angle_cap_hit:
        pressure_note += (
            f"\n\n⚠️  ANGLE CAP HIT: you have opened the maximum "
            f"{cfg.max_angles_per_worker} angles. check_query_size will reject. "
            f"Finish any outstanding inspection, then call `done` or "
            f"`abandon_angle` to free a slot."
        )
    return (
        "next-action candidates (strongly recommended): "
        + ", ".join(f"`{t}`" for t in candidates)
        + "; optional deep-dive: "
        + ", ".join(f"`{t}`" for t in always_available)
        + pressure_note
    )


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
