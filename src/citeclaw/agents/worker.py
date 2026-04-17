"""Sub-topic worker loop for ExpandBySearch (post-refactor).

Given a :class:`~citeclaw.agents.state.SubTopicSpec`, runs a
capped-iteration LLM loop where the model issues structured-output
tool calls that flow through
:class:`~citeclaw.agents.tool_dispatch.WorkerDispatcher`. The loop
terminates when the model calls ``done`` successfully, the budget is
exhausted, or ``worker_max_turns`` is reached.

The tool surface is 7 tools: ``check_query_size``, ``fetch_results``,
``query_diagnostics``, ``search_within_df``, ``get_paper``,
``diagnose_miss``, ``done``. Deterministic post-fetch work (sampling,
distributions, topic model, reference verification) is embedded in
``fetch_results`` — the model no longer orchestrates inspection.

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
    QueryResult,
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
    state = WorkerState(
        sub_topic_id=spec.id,
        structural_priors=priors,
        reference_papers=spec.reference_papers,
    )
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
        max_queries_per_worker=agent_config.max_queries_per_worker,
    )

    last_tool_result: dict[str, Any] | None = None
    last_tool_name: str = ""
    turn = 0
    failure_reason = ""
    auto_closed = False

    while turn < agent_config.worker_max_turns:
        turn += 1
        if turn == 1:
            user_msg = initial_user
        else:
            active_query = "(none)"
            if state.active_query is not None:
                aq = state.active_query
                active_query = f"{aq.query!r}"
                if aq.filters:
                    active_query += f" with filters={aq.filters}"
            valid_next = _compute_valid_next_tools(
                state, agent_config, last_tool_name, turn=turn,
            )
            hint_list = _compute_relevant_hints(
                last_tool_name, last_tool_result, state, agent_config,
            )
            hints_rendered = (
                "\n".join(f"- {h}" for h in hint_list)
                if hint_list else "(no situational hints this turn)"
            )
            user_msg = USER_TEMPLATE_CONTINUE.format(
                sub_topic_id=spec.id,
                tool_results=_render_tool_result(last_tool_name, last_tool_result),
                n_queries=len(state.queries),
                max_queries_per_worker=agent_config.max_queries_per_worker,
                active_fingerprint=(state.active_fingerprint or "(none)"),
                active_query=active_query,
                n_cumulative=len(state.cumulative_paper_ids),
                pending_misses=dispatcher.pending_miss_count(),
                turn=turn,
                max_turns=agent_config.worker_max_turns,
                valid_next_tools=valid_next,
                situational_hints=hints_rendered,
            )

        tokens_before = ctx.budget.llm_total_tokens
        try:
            resp = llm_client.call(
                SYSTEM,
                user_msg,
                category=f"expand_by_search_worker:{spec.id}",
                response_schema=RESPONSE_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
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
            tokens_in=None,
            tokens_out=tokens_spent,
        )

        try:
            decoded = _parse_tool_call(resp.text or "")
        except _ParseError as exc:
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

        if dispatcher.done_called:
            break
        if ctx.budget.is_exhausted(ctx.config):
            failure_reason = "budget_exhausted"
            break
        # Auto-close safety net on the penultimate turn: if the
        # worker has fetched papers but never closed, force a clean
        # done() so the papers aren't wasted. Transparency: sets the
        # auto_closed flag on SubTopicResult and emits an event.
        if turn == agent_config.worker_max_turns - 1:
            before_done = dispatcher.done_called
            _attempt_auto_close(
                dispatcher, state, spec=spec, logger=logger,
                worker_id=worker_id, turn=turn,
            )
            if dispatcher.done_called and not before_done:
                auto_closed = True
                break

    dispatcher.store.drop_all_for_worker(worker_id)

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

    query_results: list[QueryResult] = []
    for fp, q in state.queries.items():
        query_results.append(QueryResult(
            query=q.query,
            filters=dict(q.filters),
            fingerprint=fp,
            n_fetched=q.n_fetched or 0,
            total_in_corpus=q.total_in_corpus or 0,
            papers_added_to_cumulative=(q.n_fetched or 0),
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
        query_results=query_results,
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

    Accepts: bare JSON, ```json fenced```, or prose-before-brace.
    """
    if not text:
        raise _ParseError("empty response")
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
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

    Simplified from v2: there is no per-query checklist to abandon.
    Steps:

    1. If the worker has fetched zero papers, attempt one last fetch
       on the supervisor's ``initial_query_sketch``. If that also
       returns 0, give up (worker will finalise as ``failed``).
    2. Auto-consume every pending verification miss with a synthetic
       ``diagnose_miss(action_taken=accept_gap)`` so ``_pre_done``
       won't reject on undiagnosed misses.
    3. Call ``done`` via the dispatcher with
       ``coverage_assessment=limited`` and a summary that flags the
       auto-close.

    Every dispatch goes through the normal machinery so the
    transcript shows exactly what the rescuer did.
    """
    had_cumulative_before = bool(state.cumulative_paper_ids)
    had_verification_before = any(
        q.df_id is not None for q in state.queries.values()
    )

    if not state.cumulative_paper_ids:
        sketch = spec.initial_query_sketch
        if sketch:
            sz = dispatcher.dispatch("check_query_size", {"query": sketch})
            if isinstance(sz, dict) and "error" not in sz and (sz.get("total") or 0) > 0:
                dispatcher.dispatch("fetch_results", {"query": sketch})
        if not state.cumulative_paper_ids:
            return  # genuinely nothing to close

    # Emit the transparency event.
    try:
        logger.log_auto_close_invoked(
            worker_id=worker_id,
            spec_id=spec.id,
            turn=turn,
            had_cumulative=had_cumulative_before,
            had_verification=had_verification_before,
            angles_abandoned=0,
        )
    except Exception:  # noqa: BLE001
        pass

    # Consume any pending misses with synthetic diagnose_miss calls.
    pending_titles = list(state.pending_miss_titles)
    consumed = len(state.miss_diagnoses)
    for title in pending_titles[consumed:]:
        dispatcher.dispatch("diagnose_miss", {
            "target_title": title,
            "hypotheses": ["auto-closer: not reached within worker budget"],
            "action_taken": "accept_gap",
            "queries_used": [q.query for q in state.queries.values() if q.query],
        })

    summary = (
        f"auto-closed on turn {turn} of {dispatcher.config.worker_max_turns} "
        f"(reached turn budget); {len(state.cumulative_paper_ids)} papers "
        f"fetched across {len(state.queries)} queries"
    )
    dispatcher.dispatch("done", {
        "paper_ids": sorted(state.cumulative_paper_ids),
        "coverage_assessment": "limited",
        "summary": summary,
    })


def _compute_relevant_hints(
    last_tool_name: str,
    last_tool_result: dict[str, Any] | None,
    state: WorkerState,
    cfg: "AgentConfig",
) -> list[str]:
    """State-derived hints surfaced per turn.

    Post-refactor: drops the refinement-cap hint (no longer a
    concept) and re-phrases the cap-related hints in terms of
    queries, not angles.
    """
    hints: list[str] = []

    # 1. Size-band guidance — fires after a check_query_size return.
    if (
        last_tool_name == "check_query_size"
        and isinstance(last_tool_result, dict)
        and isinstance(last_tool_result.get("total"), int)
    ):
        total = last_tool_result["total"]
        cap = getattr(cfg, "fetch_total_cap", 50_000)
        if total == 0:
            hints.append(
                "total=0 ⇒ the query is over-constrained. Drop a '+' "
                "clause OR add '|' alternatives. STAY on this sub-topic "
                "— do not switch."
            )
        elif total < 10:
            hints.append(
                f"total={total} is very thin. Broaden by dropping a '+' "
                "clause or adding '|' alternatives before fetch_results."
            )
        elif total > cap:
            hints.append(
                f"total={total:,} exceeds the fetch cap ({cap:,}). Add "
                "a structural filter (year / fieldsOfStudy / venue) — "
                "NOT more '+' clauses."
            )
        elif total > 5_000:
            hints.append(
                f"total={total:,} is large. Narrow with a structural "
                "filter (year / fieldsOfStudy / venue) if you want a "
                "denser sample."
            )
        elif 50 <= total <= 5_000:
            hints.append(
                f"total={total:,} is in the sweet spot — proceed to "
                "fetch_results with the SAME (query, filters) pair."
            )

    # 2. Query-cap awareness.
    n_queries = len(state.queries)
    if n_queries >= cfg.max_queries_per_worker:
        hints.append(
            f"{n_queries}/{cfg.max_queries_per_worker} queries used (cap "
            "reached). New check_query_size calls on a distinct "
            "(query, filters) pair will be rejected — diagnose any "
            "pending misses and call done()."
        )
    elif n_queries == cfg.max_queries_per_worker - 1:
        hints.append(
            f"{n_queries}/{cfg.max_queries_per_worker} queries — one slot "
            "left. Make the next query count."
        )

    # 3. Pending-miss nudge.
    pending = len(state.pending_miss_titles) - len(state.miss_diagnoses)
    if pending > 0:
        hints.append(
            f"{pending} auto-detected reference miss(es) await diagnose_miss "
            "— each must be explained before done() is accepted."
        )

    return hints


def _compute_valid_next_tools(
    state: WorkerState,
    cfg: "AgentConfig",
    last_tool_name: str,
    *,
    turn: int,
) -> str:
    """Post-refactor candidate set.

    The fetch-inspect-verify chain collapsed into ``fetch_results``,
    so there's no per-angle checklist to drive. Logic:

    * pending misses exist → ``diagnose_miss`` only
    * no queries opened yet → ``check_query_size``
    * active query not yet fetched → ``fetch_results`` (or
      ``query_diagnostics`` to assess the size first)
    * fetched, no misses pending → ``done`` preferred (+
      ``check_query_size`` if cap + time permit)
    * time pressure (≥75%) narrows the set toward ``done``
    """
    n_queries = len(state.queries)
    active = state.active_query
    cap_hit = n_queries >= cfg.max_queries_per_worker
    time_pressure = turn >= int(cfg.worker_max_turns * 0.75)
    pending_misses = len(state.pending_miss_titles) - len(state.miss_diagnoses)

    candidates: list[str] = []
    if pending_misses > 0:
        candidates = ["diagnose_miss"]
    elif n_queries == 0:
        candidates = ["check_query_size"]
    elif active is not None and active.df_id is None:
        # Size-checked but not yet fetched.
        candidates = ["fetch_results"]
        if not time_pressure:
            candidates.append("query_diagnostics")
    else:
        has_any_fetch = any(q.df_id for q in state.queries.values())
        if has_any_fetch:
            if time_pressure or cap_hit:
                candidates = ["done"]
            else:
                candidates = ["done", "check_query_size"]
        else:
            candidates = ["check_query_size"]

    always = ["get_paper", "search_within_df", "query_diagnostics"]
    pressure_note = ""
    if time_pressure:
        pressure_note = (
            f"\n\nTIME PRESSURE: turn {turn} of {cfg.worker_max_turns} — "
            f"strongly prefer `done` over opening new queries."
        )
    if cap_hit:
        pressure_note += (
            f"\n\nQUERY CAP HIT: opened {cfg.max_queries_per_worker} distinct "
            f"queries. New (query, filters) pairs will be rejected — "
            f"diagnose misses and call `done`."
        )
    return (
        "next-action candidates (strongly recommended): "
        + ", ".join(f"`{t}`" for t in candidates)
        + "; optional deep-dive: "
        + ", ".join(f"`{t}`" for t in always)
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
    try:
        blob = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        blob = str(result)
    # fetch_results returns a large digest; allow more headroom.
    limit = 12000 if tool_name == "fetch_results" else 6000
    if len(blob) > limit:
        head = blob[: int(limit * 0.65)]
        tail = blob[-int(limit * 0.30):]
        blob = f"{head}\n... [TRUNCATED {len(blob) - len(head) - len(tail)} chars] ...\n{tail}"
    return (
        f"**Previous call**: `{tool_name}`\n"
        f"**Result**:\n```json\n{blob}\n```"
    )


__all__ = ["run_sub_topic_worker"]
