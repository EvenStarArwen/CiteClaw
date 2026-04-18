"""Supervisor loop for v2 ExpandBySearch.

Runs the strategy-then-dispatch agent: emits a
:class:`~citeclaw.agents.state.SearchStrategy` via ``set_strategy``,
then dispatches workers by spec_id one at a time, finally closes via
``done``. All coordination with workers goes through this module —
external callers only see :func:`run_supervisor`.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from citeclaw.agents.dataframe_store import DataFrameStore
from citeclaw.agents.state import (
    AgentConfig,
    SearchStrategy,
    StructuralPriors,
    SubTopicSpec,
    SupervisorState,
)
from citeclaw.agents.tool_dispatch import (
    DispatcherError,
    SupervisorDispatcher,
    ToolSpec,
    error_envelope,
    is_error,
)
from citeclaw.agents.worker import run_sub_topic_worker
from citeclaw.prompts.search_agent_supervisor import (
    RESPONSE_SCHEMA,
    SYSTEM,
    USER_TEMPLATE_CONTINUE,
    USER_TEMPLATE_FIRST,
    render_seed_block,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.context import Context

log = logging.getLogger("citeclaw.agents.supervisor")


def run_supervisor(
    *,
    topic_description: str,
    filter_summary: str,
    seed_papers: list[dict[str, Any]],
    llm_client: "LLMClient",
    ctx: "Context",
    agent_config: AgentConfig,
    logger: "SearchLogger",
) -> tuple[SupervisorState, list[str]]:
    """Drive the supervisor loop end-to-end.

    Returns ``(supervisor_state, aggregate_paper_ids)`` where
    ``aggregate_paper_ids`` is the dedup'd union of every successful
    worker's ``paper_ids``.
    """
    state = SupervisorState()
    dispatcher = SupervisorDispatcher(
        supervisor_state=state,
        agent_config=agent_config,
        ctx=ctx,
    )
    shared_store = DataFrameStore()

    # Register supervisor tools. The handlers close over the shared
    # objects needed for worker dispatch.
    _register_supervisor_tools(
        dispatcher,
        shared_store=shared_store,
        topic_description=topic_description,
        filter_summary=filter_summary,
        seed_papers=seed_papers,
        llm_client=llm_client,
        ctx=ctx,
        agent_config=agent_config,
        logger=logger,
    )

    initial_user = USER_TEMPLATE_FIRST.format(
        topic_description=topic_description,
        filter_summary=filter_summary,
        seed_block=render_seed_block(seed_papers if agent_config.share_seeds_with_agents else []),
        supervisor_max_turns=agent_config.supervisor_max_turns,
        worker_max_turns=agent_config.worker_max_turns,
        max_queries_per_worker=agent_config.max_queries_per_worker,
    )

    last_tool_name = ""
    last_tool_result: dict[str, Any] | None = None
    turn = 0

    while turn < agent_config.supervisor_max_turns:
        turn += 1
        if turn == 1:
            user_msg = initial_user
        else:
            n_sub = len(state.strategy.sub_topics) if state.strategy else 0
            n_disp = len(state.sub_topic_results)
            n_succ = sum(1 for r in state.sub_topic_results if r.status == "success")
            n_fail = sum(1 for r in state.sub_topic_results if r.status == "failed")
            n_bud = sum(1 for r in state.sub_topic_results if r.status == "budget_exhausted")
            n_agg = len(state.aggregate_paper_ids())
            # Surface the full sub-topic table each turn — id + 1-line
            # description + per-worker status — so the supervisor can
            # (a) pick a valid id when dispatching without hallucinating
            # one, and (b) check coverage before calling add_sub_topics.
            # Showing only ids caused two failure modes on Gemma 4 / weak
            # models: hallucinated ids for dispatch (round-trip error)
            # and semantic-duplicate adds (two sub-topics covering the
            # same slice with different id slugs, e.g. cu_alloys_X and
            # cu_alloys_Y). Descriptions cost ~150 tokens / turn but
            # make both failure modes structurally impossible.
            dispatched_id_set = {r.spec_id for r in state.sub_topic_results}
            if state.strategy:
                spec_by_id = {s.id: s for s in state.strategy.sub_topics}
                result_by_id = {r.spec_id: r for r in state.sub_topic_results}
                dispatched_lines: list[str] = []
                for sid in sorted(dispatched_id_set):
                    spec = spec_by_id.get(sid)
                    res = result_by_id.get(sid)
                    desc = (spec.description or "").strip().replace("\n", " ") if spec else ""
                    if len(desc) > 100:
                        desc = desc[:100].rstrip() + "…"
                    if res is not None:
                        tag = f"{res.status}, {len(res.paper_ids)} papers"
                    else:
                        tag = "pending"
                    dispatched_lines.append(f"  - {sid} [{tag}]: {desc}" if desc else f"  - {sid} [{tag}]")
                remaining_lines: list[str] = []
                for s in state.strategy.sub_topics:
                    if s.id in dispatched_id_set:
                        continue
                    desc = (s.description or "").strip().replace("\n", " ")
                    if len(desc) > 100:
                        desc = desc[:100].rstrip() + "…"
                    remaining_lines.append(f"  - {s.id}: {desc}" if desc else f"  - {s.id}")
                dispatched_block = "\n".join(dispatched_lines) if dispatched_lines else "  (none)"
                remaining_block = "\n".join(remaining_lines) if remaining_lines else "  (none)"
            else:
                dispatched_block = "  (none)"
                remaining_block = "  (none)"
            user_msg = USER_TEMPLATE_CONTINUE.format(
                tool_results=_render_tool_result(last_tool_name, last_tool_result),
                n_sub_topics=n_sub,
                n_dispatched=n_disp,
                n_success=n_succ,
                n_failed=n_fail,
                n_budget=n_bud,
                n_aggregate=n_agg,
                turn=turn,
                supervisor_max_turns=agent_config.supervisor_max_turns,
                dispatched_block=dispatched_block,
                remaining_block=remaining_block,
            )

        tokens_before = ctx.budget.llm_total_tokens
        try:
            resp = llm_client.call(
                SYSTEM,
                user_msg,
                category="expand_by_search_supervisor",
                response_schema=RESPONSE_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("supervisor LLM call failed: %s", exc)
            break
        tokens_spent = ctx.budget.llm_total_tokens - tokens_before
        logger.log_supervisor_turn(
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
                "error": "response not valid JSON",
                "hint": str(exc),
            }
            continue

        tool_name = decoded.get("tool_name", "")
        tool_args = decoded.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        result = dispatcher.dispatch(tool_name, tool_args)
        logger.log_tool_call(
            scope="supervisor",
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
            log.info("supervisor: budget exhausted, stopping")
            break

    aggregate = state.aggregate_paper_ids()
    return state, aggregate


# ---------------------------------------------------------------------------
# Supervisor tool handlers
# ---------------------------------------------------------------------------


def _register_supervisor_tools(
    dispatcher: SupervisorDispatcher,
    *,
    shared_store: DataFrameStore,
    topic_description: str,
    filter_summary: str,
    seed_papers: list[dict[str, Any]],
    llm_client: "LLMClient",
    ctx: "Context",
    agent_config: AgentConfig,
    logger: "SearchLogger",
) -> None:
    def _handle_set_strategy(args: dict[str, Any], d: SupervisorDispatcher) -> dict[str, Any]:
        if d.state.strategy is not None:
            raise DispatcherError(
                "set_strategy already called",
                "the strategy is locked once set; dispatch workers or done()",
            )
        priors_raw = args.get("structural_priors") or {}
        sub_topics_raw = args.get("sub_topics")
        if not isinstance(priors_raw, dict):
            raise DispatcherError("'structural_priors' must be an object", "see schema")
        if not isinstance(sub_topics_raw, list) or not sub_topics_raw:
            raise DispatcherError(
                "'sub_topics' must be a non-empty list of {id, description, initial_query_sketch, reference_papers}",
                "aim for 3-15 sub-topics per the decomposition rubric",
            )
        if len(sub_topics_raw) > 20:
            raise DispatcherError(
                f"too many sub-topics ({len(sub_topics_raw)})",
                "max 20; consolidate similar slices",
            )

        priors = StructuralPriors(
            year_min=_opt_int(priors_raw.get("year_min")),
            year_max=_opt_int(priors_raw.get("year_max")),
            fields_of_study=_coerce_str_list(priors_raw.get("fields_of_study")),
            venue_filters=_coerce_str_list(priors_raw.get("venue_filters")),
        )
        specs: list[SubTopicSpec] = []
        seen_ids: set[str] = set()
        for idx, raw in enumerate(sub_topics_raw):
            if not isinstance(raw, dict):
                raise DispatcherError(
                    f"sub_topic[{idx}] must be an object",
                    "{id, description, initial_query_sketch, reference_papers}",
                )
            sid = str(raw.get("id") or "").strip()
            if not sid or sid in seen_ids:
                raise DispatcherError(
                    f"sub_topic[{idx}].id missing or duplicate (got {sid!r})",
                    "each sub_topic needs a unique non-empty id slug",
                )
            seen_ids.add(sid)
            specs.append(SubTopicSpec(
                id=sid,
                description=str(raw.get("description") or ""),
                initial_query_sketch=str(raw.get("initial_query_sketch") or ""),
                reference_papers=tuple(str(r) for r in (raw.get("reference_papers") or []) if r),
            ))

        d.state.strategy = SearchStrategy(
            structural_priors=priors,
            sub_topics=tuple(specs),
        )
        return {
            "acknowledged": True,
            "n_sub_topics": len(specs),
            "sub_topic_ids": [s.id for s in specs],
        }

    def _handle_add_sub_topics(args: dict[str, Any], d: SupervisorDispatcher) -> dict[str, Any]:
        """Append new sub-topic specs to an already-locked strategy.

        Addresses the one-shot-decomposition gap: if a worker's result
        reveals a gap (e.g. the sub-topic was too broad and needs to
        be split into narrower slices, or a slice the supervisor
        missed initially), the supervisor can add targeted specs
        without losing already-dispatched results. Purely additive —
        existing specs and worker records are untouched.

        The global 20-spec ceiling from ``set_strategy`` is enforced
        across the combined set so the supervisor can't pad the
        strategy into an unbounded list.
        """
        if d.state.strategy is None:
            raise DispatcherError(
                "no strategy yet — call set_strategy first",
                "add_sub_topics only augments an existing strategy",
            )
        raw = args.get("sub_topics")
        if not isinstance(raw, list) or not raw:
            raise DispatcherError(
                "'sub_topics' must be a non-empty list of {id, description, initial_query_sketch, reference_papers}",
                "each entry mirrors the set_strategy sub_topics schema",
            )
        existing_ids = {s.id for s in d.state.strategy.sub_topics}
        new_specs: list[SubTopicSpec] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise DispatcherError(
                    f"sub_topics[{idx}] must be an object",
                    "{id, description, initial_query_sketch, reference_papers}",
                )
            sid = str(item.get("id") or "").strip()
            if not sid:
                raise DispatcherError(
                    f"sub_topics[{idx}].id missing",
                    "each sub_topic needs a unique non-empty id slug",
                )
            if sid in existing_ids:
                raise DispatcherError(
                    f"sub_topic id {sid!r} already in strategy",
                    "pick a new id; can't overwrite existing sub_topics",
                )
            existing_ids.add(sid)
            new_specs.append(SubTopicSpec(
                id=sid,
                description=str(item.get("description") or ""),
                initial_query_sketch=str(item.get("initial_query_sketch") or ""),
                reference_papers=tuple(str(r) for r in (item.get("reference_papers") or []) if r),
            ))
        combined_n = len(d.state.strategy.sub_topics) + len(new_specs)
        if combined_n > 20:
            raise DispatcherError(
                f"combined strategy would have {combined_n} sub-topics",
                "max 20 across the run; consolidate before adding more",
            )
        d.state.strategy = SearchStrategy(
            structural_priors=d.state.strategy.structural_priors,
            sub_topics=tuple(list(d.state.strategy.sub_topics) + new_specs),
        )
        return {
            "acknowledged": True,
            "added": [s.id for s in new_specs],
            "n_sub_topics_total": len(d.state.strategy.sub_topics),
        }

    def _handle_dispatch(args: dict[str, Any], d: SupervisorDispatcher) -> dict[str, Any]:
        spec_id = args.get("spec_id")
        if not isinstance(spec_id, str) or not spec_id:
            raise DispatcherError("missing 'spec_id'", "pass one of the ids from set_strategy")
        strategy = d.state.strategy
        if strategy is None:
            raise DispatcherError(
                "call set_strategy first",
                "strategy must be locked before any dispatch",
            )
        spec = next((s for s in strategy.sub_topics if s.id == spec_id), None)
        if spec is None:
            raise DispatcherError(
                f"spec_id {spec_id!r} not in strategy",
                f"valid ids: {[s.id for s in strategy.sub_topics]}",
            )
        # Enforce retry cap.
        attempts = d.state.worker_failures.get(spec_id, 0)
        if attempts > d.config.supervisor_retries_per_failed_worker:
            raise DispatcherError(
                f"spec_id {spec_id!r} has failed {attempts} times",
                "beyond the retry cap; move on or call done()",
            )
        worker_id = f"{spec.id}_{uuid.uuid4().hex[:6]}"
        result = run_sub_topic_worker(
            worker_id=worker_id,
            spec=spec,
            priors=strategy.structural_priors,
            topic_description=topic_description,
            filter_summary=filter_summary,
            seed_papers=seed_papers,
            llm_client=llm_client,
            ctx=ctx,
            dataframe_store=shared_store,
            agent_config=agent_config,
            logger=logger,
        )
        d.state.record_result(result)
        if result.status != "success":
            d.state.worker_failures[spec_id] = attempts + 1
        # Enriched payload: supervisor needs per-query detail so a
        # retry decision can target the actual failure mode (spec too
        # broad? query sketch too narrow? prior too tight? S2 coverage
        # gap?). Sampled at the dispatch boundary so the supervisor's
        # context doesn't bloat with full transcripts.
        query_payload = []
        for q in result.query_results:
            query_payload.append({
                "query": q.query,
                "filters": q.filters,
                "n_fetched": q.n_fetched,
                "total_in_corpus": q.total_in_corpus,
                "papers_added_to_cumulative": q.papers_added_to_cumulative,
            })
        stop_reason = _stop_reason_label(result)
        return {
            "spec_id": result.spec_id,
            "worker_id": worker_id,
            "status": result.status,
            "n_paper_ids": len(result.paper_ids),
            "coverage_assessment": result.coverage_assessment,
            "summary": result.summary,
            "turns_used": result.turns_used,
            "failure_reason": result.failure_reason,
            "stop_reason": stop_reason,
            "auto_closed": result.auto_closed,
            "n_queries": len(result.query_results),
            "queries": query_payload,
        }

    def _handle_done(args: dict[str, Any], d: SupervisorDispatcher) -> dict[str, Any]:
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise DispatcherError("missing 'summary'", "one-paragraph run summary")
        return {
            "acknowledged": True,
            "summary": summary,
            "n_sub_topic_results": len(d.state.sub_topic_results),
            "n_aggregate_paper_ids": len(d.state.aggregate_paper_ids()),
        }

    def _pre_done(args: dict[str, Any], d: SupervisorDispatcher) -> dict[str, Any] | None:
        if not d.state.sub_topic_results:
            return error_envelope(
                "no workers dispatched",
                "dispatch at least one sub-topic before done()",
            )
        return None

    dispatcher.register_many([
        ToolSpec("set_strategy", _handle_set_strategy),
        ToolSpec("add_sub_topics", _handle_add_sub_topics),
        ToolSpec("dispatch_sub_topic_worker", _handle_dispatch),
        ToolSpec("done", _handle_done, pre_hook=_pre_done),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _parse_tool_call(text: str) -> dict[str, Any]:
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
        raise _ParseError("missing 'tool_name'")
    return data


def _render_tool_result(tool_name: str, result: dict[str, Any] | None) -> str:
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
    if len(blob) > 4000:
        blob = blob[:2000] + f"\n... [TRUNCATED {len(blob) - 2000} chars]"
    return (
        f"**Previous call**: `{tool_name}`\n"
        f"**Result**:\n```json\n{blob}\n```"
    )


def _opt_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _coerce_str_list(v: Any) -> tuple[str, ...]:
    """Normalise ``fields_of_study`` / ``venue_filters`` to a tuple of strings.

    Weaker supervisor models frequently emit these as bare strings
    (``"Computer Science"`` or ``"Biology,Medicine"``) even when the
    schema expects a list, because the SYSTEM prompt describes them
    as "EXACT S2 names (comma-joined)". ``tuple()`` on a bare string
    then splits the string character by character and the worker sees
    nonsense like ``fieldsOfStudy: ['C', 'o', 'm', 'p', 'u', ...]`` in
    its user message (observed in every reference transcript on disk
    before this coercion was added). Accept str / list / tuple / None.
    """
    if v is None:
        return ()
    if isinstance(v, str):
        return tuple(p.strip() for p in v.split(",") if p.strip())
    if isinstance(v, (list, tuple)):
        return tuple(str(p).strip() for p in v if str(p).strip())
    return ()


def _stop_reason_label(result) -> str:
    """Categorise why the worker stopped, for supervisor retry decisions.

    Labels:
      - ``called_done``   — worker cleanly completed and called done()
      - ``max_turns``     — hit the worker_max_turns budget
      - ``budget``        — shared run budget exhausted
      - ``llm_error``     — LLM call failed (timeout / 5xx / parse)
      - ``other_failure`` — any other failure_reason
    """
    if result.status == "success":
        return "called_done"
    if result.status == "budget_exhausted":
        return "budget"
    fr = (result.failure_reason or "").lower()
    if "max_turns" in fr:
        return "max_turns"
    if "llm" in fr:
        return "llm_error"
    return "other_failure"


__all__ = ["run_supervisor"]
