"""Tool dispatcher + hook enforcement for the v2 ExpandBySearch agents.

The LLM never calls tool handlers directly. Every tool invocation
flows through :class:`WorkerDispatcher` (or
:class:`SupervisorDispatcher`), which:

1. Validates the tool name exists.
2. Runs tool-specific **preconditions** (hooks) — rejects the call
   with a structured ``{"error": ..., "hint": ...}`` envelope when
   a hook fires. The LLM sees the error on the next turn and can
   course-correct.
3. Invokes the handler.
4. Runs tool-specific **post-hooks** — flips the angle checklist
   flags (``checked_top_cited``, ``checked_random``, …), records
   verification misses, increments refinement counts, etc.
5. Appends a structured entry to ``state.call_log`` for postmortem.
6. Returns the handler's result (as a dict) or the error envelope.

The dispatcher does NOT serialize to/from JSON — that's the
structured-output adapter's job. It takes a Python dict in and
returns a Python dict out.

Hook enforcement follows the v2 design doc:

- ``fetch_results``: object consistency via
  :func:`~citeclaw.agents.state.query_fingerprint`. The recomputed
  fingerprint must match one the worker already registered with
  ``check_query_size``. No temporal window — prior calls to other
  tools between the size-check and fetch are allowed.
- ``check_query_size`` with a *new* fingerprint: angle transition.
  The current active angle's checklist must be complete before the
  worker opens a new angle.
- ``done`` (worker): every angle where ``fetch_results`` ran must
  have ``checked_top_cited ∧ checked_random ∧ checked_years`` True;
  angles with ``n_fetched ≥ 500`` must also have
  ``checked_topic_model`` True. ≥1 verification cycle must have
  completed; every miss must be followed by a ``diagnose_miss``.
- ``contains``: worker must have ≥1 ``fetch_results``.
- ``diagnose_miss``: a prior ``contains(...)`` in this worker must
  have returned False.
- Query caps: ``len(worker_state.queries) ≤ max_queries_per_worker``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from citeclaw.agents.state import (
    QueryMeta,
    SupervisorState,
    WorkerState,
    query_fingerprint,
)

if TYPE_CHECKING:
    from citeclaw.agents.dataframe_store import DataFrameStore
    from citeclaw.agents.state import AgentConfig
    from citeclaw.context import Context

log = logging.getLogger("citeclaw.agents.tool_dispatch")

# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


def error_envelope(error: str, hint: str) -> dict[str, Any]:
    """Build the agent-facing error shape ``{"error": ..., "hint": ...}``."""
    return {"error": error, "hint": hint}


def is_error(result: dict[str, Any]) -> bool:
    return isinstance(result, dict) and "error" in result and "hint" in result


# ---------------------------------------------------------------------------
# Tool registration record
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """One registered tool + its pre/post-hook callables.

    Handlers take ``(args: dict, dispatcher) -> dict``. Pre-hooks
    take ``(args: dict, dispatcher) -> dict | None``: return ``None``
    to accept the call, or an error envelope dict to reject it.
    Post-hooks take ``(args: dict, result: dict, dispatcher) -> None``
    and mutate dispatcher state.
    """

    name: str
    handler: Callable[[dict, "WorkerDispatcher | SupervisorDispatcher"], dict]
    pre_hook: Callable[[dict, "WorkerDispatcher | SupervisorDispatcher"], dict | None] | None = None
    post_hook: Callable[[dict, dict, "WorkerDispatcher | SupervisorDispatcher"], None] | None = None


# ---------------------------------------------------------------------------
# WorkerDispatcher — the main event
# ---------------------------------------------------------------------------


class WorkerDispatcher:
    """Dispatches tool calls for ONE sub-topic worker.

    Lives for the duration of a single ``run_sub_topic_worker`` call.
    Holds the worker's ``WorkerState``, its ``DataFrameStore``, and a
    tool registry. The worker loop calls ``dispatch(tool_name, args)``
    and receives either a tool result or an error envelope.

    The dispatcher is the ONLY place per-angle checklist state is
    mutated — handlers can read but not write the AngleState flags;
    post-hooks do that writing on behalf of the matching handler.
    """

    CIRCUIT_BREAKER_THRESHOLD = 3

    def __init__(
        self,
        *,
        worker_state: WorkerState,
        dataframe_store: "DataFrameStore",
        agent_config: "AgentConfig",
        ctx: "Context",
        worker_id: str,
    ) -> None:
        self.state = worker_state
        self.store = dataframe_store
        self.config = agent_config
        self.ctx = ctx
        self.worker_id = worker_id
        self._tools: dict[str, ToolSpec] = {}
        self._done_called = False
        self._done_result: dict[str, Any] | None = None
        self._circuit_tripped_on_angles: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    @property
    def registered_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    @property
    def done_called(self) -> bool:
        return self._done_called

    @property
    def done_result(self) -> dict[str, Any] | None:
        return self._done_result

    # ------------------------------------------------------------------
    # Query-state helpers
    # ------------------------------------------------------------------

    def get_or_create_query(self, query: str, filters: dict[str, Any] | None) -> QueryMeta:
        """Return the :class:`QueryMeta` for ``(query, filters)``, creating if absent.

        Enforces ``max_queries_per_worker``: if creating this query
        would exceed the cap, raises :class:`DispatcherError`.
        Re-running the same (query, filters) pair reuses the cached
        record — no slot consumed.
        """
        fp = query_fingerprint(query, filters)
        existing = self.state.queries.get(fp)
        if existing is not None:
            return existing
        if len(self.state.queries) >= self.config.max_queries_per_worker:
            raise DispatcherError(
                error=(
                    f"query cap reached: this worker has already opened "
                    f"{len(self.state.queries)} distinct (query, filters) tuples "
                    f"(cap: {self.config.max_queries_per_worker})"
                ),
                hint=(
                    "diagnose any outstanding verification misses and "
                    "call done() — further refinement isn't productive "
                    "past the cap"
                ),
            )
        record = QueryMeta(
            fingerprint=fp, query=query, filters=dict(filters or {}),
        )
        self.state.queries[fp] = record
        return record

    def set_active(self, fingerprint: str) -> None:
        self.state.active_fingerprint = fingerprint

    def pending_miss_count(self) -> int:
        """Count of auto-verifier misses not yet diagnosed.

        Title-based: a pending miss is only considered handled when a
        ``diagnose_miss`` call with a matching ``target_title`` has
        been recorded. The earlier implementation subtracted list
        lengths which let a fabricated-title diagnose_miss satisfy the
        counter while real pending misses remained undiagnosed
        (observed iter-10 spectral_theory_gft). The handler for
        diagnose_miss also rejects titles outside the pending set, so
        this identity-based count now matches the gate exactly.
        """
        diagnosed = {
            str(dx.get("target_title", "")).strip().lower()
            for dx in self.state.miss_diagnoses
            if isinstance(dx, dict)
        }
        remaining = sum(
            1 for t in self.state.pending_miss_titles
            if t.strip().lower() not in diagnosed
        )
        return remaining

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Validate, hook-check, invoke, post-hook, log. Return result."""
        self.state.turn_index += 1
        call_entry: dict[str, Any] = {
            "turn": self.state.turn_index,
            "tool": tool_name,
            "args": args,
            "ts": time.time(),
        }
        spec = self._tools.get(tool_name)
        if spec is None:
            result = error_envelope(
                f"unknown tool {tool_name!r}",
                (
                    f"valid tools: {', '.join(self.registered_tools)}; "
                    f"check the tool_name field in your output"
                ),
            )
            call_entry["result"] = result
            self.state.call_log.append(call_entry)
            return result

        if not isinstance(args, dict):
            result = error_envelope(
                "tool args must be a JSON object",
                f"got {type(args).__name__} — wrap the arguments in an object",
            )
            call_entry["result"] = result
            self.state.call_log.append(call_entry)
            return result

        # Pre-hook.
        if spec.pre_hook is not None:
            try:
                maybe_err = spec.pre_hook(args, self)
            except DispatcherError as de:
                maybe_err = error_envelope(de.error, de.hint)
            if maybe_err is not None:
                call_entry["result"] = maybe_err
                self.state.call_log.append(call_entry)
                return maybe_err

        # Handler.
        try:
            result = spec.handler(args, self)
        except DispatcherError as de:
            result = error_envelope(de.error, de.hint)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s crashed", tool_name)
            result = error_envelope(
                f"tool {tool_name!r} raised {type(exc).__name__}",
                str(exc)[:200] or "(no message)",
            )

        # Post-hook — only run on success (never on error envelopes).
        if spec.post_hook is not None and not is_error(result):
            try:
                spec.post_hook(args, result, self)
            except Exception as exc:  # noqa: BLE001
                log.warning("post-hook for %s failed: %s", tool_name, exc)

        call_entry["result"] = result
        self.state.call_log.append(call_entry)

        # Worker-level done sentinel — the worker loop reads this to break.
        if tool_name == "done" and not is_error(result):
            self._done_called = True
            self._done_result = result

        # Circuit breaker: if this call errored AND the previous 2
        # calls also errored on the SAME active query, drop that
        # query from the registry so the worker can open a fresh one.
        if is_error(result):
            self._maybe_trip_circuit()

        return result

    def _maybe_trip_circuit(self) -> None:
        """Trip the circuit if the last N calls were all error envelopes.

        On trip: drop the active query's QueryMeta + its DataFrame
        from the store, and remove its papers from the cumulative
        set. The worker's next turn sees ``active_fingerprint=None``
        and can open a different query. One-shot per fingerprint so
        the circuit doesn't fire twice on the same query.
        """
        recent = [e for e in self.state.call_log[-self.CIRCUIT_BREAKER_THRESHOLD:]]
        if len(recent) < self.CIRCUIT_BREAKER_THRESHOLD:
            return
        if not all(
            isinstance(e.get("result"), dict) and "error" in e["result"]
            for e in recent
        ):
            return
        active = self.state.active_query
        if active is None:
            return
        if active.fingerprint in self._circuit_tripped_on_angles:
            return
        self._circuit_tripped_on_angles.add(active.fingerprint)
        dropped_fp = active.fingerprint
        dropped_df = active.df_id
        removed_ids = 0
        if dropped_df and dropped_df in self.store:
            try:
                df = self.store.get(dropped_df)
                for pid in df["paper_id"]:
                    if isinstance(pid, str) and pid in self.state.cumulative_paper_ids:
                        self.state.cumulative_paper_ids.remove(pid)
                        removed_ids += 1
            except Exception:  # noqa: BLE001
                pass
            self.store.drop(dropped_df)
        self.state.queries.pop(dropped_fp, None)
        self.state.active_fingerprint = None
        self.state.call_log.append({
            "turn": self.state.turn_index + 1,
            "tool": "circuit_breaker_drop_query",
            "args": {},
            "synthetic": True,
            "reason": "circuit_breaker_tripped_after_3_errors",
            "result": {
                "dropped_fingerprint": dropped_fp,
                "n_papers_removed_from_cumulative": removed_ids,
                "note": (
                    "circuit breaker: 3 consecutive tool errors on this "
                    "query — auto-dropped so the worker can open a new one"
                ),
            },
            "ts": 0,
        })
        self.state.turn_index += 1


# ---------------------------------------------------------------------------
# SupervisorDispatcher — thinner, but same shape
# ---------------------------------------------------------------------------


class SupervisorDispatcher:
    """Dispatches tool calls for the supervisor agent.

    Far simpler than WorkerDispatcher: the supervisor has two tools
    (``dispatch_sub_topic_worker``, ``done``) and its only hook is a
    ``done`` precondition (≥1 worker must have been dispatched). No
    per-angle state, no DataFrame store.
    """

    def __init__(
        self,
        *,
        supervisor_state: SupervisorState,
        agent_config: "AgentConfig",
        ctx: "Context",
    ) -> None:
        self.state = supervisor_state
        self.config = agent_config
        self.ctx = ctx
        self._tools: dict[str, ToolSpec] = {}
        self._done_called = False
        self._done_result: dict[str, Any] | None = None

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    @property
    def registered_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    @property
    def done_called(self) -> bool:
        return self._done_called

    @property
    def done_result(self) -> dict[str, Any] | None:
        return self._done_result

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.state.turn_index += 1
        call_entry: dict[str, Any] = {
            "turn": self.state.turn_index,
            "tool": tool_name,
            "args": args,
            "ts": time.time(),
        }
        spec = self._tools.get(tool_name)
        if spec is None:
            result = error_envelope(
                f"unknown tool {tool_name!r}",
                f"valid tools: {', '.join(self.registered_tools)}",
            )
            call_entry["result"] = result
            self.state.call_log.append(call_entry)
            return result

        if not isinstance(args, dict):
            result = error_envelope(
                "tool args must be a JSON object",
                f"got {type(args).__name__}",
            )
            call_entry["result"] = result
            self.state.call_log.append(call_entry)
            return result

        if spec.pre_hook is not None:
            try:
                maybe_err = spec.pre_hook(args, self)
            except DispatcherError as de:
                maybe_err = error_envelope(de.error, de.hint)
            if maybe_err is not None:
                call_entry["result"] = maybe_err
                self.state.call_log.append(call_entry)
                return maybe_err

        try:
            result = spec.handler(args, self)
        except DispatcherError as de:
            result = error_envelope(de.error, de.hint)
        except Exception as exc:  # noqa: BLE001
            log.exception("supervisor tool %s crashed", tool_name)
            result = error_envelope(
                f"tool {tool_name!r} raised {type(exc).__name__}",
                str(exc)[:200] or "(no message)",
            )

        if spec.post_hook is not None and not is_error(result):
            try:
                spec.post_hook(args, result, self)
            except Exception as exc:  # noqa: BLE001
                log.warning("post-hook for %s failed: %s", tool_name, exc)

        call_entry["result"] = result
        self.state.call_log.append(call_entry)

        if tool_name == "done" and not is_error(result):
            self._done_called = True
            self._done_result = result

        return result


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DispatcherError(Exception):
    """Raised inside handlers/hooks to produce a structured error envelope.

    Carries both the short ``error`` summary and the actionable
    ``hint`` that goes back to the LLM. The dispatcher catches this
    and converts to ``error_envelope(error, hint)``.
    """

    def __init__(self, error: str, hint: str) -> None:
        super().__init__(error)
        self.error = error
        self.hint = hint


# ---------------------------------------------------------------------------
# Shared pre/post-hook helpers the tool module will wire up
# ---------------------------------------------------------------------------


def require_df_id(args: dict[str, Any], dispatcher: WorkerDispatcher) -> dict[str, Any] | None:
    """Pre-hook shared by tools that take a ``df_id`` arg.

    If ``df_id`` is missing, empty, or doesn't match a registered
    DataFrame, we fall back to the active query's ``df_id`` when it
    exists. Weaker LLMs mutate / hallucinate long opaque strings, so
    defaulting to the obvious most-recent-fetch case eliminates a
    whole class of needless errors. The agent can still target an
    older DataFrame explicitly when it really needs to.

    Only hard-rejects when there's no active query AND no registered
    DataFrame — the agent needs to call ``fetch_results`` first.
    """
    df_id = args.get("df_id")
    if not isinstance(df_id, str) or not df_id or df_id not in dispatcher.store:
        active = dispatcher.state.active_query
        if active is not None and active.df_id and active.df_id in dispatcher.store:
            args["df_id"] = active.df_id
            return None
        return error_envelope(
            "no DataFrame available — no query has been fetched yet",
            (
                "call fetch_results first. "
                f"registered df_ids: {sorted(dispatcher.store.list_ids(worker_id=dispatcher.worker_id))}"
            ),
        )
    return None


def find_query_by_df_id(dispatcher: WorkerDispatcher, df_id: str) -> QueryMeta | None:
    """Locate the QueryMeta whose ``df_id`` matches."""
    for q in dispatcher.state.queries.values():
        if q.df_id == df_id:
            return q
    return None


__all__ = [
    "WorkerDispatcher",
    "SupervisorDispatcher",
    "ToolSpec",
    "DispatcherError",
    "error_envelope",
    "is_error",
    "require_df_id",
    "find_query_by_df_id",
]
