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
- Angle caps: ``len(worker_state.angles) ≤ max_angles_per_worker``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from citeclaw.agents.state import (
    AngleState,
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
    # Angle-state helpers used by pre- and post-hooks
    # ------------------------------------------------------------------

    def get_or_create_angle(self, query: str, filters: dict[str, Any] | None) -> AngleState:
        """Return the AngleState for ``(query, filters)``, creating if absent.

        Enforces ``max_angles_per_worker``: if creating this angle
        would exceed the cap, raises a ``DispatcherError`` which the
        caller's pre-hook converts to an error envelope.
        """
        fp = query_fingerprint(query, filters)
        existing = self.state.angles.get(fp)
        if existing is not None:
            return existing
        if len(self.state.angles) >= self.config.max_angles_per_worker:
            raise DispatcherError(
                error=(
                    f"angle cap reached: this worker has already opened "
                    f"{len(self.state.angles)} distinct (query, filters) tuples "
                    f"(cap: {self.config.max_angles_per_worker})"
                ),
                hint=(
                    "finish inspection on existing angles and call done() "
                    "with coverage_assessment set accordingly — or abort if "
                    "further queries are truly needed, the supervisor will "
                    "decide whether to re-dispatch"
                ),
            )
        angle = AngleState(fingerprint=fp, query=query, filters=dict(filters or {}))
        self.state.angles[fp] = angle
        return angle

    def set_active(self, fingerprint: str) -> None:
        self.state.active_fingerprint = fingerprint

    def recent_contains_miss(self) -> bool:
        """True iff the most recent ``contains`` call returned False.

        Walks the tail of ``call_log`` for the last ``contains``
        entry; if its result was ``False``, a subsequent
        ``diagnose_miss`` is valid. Any tool call OTHER than
        ``diagnose_miss`` between the contains and the diagnose
        does not clear the signal — the miss is only "resolved" by
        the diagnose itself (exactly-once per miss).
        """
        for entry in reversed(self.state.call_log):
            name = entry.get("tool")
            if name == "diagnose_miss":
                # A diagnose already consumed the most recent miss —
                # nothing more to diagnose unless there's been another
                # contains that returned False since then.
                return False
            if name == "contains":
                result = entry.get("result") or {}
                return result.get("contains") is False
        return False

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
        # calls also errored on the SAME active angle (or on the
        # "no-active-angle" state), synthetically invoke abandon_angle
        # (or a no-active-angle reset) so the worker escapes the loop.
        if is_error(result):
            self._maybe_trip_circuit()

        return result

    def _maybe_trip_circuit(self) -> None:
        """Trip the circuit if the last N calls were all error envelopes.

        On trip: if there's an active angle that hasn't already been
        force-abandoned, synthetically call ``abandon_angle`` (bypassing
        the tool dispatch machinery so this can't itself trigger the
        breaker). Mark the fingerprint so we don't force-abandon it
        again later in the same worker.
        """
        recent = [e for e in self.state.call_log[-self.CIRCUIT_BREAKER_THRESHOLD:]]
        if len(recent) < self.CIRCUIT_BREAKER_THRESHOLD:
            return
        if not all(
            isinstance(e.get("result"), dict) and "error" in e["result"]
            for e in recent
        ):
            return
        active = self.state.active_angle
        if active is None:
            return
        if active.fingerprint in self._circuit_tripped_on_angles:
            return
        self._circuit_tripped_on_angles.add(active.fingerprint)
        # Synthetic abandon — mirror the handler's effect directly so
        # this call doesn't itself go through dispatch (which would
        # re-check the circuit).
        abandoned_fp = active.fingerprint
        abandoned_df = active.df_id
        removed_ids = 0
        if abandoned_df and abandoned_df in self.store:
            try:
                df = self.store.get(abandoned_df)
                for pid in df["paper_id"]:
                    if isinstance(pid, str) and pid in self.state.cumulative_paper_ids:
                        self.state.cumulative_paper_ids.remove(pid)
                        removed_ids += 1
            except Exception:  # noqa: BLE001
                pass
            self.store.drop(abandoned_df)
        self.state.angles.pop(abandoned_fp, None)
        self.state.active_fingerprint = None
        self.state.call_log.append({
            "turn": self.state.turn_index + 1,
            "tool": "abandon_angle",
            "args": {},
            "synthetic": True,
            "reason": "circuit_breaker_tripped_after_3_errors",
            "result": {
                "abandoned_fingerprint": abandoned_fp,
                "n_papers_removed_from_cumulative": removed_ids,
                "note": (
                    "circuit breaker: 3 consecutive tool errors on this "
                    "angle — auto-abandoned to prevent loop"
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
    """Pre-hook shared by every tool that takes a ``df_id`` arg.

    Rejects if ``df_id`` is missing, not a string, or not registered
    in the DataFrameStore.
    """
    df_id = args.get("df_id")
    if not isinstance(df_id, str) or not df_id:
        return error_envelope(
            "missing or non-string df_id",
            "call fetch_results first; the returned df_id is what you pass here",
        )
    if df_id not in dispatcher.store:
        return error_envelope(
            f"df_id {df_id!r} not found in store",
            (
                f"registered df_ids: {sorted(dispatcher.store.list_ids(worker_id=dispatcher.worker_id))}; "
                f"call fetch_results first"
            ),
        )
    return None


def find_angle_by_df_id(dispatcher: WorkerDispatcher, df_id: str) -> AngleState | None:
    """Locate the AngleState whose ``df_id`` matches."""
    for angle in dispatcher.state.angles.values():
        if angle.df_id == df_id:
            return angle
    return None


__all__ = [
    "WorkerDispatcher",
    "SupervisorDispatcher",
    "ToolSpec",
    "DispatcherError",
    "error_envelope",
    "is_error",
    "require_df_id",
    "find_angle_by_df_id",
]
