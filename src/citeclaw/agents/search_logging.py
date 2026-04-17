"""Comprehensive structured logging for v2 ExpandBySearch runs.

The user's explicit ask: every LLM input/output and every tool
invocation must be durably logged so a human can inspect the
agent's behaviour offline. This module implements that as:

- **events.jsonl** — one line per event, machine-readable, written
  atomically on each event. Safe to tail in real time.
- **transcript.md** — rendered post-run by :meth:`finalize`; a
  human-readable, chronological story of the run with section
  headers for each sub-topic worker and each LLM turn.

One :class:`SearchLogger` instance per ``ExpandBySearch.run()`` call.
Both the supervisor and every worker call through the same logger
(passed into their dispatchers at construction), so the transcript
is a single unified timeline.

Event types (the ``type`` key on each JSONL line):

* ``run_started`` / ``run_finished``
* ``supervisor_turn`` — one LLM call on the supervisor's loop, with
  system prompt (only on turn 1 to avoid duplication),
  user message, response text, reasoning trace, token counts.
* ``worker_started`` / ``worker_finished`` — dispatch + result.
* ``worker_turn`` — one LLM call on a worker's loop.
* ``tool_call`` — every dispatch through the worker/supervisor
  dispatchers. Logged from the dispatcher via
  :meth:`SearchLogger.log_tool_call`.
* ``angle_transition`` — logged when the active angle changes.

The logger degrades gracefully: if the run_dir can't be created or
events.jsonl can't be opened, every method becomes a no-op rather
than crashing the pipeline. Runs can also be constructed with
``run_dir=None`` (see :class:`NullSearchLogger`) for tests that don't
want to touch the filesystem.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("citeclaw.agents.search_logging")


class SearchLogger:
    """Writes structured events to ``<run_dir>/events.jsonl`` and
    renders ``transcript.md`` on :meth:`finalize`.

    One instance per ``ExpandBySearch.run()`` call. Thread-safe-ish
    (CPython appends on a single file handle are atomic for single
    lines), but the v2 design is sequential so contention is not a
    concern.
    """

    def __init__(self, run_dir: Path | None) -> None:
        self._run_dir = run_dir
        self._events: list[dict[str, Any]] = []  # in-memory buffer for transcript
        self._jsonl_path: Path | None = None
        self._md_path: Path | None = None
        self._disabled = run_dir is None
        if run_dir is not None:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                self._jsonl_path = run_dir / "events.jsonl"
                self._md_path = run_dir / "transcript.md"
                self._jsonl_path.touch()
            except Exception as exc:  # noqa: BLE001
                log.warning("SearchLogger failed to open run_dir %s: %s",
                            run_dir, exc)
                self._disabled = True

    # ------------------------------------------------------------------
    # Core event writer
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **fields: Any) -> None:
        event = {"ts": time.time(), "type": event_type, **fields}
        self._events.append(event)
        if self._disabled or self._jsonl_path is None:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=_json_default) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.debug("SearchLogger append failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def log_run_started(
        self,
        *,
        topic: str,
        seed_count: int,
        filter_summary: str,
        agent_config: dict[str, Any],
        model: str | None,
    ) -> None:
        self._emit(
            "run_started",
            topic=topic,
            seed_count=seed_count,
            filter_summary=filter_summary,
            agent_config=agent_config,
            model=model,
        )

    def log_run_finished(
        self,
        *,
        n_papers_found: int,
        n_sub_topics: int,
        duration_s: float,
        llm_tokens: int,
        s2_requests: int,
        summary: str,
    ) -> None:
        self._emit(
            "run_finished",
            n_papers_found=n_papers_found,
            n_sub_topics=n_sub_topics,
            duration_s=duration_s,
            llm_tokens=llm_tokens,
            s2_requests=s2_requests,
            summary=summary,
        )

    def log_supervisor_turn(
        self,
        *,
        turn: int,
        system: str,
        user: str,
        response_text: str,
        reasoning: str,
        tokens_in: int | None,
        tokens_out: int | None,
    ) -> None:
        self._emit(
            "supervisor_turn",
            turn=turn,
            system=system if turn == 1 else "(unchanged)",
            user=user,
            response_text=response_text,
            reasoning=reasoning,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def log_worker_started(
        self,
        *,
        worker_id: str,
        spec_id: str,
        description: str,
        initial_query_sketch: str,
        reference_papers: list[str],
    ) -> None:
        self._emit(
            "worker_started",
            worker_id=worker_id,
            spec_id=spec_id,
            description=description,
            initial_query_sketch=initial_query_sketch,
            reference_papers=reference_papers,
        )

    def log_worker_finished(
        self,
        *,
        worker_id: str,
        spec_id: str,
        status: str,
        n_paper_ids: int,
        coverage_assessment: str | None,
        summary: str,
        turns_used: int,
        failure_reason: str = "",
    ) -> None:
        self._emit(
            "worker_finished",
            worker_id=worker_id,
            spec_id=spec_id,
            status=status,
            n_paper_ids=n_paper_ids,
            coverage_assessment=coverage_assessment,
            summary=summary,
            turns_used=turns_used,
            failure_reason=failure_reason,
        )

    def log_worker_turn(
        self,
        *,
        worker_id: str,
        turn: int,
        system: str,
        user: str,
        response_text: str,
        reasoning: str,
        tokens_in: int | None,
        tokens_out: int | None,
    ) -> None:
        self._emit(
            "worker_turn",
            worker_id=worker_id,
            turn=turn,
            system=system if turn == 1 else "(unchanged)",
            user=user,
            response_text=response_text,
            reasoning=reasoning,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def log_tool_call(
        self,
        *,
        scope: str,  # "worker:<id>" or "supervisor"
        turn: int,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self._emit(
            "tool_call",
            scope=scope,
            turn=turn,
            tool_name=tool_name,
            args=_redact_large(args),
            result=_redact_large(result),
            is_error="error" in (result or {}),
        )

    def log_auto_close_invoked(
        self,
        *,
        worker_id: str,
        spec_id: str,
        turn: int,
        had_cumulative: bool,
        had_verification: bool,
        angles_abandoned: int,
    ) -> None:
        """Emit one event each time the penultimate-turn rescuer fires.

        ``had_cumulative``/``had_verification`` capture the worker's
        state *before* the rescue so downstream analysis can separate
        "worker fetched papers but forgot to close" from "worker
        exhausted turns on queries returning 0".
        """
        self._emit(
            "auto_close_invoked",
            worker_id=worker_id,
            spec_id=spec_id,
            turn=turn,
            had_cumulative=bool(had_cumulative),
            had_verification=bool(had_verification),
            angles_abandoned=int(angles_abandoned),
        )

    def log_angle_transition(
        self,
        *,
        worker_id: str,
        from_fingerprint: str | None,
        to_fingerprint: str,
        query: str,
    ) -> None:
        self._emit(
            "angle_transition",
            worker_id=worker_id,
            from_fingerprint=from_fingerprint,
            to_fingerprint=to_fingerprint,
            query=query,
        )

    # ------------------------------------------------------------------
    # Finalize: render transcript.md
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Render the in-memory event buffer as a markdown transcript.

        Idempotent; safe to call multiple times. No-op when disabled.
        """
        if self._disabled or self._md_path is None:
            return
        try:
            md = _render_transcript(self._events)
            self._md_path.write_text(md, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("SearchLogger.finalize failed: %s", exc)

    def write_cost_summary(
        self,
        budget: Any,
        *,
        before_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Dump a ``cost_summary.json`` to the run directory.

        Writes the current :class:`BudgetTracker` state as returned by
        :meth:`to_dict`, plus (when provided) a ``search_only`` diff
        computed against ``before_snapshot`` — the subset of the
        run-total that was spent *inside this ExpandBySearch call*,
        which is what the user typically cares about when comparing
        models.

        Quiet no-op when the logger is disabled.
        """
        if self._disabled or self._run_dir is None:
            return
        try:
            full_state = budget.to_dict() if hasattr(budget, "to_dict") else {}
        except Exception as exc:  # noqa: BLE001
            log.debug("write_cost_summary: to_dict failed: %s", exc)
            full_state = {}
        diff: dict[str, Any] | None = None
        if before_snapshot and isinstance(before_snapshot.get("llm"), dict):
            diff = _diff_budget_snapshots(before_snapshot, full_state)
        payload = {
            "run_total": full_state,
            "search_only": diff,
            "pricing_provenance": (
                "Local USD estimate = response.usage.tokens × "
                "citeclaw.config.MODEL_PRICING. No public provider API "
                "(xAI, OpenAI, Gemini, Anthropic) returns per-call USD — "
                "token counts themselves are authoritative (reported by "
                "every provider in response.usage); only the USD "
                "conversion uses the local price table."
            ),
        }
        try:
            out_path = self._run_dir / "cost_summary.json"
            out_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.debug("write_cost_summary failed: %s", exc)

    # ------------------------------------------------------------------
    # In-memory access (for tests + postmortem tooling)
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


class NullSearchLogger(SearchLogger):
    """No-op logger for callers that don't want filesystem side-effects."""

    def __init__(self) -> None:
        super().__init__(run_dir=None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_transcript(events: list[dict[str, Any]]) -> str:
    """Render the event stream as a human-readable markdown transcript.

    Section structure:

    1. Header — run-level metadata.
    2. Supervisor section — every supervisor turn + tool call, in order.
    3. Per-worker subsection — each worker's turns + tool calls.
    4. Footer — final result.
    """
    lines: list[str] = []
    lines.append("# ExpandBySearch v2 — Run Transcript")
    lines.append("")
    # --- Header ---
    started = _find_event(events, "run_started")
    finished = _find_event(events, "run_finished")
    if started:
        lines.append("## Run metadata")
        lines.append("")
        lines.append(f"- **Topic**: {_inline(started.get('topic', ''))}")
        lines.append(f"- **Model**: `{started.get('model', '(unknown)')}`")
        lines.append(f"- **Seed papers**: {started.get('seed_count', 0)}")
        cfg = started.get("agent_config") or {}
        if cfg:
            lines.append(f"- **Agent config**: {cfg}")
        lines.append("")
        lines.append("### Downstream filter calibration")
        lines.append("")
        lines.append("```")
        lines.append(started.get("filter_summary", "(none)"))
        lines.append("```")
        lines.append("")
    if finished:
        lines.append("## Run summary")
        lines.append("")
        lines.append(f"- **Papers found**: {finished.get('n_papers_found', 0)}")
        lines.append(f"- **Sub-topics dispatched**: {finished.get('n_sub_topics', 0)}")
        lines.append(f"- **Duration**: {finished.get('duration_s', 0):.1f}s")
        lines.append(f"- **LLM tokens**: {finished.get('llm_tokens', 0):,}")
        lines.append(f"- **S2 requests**: {finished.get('s2_requests', 0):,}")
        lines.append("")
        lines.append(f"**Summary**: {_inline(finished.get('summary', ''))}")
        lines.append("")
    # --- Supervisor section ---
    lines.append("## Supervisor")
    lines.append("")
    for ev in events:
        t = ev.get("type")
        if t == "supervisor_turn":
            lines.append(f"### Supervisor turn {ev.get('turn')}")
            lines.append("")
            if ev.get("system") and ev.get("system") != "(unchanged)":
                lines.append("**System prompt**:")
                lines.append("")
                lines.append("```")
                lines.append(ev["system"])
                lines.append("```")
                lines.append("")
            lines.append("**User message**:")
            lines.append("")
            lines.append("```")
            lines.append(ev.get("user", ""))
            lines.append("```")
            lines.append("")
            if ev.get("reasoning"):
                lines.append("**Reasoning trace**:")
                lines.append("")
                lines.append("```")
                lines.append(ev["reasoning"])
                lines.append("```")
                lines.append("")
            lines.append("**Response**:")
            lines.append("")
            lines.append("```")
            lines.append(ev.get("response_text", ""))
            lines.append("```")
            lines.append("")
            lines.append(
                f"*tokens: in={ev.get('tokens_in')} out={ev.get('tokens_out')}*"
            )
            lines.append("")
        elif t == "tool_call" and ev.get("scope") == "supervisor":
            lines.append(
                f"- **tool_call**: `{ev.get('tool_name')}` "
                f"{'❌ ERROR ' if ev.get('is_error') else ''}"
                f"args={_inline_json(ev.get('args'))} → "
                f"result={_inline_json(ev.get('result'))}"
            )
            lines.append("")
    # --- Worker sections ---
    workers_seen: list[str] = []
    for ev in events:
        if ev.get("type") == "worker_started":
            wid = ev.get("worker_id")
            if wid and wid not in workers_seen:
                workers_seen.append(wid)
    for wid in workers_seen:
        lines.extend(_render_worker_section(events, worker_id=wid))
    return "\n".join(lines)


def _render_worker_section(events: list[dict[str, Any]], *, worker_id: str) -> list[str]:
    out: list[str] = []
    started = None
    finished = None
    for ev in events:
        if ev.get("type") == "worker_started" and ev.get("worker_id") == worker_id:
            started = ev
        elif ev.get("type") == "worker_finished" and ev.get("worker_id") == worker_id:
            finished = ev
    out.append(f"## Worker `{worker_id}`")
    out.append("")
    if started:
        out.append(f"- **Sub-topic id**: {started.get('spec_id')}")
        out.append(f"- **Description**: {_inline(started.get('description', ''))}")
        out.append(f"- **Initial query sketch**: `{started.get('initial_query_sketch', '')}`")
        ref = started.get("reference_papers") or []
        if ref:
            out.append(f"- **Reference papers**: {ref}")
        out.append("")
    if finished:
        out.append(f"- **Final status**: `{finished.get('status')}`")
        out.append(f"- **Coverage**: `{finished.get('coverage_assessment')}`")
        out.append(f"- **Paper IDs**: {finished.get('n_paper_ids', 0)}")
        out.append(f"- **Turns used**: {finished.get('turns_used')}")
        if finished.get("failure_reason"):
            out.append(f"- **Failure reason**: {finished['failure_reason']}")
        out.append(f"- **Summary**: {_inline(finished.get('summary', ''))}")
        out.append("")
    for ev in events:
        t = ev.get("type")
        if t == "worker_turn" and ev.get("worker_id") == worker_id:
            out.append(f"### Worker `{worker_id}` turn {ev.get('turn')}")
            out.append("")
            if ev.get("system") and ev.get("system") != "(unchanged)":
                out.append("**System prompt** (shown once per worker):")
                out.append("")
                out.append("```")
                out.append(ev["system"])
                out.append("```")
                out.append("")
            out.append("**User message**:")
            out.append("")
            out.append("```")
            out.append(ev.get("user", ""))
            out.append("```")
            out.append("")
            if ev.get("reasoning"):
                out.append("**Reasoning trace**:")
                out.append("")
                out.append("```")
                out.append(ev["reasoning"])
                out.append("```")
                out.append("")
            out.append("**Response**:")
            out.append("")
            out.append("```")
            out.append(ev.get("response_text", ""))
            out.append("```")
            out.append("")
        elif t == "tool_call" and ev.get("scope") == f"worker:{worker_id}":
            out.append(
                f"- **tool_call**: `{ev.get('tool_name')}` "
                f"{'❌ ERROR ' if ev.get('is_error') else ''}"
                f"args={_inline_json(ev.get('args'))} → "
                f"result={_inline_json(ev.get('result'))}"
            )
            out.append("")
        elif t == "angle_transition" and ev.get("worker_id") == worker_id:
            out.append(
                f"- *angle transition*: "
                f"{ev.get('from_fingerprint') or '(none)'} → {ev.get('to_fingerprint')} "
                f"for query `{ev.get('query')}`"
            )
            out.append("")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for ev in events:
        if ev.get("type") == event_type:
            return ev
    return None


def _inline(text: str, max_len: int = 400) -> str:
    if not isinstance(text, str):
        return str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


def _inline_json(obj: Any, max_len: int = 400) -> str:
    try:
        s = json.dumps(obj, default=_json_default, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = repr(obj)
    return _inline(s, max_len=max_len)


def _redact_large(obj: Any, *, max_chars: int = 20_000) -> Any:
    """Truncate huge string values so events.jsonl lines stay parseable.

    Large topic_model outputs / document lists can run into 10s of KB;
    we keep the shape but replace the overflowing value with a
    ``"<truncated N chars>"`` placeholder.
    """
    try:
        blob = json.dumps(obj, default=_json_default, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return obj
    if len(blob) <= max_chars:
        return obj
    return {"__truncated__": f"{len(blob)} chars"}


def _json_default(o: Any) -> Any:
    # Handle pandas / numpy in case something leaks through.
    try:
        import numpy as np  # type: ignore[import-not-found]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:  # noqa: BLE001
            return str(o)
    return str(o)


def _diff_budget_snapshots(
    before: dict[str, Any], after: dict[str, Any],
) -> dict[str, Any]:
    """Compute the ``after - before`` delta of a ``BudgetTracker.to_dict()``.

    Only differences the numeric totals + per-model buckets; category
    maps are kept as ``after`` (not diffed) because the search agent
    writes its own categories and the caller usually wants the full
    per-category view for this run.
    """
    b_llm = before.get("llm") or {}
    a_llm = after.get("llm") or {}

    def _num(d: dict[str, Any], k: str) -> float:
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else 0.0

    # Per-model diff.
    b_by_model = b_llm.get("by_model") or {}
    a_by_model = a_llm.get("by_model") or {}
    model_diff: dict[str, dict[str, float | int]] = {}
    for model, a_bucket in a_by_model.items():
        b_bucket = b_by_model.get(model) or {}
        delta = {}
        for k in ("input", "output", "reasoning", "calls"):
            delta[k] = int(_num(a_bucket, k) - _num(b_bucket, k))
        delta["cost_usd"] = _num(a_bucket, "cost_usd") - _num(b_bucket, "cost_usd")
        if any(delta[k] for k in ("input", "output", "reasoning", "calls")):
            model_diff[model] = delta
    b_s2 = before.get("s2") or {}
    a_s2 = after.get("s2") or {}
    return {
        "llm_tokens": int(_num(a_llm, "total_tokens") - _num(b_llm, "total_tokens")),
        "llm_calls": int(_num(a_llm, "total_calls") - _num(b_llm, "total_calls")),
        "llm_cost_usd_est": (
            _num(a_llm, "total_cost_usd_est") - _num(b_llm, "total_cost_usd_est")
        ),
        "llm_by_model": model_diff,
        "s2_api_requests": int(
            _num(a_s2, "total_api_requests") - _num(b_s2, "total_api_requests")
        ),
        "s2_cache_hits": int(
            _num(a_s2, "total_cache_hits") - _num(b_s2, "total_cache_hits")
        ),
    }


__all__ = ["SearchLogger", "NullSearchLogger"]
