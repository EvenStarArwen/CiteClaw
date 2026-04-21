"""Pipeline event sink — abstract interface for streaming run events.

:func:`citeclaw.pipeline.run_pipeline` accepts an optional ``event_sink``
keyword that defaults to :class:`NullEventSink`. The web backend wires
in a fan-out sink that forwards each event to its WebSocket subscribers.

Event taxonomy
--------------

  * ``step_start(idx, name, description)`` — fired immediately before
    ``step.run()`` is called.
  * ``step_end(idx, name, in_count, out_count, delta_collection, stats)``
    — fired after ``step.run()`` returns and the shape table row has
    been recorded. ``stats`` is defensively copied before forwarding.
  * ``paper_added(paper_id, source)`` — synthesised by the runner: one
    event per paper that appears in ``ctx.collection`` after a step
    but wasn't there before, emitted between ``step_start`` and
    ``step_end``.
  * ``paper_rejected(paper_id, category)`` — declared on the Protocol
    for third-party sinks; not emitted by the runner today (rejection
    bookkeeping currently flows through ``ctx.rejection_ledger``
    instead).
  * ``shape_table_update(rendered_shape)`` — fired once at run end.
  * ``hitl_request(run_id, papers)`` — emitted only when
    ``HumanInTheLoop`` enters web mode; the backend holds the event
    until the user POSTs labels.

Implementations
---------------

  * :class:`NullEventSink` — no-op default for CLI runs.
  * :class:`RecordingEventSink` — captures every event in order; used
    by tests to assert event sequencing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """Protocol every pipeline event sink satisfies.

    All methods return ``None``. The runner calls them in a fixed order
    documented in the module docstring; concrete sinks should not raise
    from any of these methods (a raising sink would crash the run).
    """

    def step_start(self, idx: int, name: str, description: str) -> None: ...

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None: ...

    def paper_added(self, paper_id: str, source: str) -> None: ...

    def paper_rejected(self, paper_id: str, category: str) -> None: ...

    def shape_table_update(self, rendered_shape: str) -> None: ...

    def hitl_request(
        self,
        run_id: str,
        papers: list[dict[str, Any]],
    ) -> None:
        """Emitted when ``HumanInTheLoop`` enters web mode and needs
        user labels for the given paper sample. Each dict in *papers*
        carries ``paper_id``, ``title``, ``venue``, ``year``,
        ``abstract`` (truncated). The backend holds this event until
        the user submits labels via ``POST /api/runs/{run_id}/hitl``."""
        ...


class NullEventSink:
    """No-op sink — used as the default when no caller provides one."""

    def step_start(self, idx: int, name: str, description: str) -> None:
        pass

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None:
        pass

    def paper_added(self, paper_id: str, source: str) -> None:
        pass

    def paper_rejected(self, paper_id: str, category: str) -> None:
        pass

    def shape_table_update(self, rendered_shape: str) -> None:
        pass

    def hitl_request(
        self,
        run_id: str,
        papers: list[dict[str, Any]],
    ) -> None:
        pass


@dataclass
class RecordingEventSink:
    """Test fixture that captures every event in order.

    Each entry in :attr:`events` is a ``(name, payload)`` tuple where
    ``name`` is the EventSink method name and ``payload`` is a dict of
    the kwargs that were passed. Mutable container fields in the
    payload (``stats``, ``papers``) are defensively copied so caller
    mutations after the call don't bleed into the recorded sequence.
    Tests assert on the resulting sequence to verify pipeline event
    semantics.
    """

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _record(self, _name: str, **payload: Any) -> None:
        self.events.append((_name, payload))

    def step_start(self, idx: int, name: str, description: str) -> None:
        self._record("step_start", idx=idx, name=name, description=description)

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None:
        self._record(
            "step_end",
            idx=idx,
            name=name,
            in_count=in_count,
            out_count=out_count,
            delta_collection=delta_collection,
            stats=dict(stats),
        )

    def paper_added(self, paper_id: str, source: str) -> None:
        self._record("paper_added", paper_id=paper_id, source=source)

    def paper_rejected(self, paper_id: str, category: str) -> None:
        self._record("paper_rejected", paper_id=paper_id, category=category)

    def shape_table_update(self, rendered_shape: str) -> None:
        self._record("shape_table_update", rendered_shape=rendered_shape)

    def hitl_request(
        self,
        run_id: str,
        papers: list[dict[str, Any]],
    ) -> None:
        self._record("hitl_request", run_id=run_id, papers=list(papers))

    # ----- convenience accessors used by tests -----

    def names(self) -> list[str]:
        """Return just the event-name sequence (no payloads)."""
        return [name for name, _ in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        """Return all payloads matching ``kind``."""
        return [payload for name, payload in self.events if name == kind]
