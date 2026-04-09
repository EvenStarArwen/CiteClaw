"""Pipeline event sink — abstract interface for streaming run events.

PE-03: ``run_pipeline`` accepts an optional ``event_sink`` keyword that
defaults to :class:`NullEventSink` (a no-op that preserves the legacy
CLI behavior). The web backend will wire in a fan-out sink that
forwards every event to all subscribed WebSocket clients (PE-09 lands
on this same protocol).

Event taxonomy
--------------

  * ``step_start(idx, name, description)`` — fired immediately before
    ``step.run()`` is called.
  * ``step_end(idx, name, in_count, out_count, delta_collection, stats)``
    — fired after ``step.run()`` returns and the shape table row has
    been recorded.
  * ``paper_added(paper_id, source)`` — synthesised by the runner: for
    every paper that's in ``ctx.collection`` after a step but wasn't
    before, one ``paper_added`` event is emitted between
    ``step_start`` and ``step_end``.
  * ``paper_rejected(paper_id, category)`` — declared on the Protocol
    so third-party sinks can wire it up early. Not emitted by the
    runner in v1; future step-level callers will fan out via their
    own sink reference.
  * ``shape_table_update(rendered_shape)`` — fired once at the end of
    the run with the rendered shape table.

Implementations
---------------

  * :class:`NullEventSink` — no-op default, used by every existing
    test that calls ``run_pipeline`` without an explicit sink.
  * :class:`RecordingEventSink` — captures every event in order. Used
    by ``tests/test_event_sink.py`` and any future test asserting
    event ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """Protocol for sinks consuming pipeline run events.

    The default in-process sink is :class:`NullEventSink`. The web
    backend will provide a fan-out sink in PE-04+ that forwards events
    to WebSocket subscribers.
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


class NullEventSink:
    """No-op sink. Used as the default so existing CLI runs are
    unchanged when no caller provides an explicit sink."""

    def step_start(self, idx: int, name: str, description: str) -> None:
        return None

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None:
        return None

    def paper_added(self, paper_id: str, source: str) -> None:
        return None

    def paper_rejected(self, paper_id: str, category: str) -> None:
        return None

    def shape_table_update(self, rendered_shape: str) -> None:
        return None


@dataclass
class RecordingEventSink:
    """Test fixture that captures every event in order.

    Each entry in :attr:`events` is a ``(name, payload)`` tuple where
    ``name`` is the EventSink method name and ``payload`` is a dict
    of the kwargs that were passed. Tests assert on the resulting
    sequence to verify pipeline event semantics.
    """

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def step_start(self, idx: int, name: str, description: str) -> None:
        self.events.append((
            "step_start",
            {"idx": idx, "name": name, "description": description},
        ))

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None:
        self.events.append((
            "step_end",
            {
                "idx": idx,
                "name": name,
                "in_count": in_count,
                "out_count": out_count,
                "delta_collection": delta_collection,
                "stats": dict(stats),
            },
        ))

    def paper_added(self, paper_id: str, source: str) -> None:
        self.events.append((
            "paper_added",
            {"paper_id": paper_id, "source": source},
        ))

    def paper_rejected(self, paper_id: str, category: str) -> None:
        self.events.append((
            "paper_rejected",
            {"paper_id": paper_id, "category": category},
        ))

    def shape_table_update(self, rendered_shape: str) -> None:
        self.events.append((
            "shape_table_update",
            {"rendered_shape": rendered_shape},
        ))

    # ----- convenience accessors used by tests -----

    def names(self) -> list[str]:
        """Return just the event-name sequence (no payloads)."""
        return [name for name, _ in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        """Return all payloads matching ``kind``."""
        return [payload for name, payload in self.events if name == kind]
