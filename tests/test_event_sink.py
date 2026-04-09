"""Tests for the EventSink protocol and pipeline event emission (PE-03).

The :class:`citeclaw.event_sink.EventSink` Protocol is the contract
that the (pending) web backend's WebSocket fan-out will implement.
This file pins the contract from the runner side: ``run_pipeline``
must emit a deterministic event sequence (``step_start`` →
``paper_added`` × N → ``step_end``) per step plus one
``shape_table_update`` at the end of the run.

Tests use a tiny FakeS2-backed Context with a 2-step pipeline
(``LoadSeeds`` → ``Finalize``) so the assertions stay focused on the
event protocol rather than re-litigating the full PC-08 corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, SeedPaper, Settings
from citeclaw.context import Context
from citeclaw.event_sink import (
    EventSink,
    NullEventSink,
    RecordingEventSink,
)
from citeclaw.pipeline import run_pipeline
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Protocol-level tests — no pipeline involvement
# ---------------------------------------------------------------------------


class TestNullEventSink:
    def test_satisfies_event_sink_protocol(self):
        sink = NullEventSink()
        assert isinstance(sink, EventSink)

    def test_methods_are_no_op(self):
        sink = NullEventSink()
        # Every method must be callable, accept the documented kwargs,
        # and return None without raising.
        assert sink.step_start(idx=1, name="X", description="d") is None
        assert sink.step_end(
            idx=1, name="X",
            in_count=0, out_count=0, delta_collection=0, stats={},
        ) is None
        assert sink.paper_added(paper_id="p1", source="seed") is None
        assert sink.paper_rejected(paper_id="p2", category="year") is None
        assert sink.shape_table_update(rendered_shape="table") is None


class TestRecordingEventSink:
    def test_satisfies_event_sink_protocol(self):
        sink = RecordingEventSink()
        assert isinstance(sink, EventSink)

    def test_captures_events_in_order(self):
        sink = RecordingEventSink()
        sink.step_start(idx=1, name="LoadSeeds", description="load seeds")
        sink.paper_added(paper_id="p1", source="seed")
        sink.paper_added(paper_id="p2", source="seed")
        sink.step_end(
            idx=1, name="LoadSeeds",
            in_count=0, out_count=2, delta_collection=2,
            stats={"loaded": 2},
        )
        sink.shape_table_update(rendered_shape="rendered")

        assert sink.names() == [
            "step_start",
            "paper_added",
            "paper_added",
            "step_end",
            "shape_table_update",
        ]
        assert sink.events[0][1]["name"] == "LoadSeeds"
        assert sink.events[1][1]["paper_id"] == "p1"
        assert sink.events[3][1]["delta_collection"] == 2
        assert sink.events[4][1]["rendered_shape"] == "rendered"

    def test_of_helper_filters_by_kind(self):
        sink = RecordingEventSink()
        sink.step_start(idx=1, name="A", description="")
        sink.paper_added(paper_id="p1", source="seed")
        sink.paper_added(paper_id="p2", source="seed")
        sink.step_end(
            idx=1, name="A", in_count=0, out_count=2,
            delta_collection=2, stats={},
        )

        added = sink.of("paper_added")
        assert len(added) == 2
        assert {a["paper_id"] for a in added} == {"p1", "p2"}

    def test_step_end_stats_dict_is_copied(self):
        """Mutating the stats dict after step_end must not affect the
        recorded payload — RecordingEventSink takes a defensive copy."""
        sink = RecordingEventSink()
        live_stats = {"loaded": 1}
        sink.step_end(
            idx=1, name="A",
            in_count=0, out_count=1, delta_collection=1,
            stats=live_stats,
        )
        live_stats["mutated"] = "after"
        recorded = sink.of("step_end")[0]
        assert recorded["stats"] == {"loaded": 1}
        assert "mutated" not in recorded["stats"]


# ---------------------------------------------------------------------------
# Pipeline integration — run_pipeline must emit the event sequence
# ---------------------------------------------------------------------------


def _build_minimal_pipeline_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Context:
    """A 2-step pipeline (LoadSeeds → Finalize) over a 2-paper corpus.

    Kept tiny so the event-sequence assertions stay focused on the
    runner contract rather than dragging in the PC-08 corpus.
    """
    monkeypatch.setenv("CITECLAW_NO_DASHBOARD", "1")
    fs = FakeS2Client()
    fs.add(make_paper("EVT-1", title="Event Seed 1", year=2022))
    fs.add(make_paper("EVT-2", title="Event Seed 2", year=2023))

    cfg = Settings(
        screening_model="stub",
        data_dir=tmp_path / "evt_data",
        topic_description="Event sink test corpus",
        seed_papers=[
            SeedPaper(paper_id="EVT-1"),
            SeedPaper(paper_id="EVT-2"),
        ],
        max_papers_total=10_000,
        blocks={},
        pipeline=[
            {"step": "LoadSeeds"},
            {"step": "Finalize"},
        ],
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(cfg.data_dir / "cache.db")
    budget = BudgetTracker()
    return Context(config=cfg, s2=fs, cache=cache, budget=budget)


class TestRunPipelineEventEmission:
    def test_run_pipeline_default_sink_is_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Calling ``run_pipeline`` without an explicit event_sink
        must not raise — it picks up :class:`NullEventSink`."""
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        result = run_pipeline(ctx)
        assert isinstance(result, dict)
        # Collection has the seeds.
        assert len(ctx.collection) == 2

    def test_run_pipeline_emits_step_start_end_for_each_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        starts = sink.of("step_start")
        ends = sink.of("step_end")
        # 2 user-defined steps + 1 auto-injected MergeDuplicates =
        # 3 step_start / step_end pairs.
        assert len(starts) == len(ends)
        assert len(starts) == 3

        names_in = [s["name"] for s in starts]
        names_out = [e["name"] for e in ends]
        assert names_in == ["LoadSeeds", "MergeDuplicates", "Finalize"]
        assert names_out == ["LoadSeeds", "MergeDuplicates", "Finalize"]

    def test_run_pipeline_emits_paper_added_for_loaded_seeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """LoadSeeds adds 2 seeds to ctx.collection → 2 paper_added
        events between its step_start and step_end."""
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        added = sink.of("paper_added")
        assert len(added) == 2
        added_ids = {a["paper_id"] for a in added}
        assert added_ids == {"EVT-1", "EVT-2"}
        # Each added event carries the source label stamped by LoadSeeds.
        for entry in added:
            assert entry["source"] == "seed"

    def test_paper_added_events_land_inside_their_step_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The paper_added events for seeds must appear AFTER
        LoadSeeds' step_start and BEFORE LoadSeeds' step_end — that's
        the runner's per-step window contract."""
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        names = sink.names()
        load_start_idx = names.index("step_start")  # first step_start = LoadSeeds
        load_end_idx = next(
            i for i, n in enumerate(names[load_start_idx:], start=load_start_idx)
            if n == "step_end"
        )
        # Every paper_added between load_start_idx and load_end_idx is
        # the LoadSeeds delta.
        added_inside = [
            n for n in names[load_start_idx + 1: load_end_idx]
            if n == "paper_added"
        ]
        assert len(added_inside) == 2

    def test_run_pipeline_emits_shape_table_update_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        updates = sink.of("shape_table_update")
        assert len(updates) == 1
        rendered = updates[0]["rendered_shape"]
        # The rendered shape table mentions the user-defined steps.
        assert "LoadSeeds" in rendered
        assert "Finalize" in rendered

    def test_step_end_payload_carries_in_out_delta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        load_end = next(
            e for e in sink.of("step_end") if e["name"] == "LoadSeeds"
        )
        assert load_end["in_count"] == 0
        assert load_end["out_count"] == 2
        assert load_end["delta_collection"] == 2
        assert isinstance(load_end["stats"], dict)
        assert load_end["stats"].get("loaded") == 2

    def test_event_sequence_is_well_formed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Every step_start must be matched by a step_end with the
        same name+idx, in order. shape_table_update fires exactly
        once at the end."""
        ctx = _build_minimal_pipeline_ctx(tmp_path, monkeypatch)
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        # Walk the event stream and check matching.
        open_steps: list[tuple[int, str]] = []
        shape_updates = 0
        for name, payload in sink.events:
            if name == "step_start":
                open_steps.append((payload["idx"], payload["name"]))
            elif name == "step_end":
                assert open_steps, f"step_end with no open step: {payload}"
                idx, step_name = open_steps.pop()
                assert payload["idx"] == idx
                assert payload["name"] == step_name
            elif name == "shape_table_update":
                shape_updates += 1
        assert open_steps == [], f"unclosed steps: {open_steps}"
        assert shape_updates == 1
