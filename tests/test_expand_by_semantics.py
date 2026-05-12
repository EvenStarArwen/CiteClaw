"""Tests for ``citeclaw.steps.expand_by_semantics.ExpandBySemantics``."""

from __future__ import annotations

import pytest

from citeclaw.context import Context
from citeclaw.filters.builder import build_blocks
from citeclaw.models import PaperRecord
from citeclaw.steps.expand_by_semantics import ExpandBySemantics
from tests.fakes import FakeS2Client, make_paper


def _basic_screener():
    blocks = build_blocks({
        "y": {"type": "YearFilter", "min": 2020},
        "screen": {"type": "Sequential", "layers": ["y"]},
    })
    return blocks["screen"]


# ---------------------------------------------------------------------------
# Run-loop early-exit guards
# ---------------------------------------------------------------------------


class TestEarlyExits:
    def test_no_screener_passes_signal_through(self, ctx: Context):
        # Augmentation steps must pass the input signal through unchanged
        # when they have nothing to add — otherwise the downstream
        # snowball dies on an empty signal even though ctx.collection
        # still has every accepted paper.
        signal = [PaperRecord(paper_id="p1"), PaperRecord(paper_id="p2")]
        result = ExpandBySemantics().run(signal, ctx)
        assert [p.paper_id for p in result.signal] == ["p1", "p2"]
        assert result.stats == {"reason": "no screener"}

    def test_no_anchors_passes_signal_through(self, ctx: Context):
        # Signal of records without paper_ids leaves no usable anchor.
        signal = [PaperRecord(paper_id="")]
        result = ExpandBySemantics(screener=_basic_screener()).run(signal, ctx)
        assert result.stats["reason"] == "no_anchors"
        # Pass-through still preserves the (useless) input rather than
        # nuking the signal — kept here so the contract is explicit.
        assert len(result.signal) == 1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_recommendations_screened_and_collected(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        anchor = make_paper("anchor", year=2024)
        rec_old = make_paper("rec_old", year=2010)
        rec_new = make_paper("rec_new", year=2024)
        fake_s2.add(anchor)
        fake_s2.add(rec_old)
        fake_s2.add(rec_new)
        fake_s2.register_recommendations(
            ["anchor"],
            [{"paperId": "rec_old"}, {"paperId": "rec_new"}],
        )

        signal = [PaperRecord(paper_id="anchor")]
        step = ExpandBySemantics(
            screener=_basic_screener(),
            max_anchor_papers=1,
            limit=10,
        )
        result = step.run(signal, ctx)

        # Output signal is the input anchor PLUS any newly-screened
        # neighbours (augmentation, not consumption).
        passed_ids = [p.paper_id for p in result.signal]
        assert passed_ids == ["anchor", "rec_new"]
        assert result.stats["anchor_count"] == 1
        assert result.stats["accepted"] == 1
        assert result.stats["rejected"] == 1
        assert "rec_new" in ctx.collection
        assert ctx.collection["rec_new"].source == "semantic"

    def test_max_anchor_papers_caps_signal(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        for pid in ("a1", "a2", "a3"):
            fake_s2.add(make_paper(pid, year=2024))
        # Register against the (sorted) tuple of the FIRST anchor only.
        fake_s2.register_recommendations(["a1"], [])

        signal = [PaperRecord(paper_id=p) for p in ("a1", "a2", "a3")]
        step = ExpandBySemantics(
            screener=_basic_screener(), max_anchor_papers=1, limit=10,
        )
        result = step.run(signal, ctx)
        assert result.stats["anchor_count"] == 1

    def test_negative_anchors_passed_when_flag_set(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        anchor = make_paper("anchor", year=2024)
        fake_s2.add(anchor)
        fake_s2.register_recommendations(["anchor"], [])
        ctx.rejected.update({"reject_a", "reject_b"})

        signal = [PaperRecord(paper_id="anchor")]
        step = ExpandBySemantics(
            screener=_basic_screener(),
            max_anchor_papers=1,
            limit=10,
            use_rejected_as_negatives=True,
        )
        result = step.run(signal, ctx)
        assert result.stats["negative_count"] == 2

    def test_idempotent_second_run(self, ctx: Context, fake_s2: FakeS2Client):
        anchor = make_paper("anchor", year=2024)
        rec = make_paper("rec1", year=2024)
        fake_s2.add(anchor)
        fake_s2.add(rec)
        fake_s2.register_recommendations(["anchor"], [{"paperId": "rec1"}])

        signal = [PaperRecord(paper_id="anchor")]
        step = ExpandBySemantics(screener=_basic_screener(), max_anchor_papers=1)
        first = step.run(signal, ctx)
        assert first.stats["accepted"] == 1

        second = step.run(signal, ctx)
        assert second.stats["reason"] == "already_searched"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_fetch_recommendations_failure_returns_stats(
        self, ctx: Context, fake_s2: FakeS2Client, monkeypatch: pytest.MonkeyPatch,
    ):
        anchor = make_paper("anchor", year=2024)
        fake_s2.add(anchor)

        def boom(*a, **kw):
            raise RuntimeError("S2 down")

        monkeypatch.setattr(fake_s2, "fetch_recommendations", boom)
        signal = [PaperRecord(paper_id="anchor")]
        result = ExpandBySemantics(screener=_basic_screener()).run(signal, ctx)
        assert result.stats["reason"] == "fetch_failed"
        assert "S2 down" in result.stats["error"]


# ---------------------------------------------------------------------------
# Per-paper mode
# ---------------------------------------------------------------------------


class TestPerPaperMode:
    def test_one_call_per_anchor_aggregates_neighbours(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        a1 = make_paper("a1", year=2024)
        a2 = make_paper("a2", year=2024)
        n_a1 = make_paper("n_a1", year=2024)
        n_a2 = make_paper("n_a2", year=2024)
        n_shared = make_paper("n_shared", year=2024)
        for p in (a1, a2, n_a1, n_a2, n_shared):
            fake_s2.add(p)
        fake_s2.register_recommendations_for_paper(
            "a1", [{"paperId": "n_a1"}, {"paperId": "n_shared"}],
        )
        fake_s2.register_recommendations_for_paper(
            "a2", [{"paperId": "n_a2"}, {"paperId": "n_shared"}],
        )

        signal = [PaperRecord(paper_id="a1"), PaperRecord(paper_id="a2")]
        step = ExpandBySemantics(
            screener=_basic_screener(),
            mode="per_paper",
            recs_per_paper=10,
            max_workers=2,
        )
        result = step.run(signal, ctx)

        # Anchors flow through; new screened neighbours append.
        passed_ids = {p.paper_id for p in result.signal}
        assert passed_ids == {"a1", "a2", "n_a1", "n_a2", "n_shared"}
        assert result.stats["mode"] == "per_paper"
        assert result.stats["anchor_count"] == 2
        # Two registered anchors -> two fan-out calls.
        assert fake_s2.calls.get("fetch_recommendations_for_paper") == 2

    def test_per_paper_uses_entire_signal_no_anchor_cap(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        """max_anchor_papers is multi_anchor-only — per_paper ignores it."""
        for pid in ("p1", "p2", "p3", "p4", "p5"):
            fake_s2.add(make_paper(pid, year=2024))
            fake_s2.register_recommendations_for_paper(pid, [])
        signal = [PaperRecord(paper_id=p) for p in ("p1", "p2", "p3", "p4", "p5")]
        step = ExpandBySemantics(
            screener=_basic_screener(),
            mode="per_paper",
            max_anchor_papers=2,  # would cap in multi_anchor mode, ignored here
        )
        result = step.run(signal, ctx)
        assert result.stats["anchor_count"] == 5
        assert fake_s2.calls.get("fetch_recommendations_for_paper") == 5

    def test_unknown_mode_falls_back_to_multi_anchor(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        anchor = make_paper("anchor", year=2024)
        fake_s2.add(anchor)
        fake_s2.register_recommendations(["anchor"], [])
        signal = [PaperRecord(paper_id="anchor")]
        step = ExpandBySemantics(screener=_basic_screener(), mode="bogus")
        result = step.run(signal, ctx)
        assert result.stats["mode"] == "multi_anchor"
