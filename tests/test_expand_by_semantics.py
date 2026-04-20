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
    def test_no_screener_returns_empty(self, ctx: Context):
        signal = [PaperRecord(paper_id="p1")]
        result = ExpandBySemantics().run(signal, ctx)
        assert result.signal == []
        assert result.stats == {"reason": "no screener"}

    def test_no_anchors_returns_empty(self, ctx: Context):
        # Signal of records without paper_ids leaves no usable anchor.
        signal = [PaperRecord(paper_id="")]
        result = ExpandBySemantics(screener=_basic_screener()).run(signal, ctx)
        assert result.stats["reason"] == "no_anchors"


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

        passed_ids = {p.paper_id for p in result.signal}
        assert passed_ids == {"rec_new"}
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
