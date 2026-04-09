"""Tests for the ReinforceGraph step (PD-01).

The headline scenario from the spec: a hand-built collection + seen
set where a high-pagerank rejected paper is restored. The test
constructs a tiny universe where one rejected paper (REJ_HIGH) cites
several collection papers — giving it many incoming edges in the
combined citation graph and therefore a meaningfully higher PageRank
than its peer (REJ_LOW), which has no references.

ReinforceGraph hydrates the rejected pool via ``ctx.s2.fetch_metadata``,
builds the combined citation graph, runs PageRank, applies the
``percentile_floor + top_n`` cap, re-screens the survivors with a
source-less ``FilterContext``, and restores the survivors into
``ctx.collection`` with ``source="reinforced"`` plus an entry in
``ctx.reinforcement_log``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.filters.builder import build_blocks
from citeclaw.models import PaperRecord
from citeclaw.steps import build_step
from citeclaw.steps.reinforce_graph import ReinforceGraph
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_ctx_with_collection(
    tmp_path: Path,
    fs: FakeS2Client,
    collection: dict[str, PaperRecord],
    seen: set[str],
) -> Context:
    cfg = Settings(data_dir=tmp_path, screening_model="stub")
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    ctx = Context(config=cfg, s2=fs, cache=cache, budget=budget)
    ctx.collection.update(collection)
    ctx.seen.update(seen)
    return ctx


def _make_record(
    pid: str,
    *,
    references: list[str] | None = None,
    year: int = 2020,
    citation_count: int = 100,
    venue: str = "Nature",
    title: str | None = None,
) -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=title or f"Paper {pid}",
        abstract=f"Abstract of {pid}.",
        year=year,
        venue=venue,
        citation_count=citation_count,
        references=references or [],
    )


def _build_corpus_with_high_low_rejected() -> tuple[
    FakeS2Client, dict[str, PaperRecord], set[str],
]:
    """The headline test fixture: 4 collection papers (C1..C4) plus
    two rejected papers in ctx.seen — REJ_HIGH (cites C1/C2/C3, so it
    accumulates incoming edges in the combined graph) and REJ_LOW
    (no references, orphan node, baseline pagerank).

    REJ_HIGH and REJ_LOW are pre-registered in the FakeS2Client so
    ``fetch_metadata`` returns them with their full reference lists.
    """
    fs = FakeS2Client()

    # Collection papers — all with empty references so they don't
    # contribute their own incoming edges to anybody.
    collection: dict[str, PaperRecord] = {
        "C1": _make_record("C1", year=2019, citation_count=500),
        "C2": _make_record("C2", year=2020, citation_count=400),
        "C3": _make_record("C3", year=2021, citation_count=300),
        "C4": _make_record("C4", year=2022, citation_count=200),
    }

    # Pre-register REJ_HIGH in the fake S2 corpus with references
    # pointing at C1/C2/C3 — those refs become incoming edges to
    # REJ_HIGH in the combined citation graph (cited→citer convention,
    # i.e. each ref C → REJ_HIGH means flow accumulates at REJ_HIGH).
    fs.add(
        make_paper(
            "REJ_HIGH",
            title="High-PageRank Rejected Paper",
            year=2018,
            citation_count=200,
            venue="Nature",
            references=["C1", "C2", "C3"],
        )
    )
    # REJ_LOW has no references → no incoming edges → only the
    # PageRank teleport baseline. It must NOT be picked over REJ_HIGH.
    fs.add(
        make_paper(
            "REJ_LOW",
            title="Low-PageRank Rejected Paper",
            year=2018,
            citation_count=10,
            venue="arXiv",
            references=[],
        )
    )

    seen = set(collection.keys()) | {"REJ_HIGH", "REJ_LOW"}
    return fs, collection, seen


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------


class TestReinforceGraphConstructor:
    def test_default_args(self):
        step = ReinforceGraph(screener=object())
        assert step.metric == "pagerank"
        assert step.top_n == 30
        assert step.percentile_floor == 0.9

    def test_invalid_metric_raises_at_init(self):
        with pytest.raises(ValueError, match="pagerank"):
            ReinforceGraph(screener=object(), metric="betweenness")

    def test_invalid_percentile_floor_raises(self):
        with pytest.raises(ValueError, match="percentile_floor"):
            ReinforceGraph(screener=object(), percentile_floor=1.5)
        with pytest.raises(ValueError, match="percentile_floor"):
            ReinforceGraph(screener=object(), percentile_floor=-0.1)


# ---------------------------------------------------------------------------
# Headline test: high-pagerank rejected paper is restored
# ---------------------------------------------------------------------------


class TestReinforceGraphHighPagerankRescue:
    def test_high_pagerank_rejected_paper_is_restored(self, tmp_path: Path):
        fs, collection, seen = _build_corpus_with_high_low_rejected()
        ctx = _build_ctx_with_collection(tmp_path, fs, collection, seen)

        blocks = build_blocks(
            {"yr": {"type": "YearFilter", "min": 2015, "max": 2030}},
        )
        step = ReinforceGraph(
            screener=blocks["yr"],
            metric="pagerank",
            top_n=30,
            percentile_floor=0.9,
        )
        result = step.run([], ctx)

        # REJ_HIGH must be restored.
        assert "REJ_HIGH" in ctx.collection
        assert ctx.collection["REJ_HIGH"].source == "reinforced"
        assert ctx.collection["REJ_HIGH"].llm_verdict == "accept"

        # REJ_LOW must NOT be restored — its baseline pagerank is below
        # the percentile floor.
        assert "REJ_LOW" not in ctx.collection

        # The reinforcement_log gets one entry per restored paper.
        assert len(ctx.reinforcement_log) == 1
        log_entry = ctx.reinforcement_log[0]
        assert log_entry["paper_id"] == "REJ_HIGH"
        assert log_entry["metric"] == "pagerank"
        assert log_entry["score"] > 0
        assert "rescued" in log_entry["reason"]

        # Stats reflect the rescue path.
        assert result.stats["rejected_pool"] == 2
        assert result.stats["after_floor"] == 1
        assert result.stats["candidates_screened"] == 1
        assert result.stats["accepted"] == 1
        assert result.stats["rejected_again"] == 0
        assert result.stats["metric"] == "pagerank"

    def test_screener_can_still_reject_a_high_pagerank_candidate(
        self, tmp_path: Path,
    ):
        """Even REJ_HIGH gets dropped if the rescue screener rejects
        it — proving the second-pass filter is honoured."""
        fs, collection, seen = _build_corpus_with_high_low_rejected()
        ctx = _build_ctx_with_collection(tmp_path, fs, collection, seen)

        # Strict year filter that REJ_HIGH (year=2018) doesn't satisfy.
        blocks = build_blocks(
            {"yr_strict": {"type": "YearFilter", "min": 2025}},
        )
        step = ReinforceGraph(
            screener=blocks["yr_strict"],
            top_n=30,
            percentile_floor=0.9,
        )
        result = step.run([], ctx)

        assert "REJ_HIGH" not in ctx.collection
        assert ctx.reinforcement_log == []
        assert result.stats["accepted"] == 0
        assert result.stats["rejected_again"] == 1


# ---------------------------------------------------------------------------
# Short-circuit / edge cases
# ---------------------------------------------------------------------------


class TestReinforceGraphShortCircuits:
    def test_no_screener_returns_empty(self, tmp_path: Path):
        fs = FakeS2Client()
        ctx = _build_ctx_with_collection(tmp_path, fs, {}, set())
        step = ReinforceGraph(screener=None)
        result = step.run([], ctx)
        assert result.signal == []
        assert result.stats["reason"] == "no screener"

    def test_no_rejected_pool(self, tmp_path: Path):
        """When ctx.seen ⊆ ctx.collection there's nothing to rescue."""
        fs = FakeS2Client()
        collection = {"C1": _make_record("C1")}
        ctx = _build_ctx_with_collection(tmp_path, fs, collection, {"C1"})
        blocks = build_blocks({"yr": {"type": "YearFilter", "min": 2015, "max": 2030}})
        step = ReinforceGraph(screener=blocks["yr"])
        result = step.run([], ctx)
        assert result.stats["reason"] == "no_rejected"

    def test_rejected_papers_with_no_cached_metadata_use_placeholder(
        self, tmp_path: Path,
    ):
        """When fetch_metadata fails for a rejected paper, the step
        uses a placeholder PaperRecord and keeps going (no crash)."""
        fs = FakeS2Client()
        collection = {"C1": _make_record("C1")}
        # ctx.seen has a rejected paper that's NOT in the fake corpus
        # → fetch_metadata raises KeyError → placeholder is used.
        ctx = _build_ctx_with_collection(
            tmp_path, fs, collection, {"C1", "MISSING"},
        )
        blocks = build_blocks({"yr": {"type": "YearFilter", "min": 2015, "max": 2030}})
        step = ReinforceGraph(
            screener=blocks["yr"],
            top_n=30,
            percentile_floor=0.0,  # disable floor so MISSING can survive
        )
        result = step.run([], ctx)
        # MISSING is a placeholder with year=None → YearFilter rejects it.
        # Step still runs to completion.
        assert result.stats["rejected_pool"] == 1
        assert result.stats["accepted"] == 0


# ---------------------------------------------------------------------------
# Percentile floor mechanics
# ---------------------------------------------------------------------------


class TestReinforceGraphPercentileFloor:
    def test_floor_zero_keeps_everything(self, tmp_path: Path):
        fs, collection, seen = _build_corpus_with_high_low_rejected()
        ctx = _build_ctx_with_collection(tmp_path, fs, collection, seen)
        blocks = build_blocks({"yr": {"type": "YearFilter", "min": 2015, "max": 2030}})
        step = ReinforceGraph(
            screener=blocks["yr"],
            top_n=30,
            percentile_floor=0.0,
        )
        result = step.run([], ctx)
        # Both REJ_HIGH and REJ_LOW survive the floor + screener.
        assert result.stats["after_floor"] == 2
        assert result.stats["accepted"] == 2

    def test_top_n_caps_the_rescue(self, tmp_path: Path):
        fs, collection, seen = _build_corpus_with_high_low_rejected()
        ctx = _build_ctx_with_collection(tmp_path, fs, collection, seen)
        blocks = build_blocks({"yr": {"type": "YearFilter", "min": 2015, "max": 2030}})
        step = ReinforceGraph(
            screener=blocks["yr"],
            top_n=1,
            percentile_floor=0.0,  # floor disabled so the cap is the only filter
        )
        result = step.run([], ctx)
        # top_n=1 cuts to one candidate even though both passed the floor.
        assert result.stats["candidates_screened"] == 1
        assert result.stats["accepted"] == 1
        # REJ_HIGH wins by score.
        assert "REJ_HIGH" in ctx.collection


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestReinforceGraphRegistry:
    def test_in_step_registry(self):
        from citeclaw.steps import STEP_REGISTRY
        assert "ReinforceGraph" in STEP_REGISTRY

    def test_build_step_with_kwargs(self):
        step = build_step(
            {
                "step": "ReinforceGraph",
                "metric": "pagerank",
                "top_n": 5,
                "percentile_floor": 0.8,
            },
            blocks={},
        )
        assert isinstance(step, ReinforceGraph)
        assert step.top_n == 5
        assert step.percentile_floor == 0.8
