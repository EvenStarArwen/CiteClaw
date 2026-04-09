"""Tests for the ResolveSeeds step (PC-04).

The step expands ``ctx.config.seed_papers`` (now a mix of
``{paper_id: ...}`` and ``{title: ...}`` entries) into a flat
``ctx.resolved_seed_ids`` list. With ``include_siblings=True`` it also
walks each primary's ``external_ids`` and adds preprint↔published
siblings whose S2 ``paperId`` differs from the primary.

The headline scenario from the spec is "a title that resolves to two
distinct paper_ids (preprint + published)" — exercised end-to-end
in :class:`TestResolveSeedsPreprintPublishedPair`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, SeedPaper, Settings
from citeclaw.context import Context
from citeclaw.steps import build_step
from citeclaw.steps.resolve_seeds import ResolveSeeds
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_ctx(
    tmp_path: Path,
    fake_s2: FakeS2Client,
    seed_papers: list[SeedPaper],
) -> Context:
    cfg = Settings(
        data_dir=tmp_path,
        screening_model="stub",
        seed_papers=seed_papers,
    )
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    return Context(config=cfg, s2=fake_s2, cache=cache, budget=budget)


def _add_paper_under(
    fs: FakeS2Client,
    *aliases: str,
    paper: dict[str, Any],
) -> None:
    """Register the same paper dict under several lookup keys.

    The base FakeS2Client looks up by literal paper_id; the test
    needs DOI/ARXIV-prefixed lookups to resolve to the same dict so
    ``fetch_metadata("DOI:10.1/foo")`` returns the canonical record.
    """
    for alias in aliases:
        fs._papers[alias] = paper


# ---------------------------------------------------------------------------
# Schema relaxation: SeedPaper.paper_id is now optional
# ---------------------------------------------------------------------------


class TestSeedPaperSchema:
    def test_title_only_seed_validates(self):
        sp = SeedPaper(title="Some Paper Title")
        assert sp.paper_id == ""
        assert sp.title == "Some Paper Title"

    def test_paper_id_only_seed_still_works(self):
        sp = SeedPaper(paper_id="DOI:10.1/abc")
        assert sp.paper_id == "DOI:10.1/abc"
        assert sp.title == ""

    def test_both_fields_set(self):
        sp = SeedPaper(paper_id="DOI:10.1/abc", title="Some Paper")
        assert sp.paper_id == "DOI:10.1/abc"
        assert sp.title == "Some Paper"


# ---------------------------------------------------------------------------
# ResolveSeeds — direct paper_id entries pass through unchanged
# ---------------------------------------------------------------------------


class TestResolveSeedsDirectIds:
    def test_paper_id_only_passes_through(self, tmp_path: Path):
        fs = FakeS2Client()
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(paper_id="P1"), SeedPaper(paper_id="P2")],
        )
        step = ResolveSeeds()
        result = step.run([], ctx)
        assert ctx.resolved_seed_ids == ["P1", "P2"]
        assert result.stats["primaries_resolved"] == 2
        assert result.stats["siblings_added"] == 0

    def test_dedup_across_seeds(self, tmp_path: Path):
        fs = FakeS2Client()
        ctx = _build_ctx(
            tmp_path, fs,
            [
                SeedPaper(paper_id="P1"),
                SeedPaper(paper_id="P1"),  # duplicate
                SeedPaper(paper_id="P2"),
            ],
        )
        step = ResolveSeeds()
        step.run([], ctx)
        assert ctx.resolved_seed_ids == ["P1", "P2"]


# ---------------------------------------------------------------------------
# ResolveSeeds — title-only entries route through search_match
# ---------------------------------------------------------------------------


class TestResolveSeedsTitleEntries:
    def test_title_resolves_via_search_match(self, tmp_path: Path):
        fs = FakeS2Client()
        fs.register_search_match("Foundational Paper", make_paper("FOUND-1"))
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(title="Foundational Paper")],
        )
        ResolveSeeds().run([], ctx)
        assert ctx.resolved_seed_ids == ["FOUND-1"]

    def test_unmatched_title_skipped_with_log(self, tmp_path: Path):
        fs = FakeS2Client()
        # No registration → search_match returns None.
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(title="Nonexistent Title")],
        )
        result = ResolveSeeds().run([], ctx)
        assert ctx.resolved_seed_ids == []
        assert result.stats["unresolved_titles"] == 1
        assert result.stats["primaries_resolved"] == 0

    def test_mixed_paper_id_and_title_entries(self, tmp_path: Path):
        fs = FakeS2Client()
        fs.register_search_match("Some Paper", make_paper("FROM-TITLE"))
        ctx = _build_ctx(
            tmp_path, fs,
            [
                SeedPaper(paper_id="DIRECT-1"),
                SeedPaper(title="Some Paper"),
                SeedPaper(paper_id="DIRECT-2"),
            ],
        )
        ResolveSeeds().run([], ctx)
        assert ctx.resolved_seed_ids == ["DIRECT-1", "FROM-TITLE", "DIRECT-2"]


# ---------------------------------------------------------------------------
# ResolveSeeds — preprint + published sibling expansion
# ---------------------------------------------------------------------------


class TestResolveSeedsPreprintPublishedPair:
    """Headline test: a title that resolves to two distinct paper_ids
    (preprint + published) when ``include_siblings=True``."""

    def _build_pair_corpus(self) -> FakeS2Client:
        fs = FakeS2Client()
        # The published version (canonical S2 paperId = PUBLISHED).
        # Carries an ArXiv external_id that points at the preprint.
        published = make_paper(
            "PUBLISHED",
            title="Highly Accurate Protein Structure Prediction",
            year=2021,
            external_ids={
                "DOI": "10.1038/s41586-021-03819-2",
                "ArXiv": "2107.00001",
            },
        )
        # The preprint (separate S2 paperId = PREPRINT). Same title +
        # the same ArXiv id but lives at a different paperId in S2.
        preprint = make_paper(
            "PREPRINT",
            title="Highly Accurate Protein Structure Prediction",
            year=2021,
            external_ids={"ArXiv": "2107.00001"},
        )
        fs.add(published)
        fs.add(preprint)

        # Title match resolves to the published version.
        fs.register_search_match(
            "Highly Accurate Protein Structure Prediction",
            published,
        )

        # DOI lookup returns the published version (same paperId).
        _add_paper_under(fs, "DOI:10.1038/s41586-021-03819-2", paper=published)
        # ArXiv lookup returns a DIFFERENT paperId (the preprint).
        # This is the preprint↔published divergence the step targets.
        _add_paper_under(fs, "ARXIV:2107.00001", paper=preprint)
        return fs

    def test_include_siblings_pulls_in_preprint(self, tmp_path: Path):
        fs = self._build_pair_corpus()
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(title="Highly Accurate Protein Structure Prediction")],
        )
        result = ResolveSeeds(include_siblings=True).run([], ctx)
        # Both paper IDs end up in the resolved list.
        assert "PUBLISHED" in ctx.resolved_seed_ids
        assert "PREPRINT" in ctx.resolved_seed_ids
        assert ctx.resolved_seed_ids[0] == "PUBLISHED"  # primary first
        assert result.stats["primaries_resolved"] == 1
        assert result.stats["siblings_added"] == 1

    def test_include_siblings_off_returns_only_primary(self, tmp_path: Path):
        fs = self._build_pair_corpus()
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(title="Highly Accurate Protein Structure Prediction")],
        )
        ResolveSeeds(include_siblings=False).run([], ctx)
        assert ctx.resolved_seed_ids == ["PUBLISHED"]

    def test_include_siblings_dedups_when_external_id_resolves_to_self(
        self, tmp_path: Path,
    ):
        """If the DOI lookup happens to return the same paper_id as the
        primary (the common case), no sibling is added — the
        deduplication keeps the resolved list clean."""
        fs = FakeS2Client()
        primary = make_paper(
            "P1",
            title="A Paper",
            external_ids={"DOI": "10.1/abc"},
        )
        fs.add(primary)
        _add_paper_under(fs, "DOI:10.1/abc", paper=primary)
        ctx = _build_ctx(
            tmp_path, fs, [SeedPaper(paper_id="P1")],
        )
        result = ResolveSeeds(include_siblings=True).run([], ctx)
        assert ctx.resolved_seed_ids == ["P1"]
        assert result.stats["siblings_added"] == 0


# ---------------------------------------------------------------------------
# Wiring with LoadSeeds: when ResolveSeeds runs first, LoadSeeds reads
# its output instead of cfg.seed_papers.
# ---------------------------------------------------------------------------


class TestResolveSeedsLoadSeedsHandoff:
    def test_load_seeds_consumes_resolved_seed_ids(self, tmp_path: Path):
        fs = FakeS2Client()
        # Two papers in the corpus
        fs.add(make_paper("RESOLVED-1", title="Paper One"))
        fs.add(make_paper("RESOLVED-2", title="Paper Two"))
        # Title match resolves to RESOLVED-1
        fs.register_search_match("Paper One", fs._papers["RESOLVED-1"])

        ctx = _build_ctx(
            tmp_path, fs,
            [
                SeedPaper(title="Paper One"),
                SeedPaper(paper_id="RESOLVED-2"),
            ],
        )
        # Build steps via the registry to confirm wiring works.
        resolve = build_step({"step": "ResolveSeeds"}, blocks={})
        load = build_step({"step": "LoadSeeds"}, blocks={})

        resolve.run([], ctx)
        assert ctx.resolved_seed_ids == ["RESOLVED-1", "RESOLVED-2"]

        load_result = load.run([], ctx)
        loaded_ids = sorted(p.paper_id for p in load_result.signal)
        assert loaded_ids == ["RESOLVED-1", "RESOLVED-2"]
        # ctx.collection is populated; both seeds are stamped source="seed"
        assert all(
            ctx.collection[pid].source == "seed"
            for pid in loaded_ids
        )

    def test_load_seeds_falls_back_when_no_resolved_seed_ids(
        self, tmp_path: Path,
    ):
        """Legacy single-step pipelines (no ResolveSeeds) keep working:
        LoadSeeds reads ``cfg.seed_papers`` directly."""
        fs = FakeS2Client()
        fs.add(make_paper("LEGACY-1", title="Legacy Paper"))
        ctx = _build_ctx(
            tmp_path, fs,
            [SeedPaper(paper_id="LEGACY-1")],
        )
        # Don't run ResolveSeeds.
        assert ctx.resolved_seed_ids == []
        load = build_step({"step": "LoadSeeds"}, blocks={})
        result = load.run([], ctx)
        assert [p.paper_id for p in result.signal] == ["LEGACY-1"]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestResolveSeedsRegistry:
    def test_in_step_registry(self):
        from citeclaw.steps import STEP_REGISTRY
        assert "ResolveSeeds" in STEP_REGISTRY

    def test_build_step_with_include_siblings_kwarg(self):
        step = build_step(
            {"step": "ResolveSeeds", "include_siblings": True},
            blocks={},
        )
        assert isinstance(step, ResolveSeeds)
        assert step.include_siblings is True

    def test_build_step_default_include_siblings_false(self):
        step = build_step({"step": "ResolveSeeds"}, blocks={})
        assert isinstance(step, ResolveSeeds)
        assert step.include_siblings is False
