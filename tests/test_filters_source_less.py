"""PC-05: pin the source-less FilterContext contract.

The ``ExpandBy*`` family of steps (``ExpandBySearch`` /
``ExpandBySemantics`` / ``ExpandByAuthor``) builds its
:class:`citeclaw.filters.base.FilterContext` with
``source=None`` / ``source_refs=None`` / ``source_citers=None`` because
those steps have no upstream citation edge to anchor on. Every filter
atom and similarity measure used in their screener cascades MUST
tolerate this case without crashing.

This file pins the contract per the PC-05 audit:

  * **Atoms** (year, citation, predicates, llm_query) — none of them
    actually read ``fctx.source*``; they only consult ``paper.*`` (or
    ``fctx.ctx`` for the LLM dispatcher). The audit confirmed this; the
    tests below assert it stays that way.
  * **Measures** (RefSim, CitSim, SemanticSim) — must return ``None``
    when their source-side data is missing rather than raising. The
    ``SimilarityFilter`` composer interprets ``None`` according to its
    ``on_no_data`` knob.
  * **End-to-end** — running ``apply_block`` on a Sequential cascade
    of an atom + a measure with a source-less FilterContext yields a
    valid (passed, rejected) split with no exceptions.

If any future change accidentally introduces an unguarded ``fctx.source``
access, one of the tests below will start raising ``AttributeError``.
"""

from __future__ import annotations

import pytest

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.predicates import CitAtLeast, VenueIn, YearAtLeast
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.base import FilterContext, FilterOutcome
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.filters.measures.cit_sim import CitSimMeasure
from citeclaw.filters.measures.ref_sim import RefSimMeasure
from citeclaw.filters.measures.semantic_sim import SemanticSimMeasure
from citeclaw.filters.blocks.similarity import SimilarityFilter
from citeclaw.filters.runner import apply_block
from citeclaw.models import PaperRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_less_fctx(ctx) -> FilterContext:
    """Build the FilterContext that the ExpandBy* family produces."""
    return FilterContext(
        ctx=ctx, source=None, source_refs=None, source_citers=None,
    )


def _paper(
    pid: str = "candidate",
    *,
    year: int | None = 2022,
    venue: str | None = "Nature",
    citation_count: int | None = 50,
    title: str = "Candidate Paper",
    abstract: str | None = "Candidate abstract.",
) -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        citation_count=citation_count,
    )


# ---------------------------------------------------------------------------
# Atoms — every check() must work with source=None
# ---------------------------------------------------------------------------


class TestAtomsTolerateSourceNone:
    def test_year_filter(self, ctx):
        f = YearFilter(min=2018, max=2025)
        out = f.check(_paper(year=2022), _source_less_fctx(ctx))
        assert isinstance(out, FilterOutcome)
        assert out.passed

    def test_year_filter_rejects_out_of_range(self, ctx):
        f = YearFilter(min=2020)
        out = f.check(_paper(year=2010), _source_less_fctx(ctx))
        assert not out.passed

    def test_citation_filter(self, ctx):
        f = CitationFilter(beta=5.0)
        out = f.check(_paper(citation_count=10_000), _source_less_fctx(ctx))
        assert isinstance(out, FilterOutcome)
        assert out.passed

    def test_citation_filter_rejects_low_cited(self, ctx):
        f = CitationFilter(beta=100.0)
        out = f.check(_paper(citation_count=5), _source_less_fctx(ctx))
        assert not out.passed

    def test_citation_filter_missing_data(self, ctx):
        f = CitationFilter(beta=5.0)
        out = f.check(_paper(citation_count=None), _source_less_fctx(ctx))
        assert not out.passed
        assert out.category == "missing_data"

    def test_venue_in(self, ctx):
        f = VenueIn(values=["Nature", "Science"])
        out = f.check(_paper(venue="Nature"), _source_less_fctx(ctx))
        assert out.passed

    def test_venue_in_rejects_unmatched(self, ctx):
        f = VenueIn(values=["arXiv"])
        out = f.check(_paper(venue="Nature"), _source_less_fctx(ctx))
        assert not out.passed

    def test_cit_at_least(self, ctx):
        f = CitAtLeast(n=10)
        out = f.check(_paper(citation_count=20), _source_less_fctx(ctx))
        assert out.passed

    def test_cit_at_least_rejects(self, ctx):
        f = CitAtLeast(n=100)
        out = f.check(_paper(citation_count=10), _source_less_fctx(ctx))
        assert not out.passed

    def test_year_at_least(self, ctx):
        f = YearAtLeast(n=2020)
        out = f.check(_paper(year=2022), _source_less_fctx(ctx))
        assert out.passed

    def test_year_at_least_rejects(self, ctx):
        f = YearAtLeast(n=2025)
        out = f.check(_paper(year=2020), _source_less_fctx(ctx))
        assert not out.passed

    def test_llm_filter_via_apply_block(self, ctx):
        """LLMFilter doesn't expose .check directly — it goes through
        apply_block / dispatch_batch. Drive a single-filter Sequential
        with the stub LLM client (configured in the conftest's `ctx`
        fixture via screening_model="stub")."""
        f = LLMFilter(scope="title", prompt="is relevant")
        block = Sequential(layers=[f])
        passed, rejected = apply_block(
            [_paper("p1", title="A title"), _paper("p2", title="Another")],
            block,
            _source_less_fctx(ctx),
        )
        # Stub returns match=True for everything; assert no crash + sane shape.
        assert len(passed) + len(rejected) == 2


# ---------------------------------------------------------------------------
# Measures — every compute() must return None when source-side data is
# missing, never raise.
# ---------------------------------------------------------------------------


class TestMeasuresTolerateSourceNone:
    def test_ref_sim_returns_none(self, ctx):
        m = RefSimMeasure()
        out = m.compute(_paper("CITER1"), _source_less_fctx(ctx))
        assert out is None

    def test_cit_sim_returns_none(self, ctx):
        m = CitSimMeasure()
        out = m.compute(_paper("CITER1"), _source_less_fctx(ctx))
        assert out is None

    def test_cit_sim_heavy_cite_shortcut_still_fires_without_source(self, ctx):
        """The ``pass_if_cited_at_least`` shortcut runs BEFORE the
        source-side check, so a heavily cited candidate still gets a
        max-similarity score even when source_citers is None."""
        m = CitSimMeasure(pass_if_cited_at_least=100)
        out = m.compute(
            _paper("CITER1", citation_count=500),
            _source_less_fctx(ctx),
        )
        assert out == 1.0

    def test_semantic_sim_returns_none_compute(self, ctx):
        m = SemanticSimMeasure(embedder="s2")
        out = m.compute(_paper("SEED"), _source_less_fctx(ctx))
        assert out is None

    def test_semantic_sim_prefetch_no_source(self, ctx):
        """Prefetch should warm up candidate embeddings without
        crashing when there's no source paper to add to the batch."""
        m = SemanticSimMeasure(embedder="s2")
        # Two real papers from the chain corpus the conftest builds.
        m.prefetch(
            [_paper("SEED"), _paper("CITER1")],
            _source_less_fctx(ctx),
        )
        # No exception = pass; the conftest's ctx.s2 is FakeS2Client so
        # this is a no-op call but it must not raise.

    def test_semantic_sim_external_backend_compute_returns_none(self, ctx):
        """Even the not-yet-wired external backends must short-circuit
        on source=None *before* attempting to call .embed() — otherwise
        they'd raise NotImplementedError."""
        m = SemanticSimMeasure(embedder="voyage:voyage-3")
        out = m.compute(_paper("SEED"), _source_less_fctx(ctx))
        assert out is None  # short-circuit kicked in; embed() never called


# ---------------------------------------------------------------------------
# SimilarityFilter — composes measures; must respect on_no_data when all
# measures return None under source-less mode.
# ---------------------------------------------------------------------------


class TestSimilarityFilterSourceLess:
    def test_on_no_data_pass_default(self, ctx):
        sf = SimilarityFilter(
            threshold=0.5,
            measures=[RefSimMeasure(), SemanticSimMeasure()],
            on_no_data="pass",
        )
        outcomes = sf.check_batch([_paper("p1"), _paper("p2")], _source_less_fctx(ctx))
        assert all(o.passed for o in outcomes)

    def test_on_no_data_reject(self, ctx):
        sf = SimilarityFilter(
            threshold=0.5,
            measures=[RefSimMeasure(), SemanticSimMeasure()],
            on_no_data="reject",
        )
        outcomes = sf.check_batch([_paper("p1"), _paper("p2")], _source_less_fctx(ctx))
        assert all(not o.passed for o in outcomes)

    def test_cit_sim_shortcut_within_similarity_filter(self, ctx):
        """Heavily cited candidates pass via CitSim's shortcut even
        when no source-side data exists."""
        sf = SimilarityFilter(
            threshold=0.5,
            measures=[CitSimMeasure(pass_if_cited_at_least=100)],
            on_no_data="reject",
        )
        outcomes = sf.check_batch(
            [_paper("p1", citation_count=500), _paper("p2", citation_count=10)],
            _source_less_fctx(ctx),
        )
        assert outcomes[0].passed   # 500 cites → CitSim returns 1.0 → ≥ 0.5
        assert not outcomes[1].passed   # 10 cites → None → on_no_data=reject


# ---------------------------------------------------------------------------
# End-to-end: a Sequential cascade of atoms + a similarity filter must
# survive an apply_block run with the source-less FilterContext used
# by the ExpandBy* family.
# ---------------------------------------------------------------------------


class TestApplyBlockSourceLess:
    def test_sequential_cascade_runs_without_crash(self, ctx):
        cascade = Sequential(
            layers=[
                YearFilter(min=2018, max=2025),
                CitationFilter(beta=1.0),
                SimilarityFilter(
                    threshold=0.5,
                    measures=[CitSimMeasure(pass_if_cited_at_least=100)],
                    on_no_data="pass",
                ),
            ],
        )
        papers = [
            _paper("p1", year=2020, citation_count=200),  # passes everything
            _paper("p2", year=2010, citation_count=200),  # year reject
            _paper("p3", year=2022, citation_count=0),    # citation reject
        ]
        passed, rejected = apply_block(papers, cascade, _source_less_fctx(ctx))
        passed_ids = {p.paper_id for p in passed}
        assert passed_ids == {"p1"}
        # p2 + p3 rejected, no exceptions raised.
        assert len(rejected) == 2
