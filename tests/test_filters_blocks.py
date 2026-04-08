"""Tests for composite filter blocks: Sequential, Any, Not, Route, SimilarityFilter."""

from __future__ import annotations

import pytest

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.predicates import VenueIn, YearAtLeast
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route, RouteCase
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.filters.blocks.similarity import SimilarityFilter
from citeclaw.models import PaperRecord


# ---------------------------------------------------------------------------
# Helper stub filters for testing block semantics independently of atoms
# ---------------------------------------------------------------------------


class _AlwaysPass:
    def __init__(self, name="pass") -> None:
        self.name = name
        self.calls = 0

    def check(self, paper, fctx):
        self.calls += 1
        return PASS


class _AlwaysReject:
    def __init__(self, name="reject") -> None:
        self.name = name
        self.calls = 0

    def check(self, paper, fctx):
        self.calls += 1
        return FilterOutcome(False, "no", "no_cat")


class _AlwaysReject2(_AlwaysReject):
    """Distinct type so Sequential/Any can have multiple reject layers."""


def _fctx(ctx):
    return FilterContext(ctx=ctx)


# ---------------------------------------------------------------------------
# Sequential (AND, short-circuit)
# ---------------------------------------------------------------------------


class TestSequential:
    def test_empty_sequential_passes(self, ctx):
        seq = Sequential(layers=[])
        assert seq.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed

    def test_all_pass(self, ctx):
        a, b = _AlwaysPass("a"), _AlwaysPass("b")
        seq = Sequential(layers=[a, b])
        assert seq.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed
        assert a.calls == 1 and b.calls == 1

    def test_short_circuit_on_reject(self, ctx):
        r, p = _AlwaysReject("r"), _AlwaysPass("p")
        seq = Sequential(layers=[r, p])
        out = seq.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed
        assert r.calls == 1 and p.calls == 0  # second layer never ran


# ---------------------------------------------------------------------------
# Any_ (OR, short-circuit)
# ---------------------------------------------------------------------------


class TestAny:
    def test_empty_is_reject(self, ctx):
        any_ = Any_(layers=[])
        out = any_.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed

    def test_first_pass_short_circuits(self, ctx):
        a, b = _AlwaysPass("a"), _AlwaysPass("b")
        any_ = Any_(layers=[a, b])
        any_.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert a.calls == 1 and b.calls == 0

    def test_all_reject(self, ctx):
        r1, r2 = _AlwaysReject("r1"), _AlwaysReject2("r2")
        any_ = Any_(layers=[r1, r2])
        out = any_.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed
        assert r1.calls == 1 and r2.calls == 1

    def test_second_pass_wins(self, ctx):
        r1, p = _AlwaysReject("r1"), _AlwaysPass("p")
        any_ = Any_(layers=[r1, p])
        assert any_.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# Not_
# ---------------------------------------------------------------------------


class TestNot:
    def test_invert_pass(self, ctx):
        n = Not_(layer=_AlwaysPass("inner"))
        out = n.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed
        assert "not(" in out.reason

    def test_invert_reject(self, ctx):
        n = Not_(layer=_AlwaysReject("inner"))
        assert n.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# Route (if/elif/else)
# ---------------------------------------------------------------------------


class TestRoute:
    def test_select_matching_predicate(self, ctx):
        target1 = _AlwaysPass("target1")
        target2 = _AlwaysReject("target2")
        route = Route(cases=[
            RouteCase(predicate=VenueIn(values=["arXiv"]), target=target1),
            RouteCase(predicate=None, target=target2, is_default=True),
        ])
        chosen = route.select(PaperRecord(paper_id="p", venue="arXiv"), _fctx(ctx))
        assert chosen is target1

    def test_select_default(self, ctx):
        target1 = _AlwaysPass("target1")
        default = _AlwaysPass("default")
        route = Route(cases=[
            RouteCase(predicate=VenueIn(values=["arXiv"]), target=target1),
            RouteCase(predicate=None, target=default, is_default=True),
        ])
        chosen = route.select(PaperRecord(paper_id="p", venue="Nature"), _fctx(ctx))
        assert chosen is default

    def test_no_match_returns_none(self, ctx):
        route = Route(cases=[
            RouteCase(predicate=VenueIn(values=["arXiv"]), target=_AlwaysPass("t")),
        ])
        assert route.select(PaperRecord(paper_id="p", venue="Cell"), _fctx(ctx)) is None

    def test_check_no_match_rejects(self, ctx):
        route = Route(cases=[
            RouteCase(predicate=YearAtLeast(n=2025), target=_AlwaysPass("t")),
        ])
        out = route.check(PaperRecord(paper_id="p", year=2020), _fctx(ctx))
        assert not out.passed
        assert out.category == "no_route_match"

    def test_check_delegates_to_target(self, ctx):
        target = _AlwaysPass("t")
        route = Route(cases=[RouteCase(predicate=None, target=target, is_default=True)])
        assert route.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed
        assert target.calls == 1


# ---------------------------------------------------------------------------
# SimilarityFilter
# ---------------------------------------------------------------------------


class _MeasureConst:
    def __init__(self, name, value):
        self.name = name
        self._value = value
        self.prefetch_called = False

    def compute(self, paper, fctx):
        return self._value

    def prefetch(self, papers, fctx):
        self.prefetch_called = True


class _MeasureNone(_MeasureConst):
    def __init__(self, name="none"):
        super().__init__(name, None)


class TestSimilarityFilter:
    def test_max_score_passes(self, ctx):
        f = SimilarityFilter(
            threshold=0.5,
            measures=[_MeasureConst("a", 0.3), _MeasureConst("b", 0.7)],
        )
        out = f.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert out.passed

    def test_below_threshold_rejects(self, ctx):
        f = SimilarityFilter(
            threshold=0.5,
            measures=[_MeasureConst("a", 0.2), _MeasureConst("b", 0.4)],
        )
        out = f.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed
        assert out.category == "similarity"
        # The rejected reason should name the winning measure.
        assert "b" in out.reason

    def test_none_measures_excluded_from_max(self, ctx):
        """None measures are skipped, so a single winning measure still decides."""
        f = SimilarityFilter(
            threshold=0.5,
            measures=[_MeasureNone(), _MeasureConst("b", 0.8)],
        )
        assert f.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed

    def test_all_none_defaults_to_pass(self, ctx):
        f = SimilarityFilter(threshold=0.5, measures=[_MeasureNone(), _MeasureNone("n2")])
        assert f.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed

    def test_all_none_on_no_data_reject(self, ctx):
        f = SimilarityFilter(
            threshold=0.5, measures=[_MeasureNone()], on_no_data="reject",
        )
        out = f.check(PaperRecord(paper_id="p"), _fctx(ctx))
        assert not out.passed
        assert out.category == "similarity_no_data"

    def test_invalid_on_no_data_raises(self):
        with pytest.raises(ValueError):
            SimilarityFilter(threshold=0.5, measures=[], on_no_data="maybe")

    def test_check_batch_calls_prefetch(self, ctx):
        m = _MeasureConst("a", 0.9)
        f = SimilarityFilter(threshold=0.5, measures=[m])
        outcomes = f.check_batch(
            [PaperRecord(paper_id="p1"), PaperRecord(paper_id="p2")], _fctx(ctx),
        )
        assert len(outcomes) == 2
        assert all(o.passed for o in outcomes)
        assert m.prefetch_called is True

    def test_check_batch_tolerates_prefetch_errors(self, ctx):
        class Broken(_MeasureConst):
            def prefetch(self, papers, fctx):
                raise RuntimeError("boom")

        f = SimilarityFilter(threshold=0.5, measures=[Broken("b", 0.9)])
        outcomes = f.check_batch([PaperRecord(paper_id="p")], _fctx(ctx))
        assert len(outcomes) == 1
        assert outcomes[0].passed  # Per-paper compute still runs.

    def test_empty_measures_no_data(self, ctx):
        f = SimilarityFilter(threshold=0.5, measures=[])
        assert f.check(PaperRecord(paper_id="p"), _fctx(ctx)).passed


# ---------------------------------------------------------------------------
# Smoke tests with real atoms
# ---------------------------------------------------------------------------


class TestBlocksWithRealAtoms:
    def test_sequential_of_year_and_citation(self, ctx):
        seq = Sequential(
            layers=[YearFilter(min=2018), CitationFilter(beta=5)],
        )
        good = PaperRecord(paper_id="p", year=2019, citation_count=10_000)
        bad_year = PaperRecord(paper_id="p", year=2000, citation_count=10_000)
        bad_cit = PaperRecord(paper_id="p", year=2019, citation_count=1)
        assert seq.check(good, _fctx(ctx)).passed
        assert not seq.check(bad_year, _fctx(ctx)).passed
        assert not seq.check(bad_cit, _fctx(ctx)).passed
