"""Tests for :func:`citeclaw.filters.runner.apply_block` — the batched walker."""

from __future__ import annotations

from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.predicates import VenueIn
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route, RouteCase
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord


class _Remember:
    """Test helper: remember which papers it saw, and optionally fail a subset."""

    def __init__(self, name: str, *, reject_ids: set[str] | None = None) -> None:
        self.name = name
        self.reject_ids = reject_ids or set()
        self.seen: list[str] = []

    def check(self, paper, fctx):
        self.seen.append(paper.paper_id)
        if paper.paper_id in self.reject_ids:
            return FilterOutcome(False, f"reject {paper.paper_id}", self.name)
        return PASS


def _fctx(ctx):
    return FilterContext(ctx=ctx)


def _papers(*ids, **overrides):
    return [PaperRecord(paper_id=pid, **overrides) for pid in ids]


# ---------------------------------------------------------------------------
# Leaf behavior
# ---------------------------------------------------------------------------


class TestLeafBlocks:
    def test_empty_signal(self, ctx):
        r = _Remember("r")
        passed, rejected = apply_block([], r, _fctx(ctx))
        assert passed == []
        assert rejected == []
        assert r.seen == []

    def test_per_paper_check_atom(self, ctx):
        atom = YearFilter(min=2020)
        ps = _papers("a", "b", "c")
        ps[0].year, ps[1].year, ps[2].year = 2019, 2021, 2022
        passed, rejected = apply_block(ps, atom, _fctx(ctx))
        assert {p.paper_id for p in passed} == {"b", "c"}
        assert len(rejected) == 1
        assert rejected[0][0].paper_id == "a"


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


class TestSequential:
    def test_and_short_circuits_when_empty(self, ctx):
        a = _Remember("a", reject_ids={"p1", "p2"})
        b = _Remember("b")
        seq = Sequential(layers=[a, b])
        passed, rejected = apply_block(_papers("p1", "p2"), seq, _fctx(ctx))
        assert passed == []
        assert len(rejected) == 2
        assert b.seen == []  # short-circuited because `a` dropped everything

    def test_and_propagates_survivors(self, ctx):
        a = _Remember("a", reject_ids={"p1"})
        b = _Remember("b", reject_ids={"p3"})
        seq = Sequential(layers=[a, b])
        passed, rejected = apply_block(_papers("p1", "p2", "p3"), seq, _fctx(ctx))
        assert {p.paper_id for p in passed} == {"p2"}
        assert len(rejected) == 2
        assert b.seen == ["p2", "p3"]  # only survivors reach b


# ---------------------------------------------------------------------------
# Any_
# ---------------------------------------------------------------------------


class TestAny:
    def test_or_accepts_on_first_pass(self, ctx):
        a = _Remember("a", reject_ids={"p1"})
        b = _Remember("b")
        any_ = Any_(layers=[a, b])
        passed, rejected = apply_block(_papers("p1", "p2"), any_, _fctx(ctx))
        assert {p.paper_id for p in passed} == {"p1", "p2"}
        assert rejected == []
        # p1 got dropped by a, was re-tried in b; p2 passed a and never went to b.
        assert b.seen == ["p1"]

    def test_or_all_reject(self, ctx):
        a = _Remember("a", reject_ids={"p1", "p2"})
        b = _Remember("b", reject_ids={"p1", "p2"})
        any_ = Any_(layers=[a, b])
        passed, rejected = apply_block(_papers("p1", "p2"), any_, _fctx(ctx))
        assert passed == []
        assert len(rejected) == 2


# ---------------------------------------------------------------------------
# Not_
# ---------------------------------------------------------------------------


class TestNot:
    def test_invert_preserves_input_papers(self, ctx):
        inner = _Remember("inner", reject_ids={"p2"})  # rejects p2, passes p1, p3
        not_ = Not_(layer=inner)
        passed, rejected = apply_block(_papers("p1", "p2", "p3"), not_, _fctx(ctx))
        # After inversion, p2 is accepted; p1 and p3 are rejected.
        assert {p.paper_id for p in passed} == {"p2"}
        rejected_ids = {p.paper_id for p, _ in rejected}
        assert rejected_ids == {"p1", "p3"}


# ---------------------------------------------------------------------------
# Route partitioning
# ---------------------------------------------------------------------------


class TestRoute:
    def test_partitioned_dispatch(self, ctx):
        arxiv_target = _Remember("arxiv_t")
        default_target = _Remember("default_t")
        route = Route(cases=[
            RouteCase(predicate=VenueIn(values=["arXiv"]), target=arxiv_target),
            RouteCase(predicate=None, target=default_target, is_default=True),
        ])
        ps = [
            PaperRecord(paper_id="p1", venue="arXiv"),
            PaperRecord(paper_id="p2", venue="Nature"),
            PaperRecord(paper_id="p3", venue="arXiv preprint"),
            PaperRecord(paper_id="p4", venue="Cell"),
        ]
        passed, rejected = apply_block(ps, route, _fctx(ctx))
        assert rejected == []
        # Each target should only see its partition.
        assert sorted(arxiv_target.seen) == ["p1", "p3"]
        assert sorted(default_target.seen) == ["p2", "p4"]
        assert len(passed) == 4

    def test_no_match_rejects(self, ctx):
        """A Route with no matching case and no default drops the paper."""
        target = _Remember("t")
        route = Route(cases=[
            RouteCase(predicate=VenueIn(values=["arXiv"]), target=target),
        ])
        ps = [PaperRecord(paper_id="p1", venue="Nature")]
        passed, rejected = apply_block(ps, route, _fctx(ctx))
        assert passed == []
        assert len(rejected) == 1
        assert rejected[0][1].category == "no_route_match"


# ---------------------------------------------------------------------------
# Nested composition
# ---------------------------------------------------------------------------


class TestNestedComposition:
    def test_sequential_of_not_and_any(self, ctx):
        # Nested tree: Sequential [ Not(reject p1), Any(reject p2, reject p3) ]
        inner_not = _Remember("inner_not", reject_ids={"p2", "p3"})
        any_a = _Remember("any_a", reject_ids={"p2"})
        any_b = _Remember("any_b")
        tree = Sequential(layers=[
            Not_(layer=inner_not),  # passes p2 and p3, rejects p1
            Any_(layers=[any_a, any_b]),  # then accepts both p2 and p3
        ])
        passed, rejected = apply_block(_papers("p1", "p2", "p3"), tree, _fctx(ctx))
        assert {p.paper_id for p in passed} == {"p2", "p3"}
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# LLMFilter dispatch (via stub)
# ---------------------------------------------------------------------------


class TestLLMDispatch:
    def test_llm_filter_in_tree(self, ctx):
        """An LLMFilter wrapped inside a Sequential should still get
        dispatched through the stub client path."""
        llm = LLMFilter(scope="title", prompt="is relevant")
        seq = Sequential(layers=[llm])
        ps = _papers("p1", "p2")
        passed, rejected = apply_block(ps, seq, _fctx(ctx))
        # Stub always says match=true, so everything passes.
        assert {p.paper_id for p in passed} == {"p1", "p2"}
        assert rejected == []


# ---------------------------------------------------------------------------
# record_rejections bookkeeping
# ---------------------------------------------------------------------------


class TestRecordRejections:
    def test_updates_rejected_and_counts(self, ctx):
        ps = _papers("p1", "p2")
        outcomes = [
            (ps[0], FilterOutcome(False, "year", "year")),
            (ps[1], FilterOutcome(False, "citation", "citation")),
        ]
        record_rejections(outcomes, _fctx(ctx))
        assert "p1" in ctx.rejected
        assert "p2" in ctx.rejected
        assert ctx.rejection_counts["year"] == 1
        assert ctx.rejection_counts["citation"] == 1

    def test_unknown_category_bucket(self, ctx):
        ps = _papers("p1")
        record_rejections(
            [(ps[0], FilterOutcome(False, "why", ""))],
            _fctx(ctx),
        )
        assert ctx.rejection_counts["unknown"] == 1
