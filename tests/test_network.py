"""Tests for :mod:`citeclaw.network` — citation graph, PageRank, saturation."""

from __future__ import annotations

import math

from citeclaw.models import PaperRecord
from citeclaw.network import build_citation_graph, compute_pagerank, compute_saturation


def _paper(pid, refs=None, sup=None, source="backward", year=2022, cit=0):
    return PaperRecord(
        paper_id=pid,
        title=pid,
        year=year,
        citation_count=cit,
        references=refs or [],
        supporting_papers=sup or [],
        source=source,
    )


class TestBuildCitationGraph:
    def test_empty_collection(self):
        g = build_citation_graph({})
        assert g.vcount() == 0
        assert g.ecount() == 0

    def test_basic_chain(self):
        coll = {
            "A": _paper("A", refs=[]),
            "B": _paper("B", refs=["A"]),
            "C": _paper("C", refs=["A", "B"]),
        }
        g = build_citation_graph(coll)
        assert g.vcount() == 3
        assert g.ecount() == 3  # A→B, A→C, B→C
        # Attributes
        assert set(g.vs["paper_id"]) == {"A", "B", "C"}

    def test_edges_only_within_collection(self):
        coll = {
            "A": _paper("A", refs=["NOT_IN_COLL"]),
            "B": _paper("B", refs=["A", "ALSO_OUT"]),
        }
        g = build_citation_graph(coll)
        assert g.ecount() == 1  # only A→B

    def test_self_loop_dropped(self):
        coll = {"A": _paper("A", refs=["A"])}
        g = build_citation_graph(coll)
        assert g.ecount() == 0

    def test_forward_supporting_paper_edge(self):
        """A paper with source=forward and supporting_papers=[X] means X was
        the source and this paper cites X. Edge should be X→self."""
        coll = {
            "SEED": _paper("SEED", refs=[]),
            "FWD": _paper("FWD", sup=["SEED"], source="forward"),
        }
        g = build_citation_graph(coll)
        assert g.ecount() == 1
        e = g.es[0]
        assert g.vs[e.source]["paper_id"] == "SEED"
        assert g.vs[e.target]["paper_id"] == "FWD"

    def test_backward_supporting_paper_edge(self):
        """A paper with source=backward and supporting_papers=[X] means it
        was pulled from X's references. Edge should be self→X (this paper
        is cited *by* X)."""
        coll = {
            "SEED": _paper("SEED", refs=[]),
            "BWD": _paper("BWD", sup=["SEED"], source="backward"),
        }
        g = build_citation_graph(coll)
        assert g.ecount() == 1
        e = g.es[0]
        assert g.vs[e.source]["paper_id"] == "BWD"
        assert g.vs[e.target]["paper_id"] == "SEED"


class TestComputePagerank:
    def test_empty(self):
        import igraph as ig

        g = ig.Graph(n=0, directed=True)
        assert compute_pagerank(g) == []

    def test_basic_ranking(self):
        coll = {
            "A": _paper("A", refs=[]),
            "B": _paper("B", refs=["A"]),
            "C": _paper("C", refs=["A", "B"]),
        }
        g = build_citation_graph(coll)
        ranked = compute_pagerank(g)
        assert len(ranked) == 3
        # Sorted descending
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
        # All should be positive
        assert all(s > 0 for s in scores)

    def test_personalized_with_seed(self):
        coll = {
            "A": _paper("A"),
            "B": _paper("B", refs=["A"]),
        }
        g = build_citation_graph(coll)
        ranked = compute_pagerank(g, seed_ids={"A"})
        assert {pid for pid, _ in ranked} == {"A", "B"}


class TestComputeSaturation:
    def test_all_already_in(self):
        assert compute_saturation(10, 0) == 1.0

    def test_none_in(self):
        assert compute_saturation(0, 10) == 0.0

    def test_partial(self):
        assert compute_saturation(3, 7) == 0.3

    def test_no_valid_references_is_nan(self):
        v = compute_saturation(0, 0)
        assert math.isnan(v)
