"""Tests for shrink_family and preprint_transform cycle policies."""

from __future__ import annotations

import igraph as ig

from citeclaw.mainpath.cycles import (
    CYCLE_REGISTRY,
    preprint_transform,
    shrink_family,
)


def _make_graph(paper_ids, years, edges, citation_counts=None):
    """Build a directed igraph with CiteClaw-style vertex attrs."""
    g = ig.Graph(directed=True)
    g.add_vertices(len(paper_ids))
    g.vs["paper_id"] = list(paper_ids)
    g.vs["year"] = list(years)
    if citation_counts is not None:
        g.vs["citation_count"] = list(citation_counts)
    label_to_idx = {p: i for i, p in enumerate(paper_ids)}
    g.add_edges([(label_to_idx[s], label_to_idx[t]) for s, t in edges])
    return g


# ---------------------------------------------------------------------------
# shrink_family
# ---------------------------------------------------------------------------


class TestShrinkFamily:
    def test_dag_passthrough(self):
        """A DAG passes through with no collapse, but as a fresh copy."""
        g = _make_graph(
            ["a", "b", "c"], [2010, 2015, 2020],
            [("a", "b"), ("b", "c")],
        )
        out, trace = shrink_family(g)
        assert out.vcount() == 3
        assert out.ecount() == 2
        assert out.is_dag()
        assert trace.policy == "shrink"
        assert trace.scc_sizes == []
        assert trace.supernode_members == {}
        assert trace.n_nodes_before == trace.n_nodes_after == 3

    def test_simple_two_cycle_collapses(self):
        """Classic 2-cycle A↔B collapses to the older representative."""
        g = _make_graph(
            ["a", "b"], [2010, 2015],
            [("a", "b"), ("b", "a")],
        )
        out, trace = shrink_family(g)
        assert out.vcount() == 1
        assert out.ecount() == 0
        assert out.is_dag()
        assert out.vs[0]["paper_id"] == "a"  # older year wins
        assert trace.scc_sizes == [2]
        assert trace.supernode_members == {"a": ["a", "b"]}

    def test_representative_uses_year_then_citation_count(self):
        """Older year wins; among same-year nodes, higher citation_count wins."""
        g = _make_graph(
            ["old", "newer", "newest"],
            [2005, 2015, 2015],
            [("old", "newer"), ("newer", "newest"), ("newest", "old")],
            citation_counts=[10, 50, 200],
        )
        out, trace = shrink_family(g)
        assert out.vcount() == 1
        assert out.vs[0]["paper_id"] == "old"
        assert trace.scc_sizes == [3]

    def test_representative_ties_between_same_year_papers(self):
        """Same year, different citation counts — higher count wins."""
        g = _make_graph(
            ["a", "b"],
            [2010, 2010],
            [("a", "b"), ("b", "a")],
            citation_counts=[5, 100],
        )
        out, _ = shrink_family(g)
        assert out.vcount() == 1
        assert out.vs[0]["paper_id"] == "b"  # higher citation count

    def test_inter_scc_edges_rewired_to_representatives(self):
        """External edges into/out of a cycle attach to the supernode."""
        g = _make_graph(
            ["pre", "a", "b", "post"],
            [2000, 2010, 2012, 2020],
            [
                ("pre", "a"),     # external → SCC member
                ("a", "b"),       # SCC internal
                ("b", "a"),       # SCC internal (closes cycle)
                ("b", "post"),    # SCC member → external
            ],
        )
        out, trace = shrink_family(g)
        assert out.vcount() == 3  # pre, {a,b}, post
        assert out.is_dag()
        pids = sorted(v["paper_id"] for v in out.vs)
        assert pids == ["a", "post", "pre"]
        # Check edges: pre→{a,b-rep=a} and a→post.
        edge_pairs = sorted(
            (out.vs[e.source]["paper_id"], out.vs[e.target]["paper_id"])
            for e in out.es
        )
        assert edge_pairs == [("a", "post"), ("pre", "a")]
        assert trace.scc_sizes == [2]
        assert trace.supernode_members == {"a": ["a", "b"]}

    def test_multiple_non_trivial_sccs(self):
        """Two independent SCCs each collapse separately."""
        g = _make_graph(
            ["a", "b", "c", "d"],
            [2010, 2012, 2015, 2018],
            [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")],
        )
        out, trace = shrink_family(g)
        assert out.vcount() == 2
        assert out.ecount() == 0
        assert sorted(trace.scc_sizes) == [2, 2]
        assert len(trace.supernode_members) == 2

    def test_duplicate_edges_deduplicated_on_collapse(self):
        """If two SCC members both point to the same external node, dedupe."""
        g = _make_graph(
            ["a", "b", "post"],
            [2010, 2012, 2020],
            [("a", "b"), ("b", "a"), ("a", "post"), ("b", "post")],
        )
        out, _ = shrink_family(g)
        assert out.vcount() == 2
        # Only ONE edge from the collapsed supernode to "post".
        assert out.ecount() == 1


# ---------------------------------------------------------------------------
# preprint_transform
# ---------------------------------------------------------------------------


class TestPreprintTransform:
    def test_dag_passthrough(self):
        g = _make_graph(
            ["a", "b", "c"], [2010, 2015, 2020],
            [("a", "b"), ("b", "c")],
        )
        out, trace = preprint_transform(g)
        assert out.vcount() == 3
        assert out.ecount() == 2
        assert out.is_dag()
        assert trace.policy == "preprint"
        assert trace.scc_sizes == []

    def test_simple_two_cycle_produces_dag(self):
        """A↔B cycle → 4 vertices (a, b, a', b'), DAG, no internal cycle."""
        g = _make_graph(
            ["a", "b"], [2010, 2015],
            [("a", "b"), ("b", "a")],
        )
        out, trace = preprint_transform(g)
        assert out.vcount() == 4
        assert out.is_dag()
        # Preprints get __preprint suffix in paper_id.
        preprints = [v["paper_id"] for v in out.vs if "__preprint" in v["paper_id"]]
        assert sorted(preprints) == ["a__preprint", "b__preprint"]
        assert trace.scc_sizes == [2]

    def test_simple_two_cycle_arc_count(self):
        """For a 2-cycle: 2 original arcs dropped, 4 preprinted arcs added."""
        g = _make_graph(
            ["a", "b"], [2010, 2015],
            [("a", "b"), ("b", "a")],
        )
        out, _ = preprint_transform(g)
        # Original 2 SCC arcs deleted; k² = 4 preprinted arcs added.
        assert out.ecount() == 4

    def test_external_out_edge_rewired_to_preprint(self):
        """Non-SCC out-arc (a, w) becomes (a', w)."""
        g = _make_graph(
            ["a", "b", "w"],
            [2010, 2015, 2020],
            [("a", "b"), ("b", "a"), ("a", "w")],  # w is external sink
        )
        out, _ = preprint_transform(g)
        # Find vertex ids by paper_id.
        pid = {v["paper_id"]: v.index for v in out.vs}
        # Edge from a__preprint to w must exist.
        eid = out.get_eid(pid["a__preprint"], pid["w"], error=False)
        assert eid != -1
        # And the direct (a, w) arc must NOT exist.
        eid_direct = out.get_eid(pid["a"], pid["w"], error=False)
        assert eid_direct == -1

    def test_external_in_edge_stays_on_original(self):
        """Non-SCC in-arc (w, a) stays on original a (not rewired)."""
        g = _make_graph(
            ["w", "a", "b"],
            [2005, 2010, 2015],
            [("w", "a"), ("a", "b"), ("b", "a")],
        )
        out, _ = preprint_transform(g)
        pid = {v["paper_id"]: v.index for v in out.vs}
        # w → a must still exist directly.
        assert out.get_eid(pid["w"], pid["a"], error=False) != -1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCycleRegistry:
    def test_registry_has_both(self):
        assert set(CYCLE_REGISTRY) == {"shrink", "preprint"}

    def test_registry_callables_produce_dag(self):
        g = _make_graph(
            ["a", "b"], [2010, 2015],
            [("a", "b"), ("b", "a")],
        )
        for policy in CYCLE_REGISTRY:
            out, _ = CYCLE_REGISTRY[policy](g)
            assert out.is_dag(), f"{policy} did not produce a DAG"
