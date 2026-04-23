"""Tests for SPC, SPLC, SPNP weight computation.

The reference graph is the 7-node subgraph from Hummon & Doreian
(1989, Figure 3) — the same one the original paper and Batagelj
(2003) use to illustrate traversal counts. Nodes are the DNA
milestones {3, 5, 12, 15, 20, 21, 22} and the edges trace every path
from node 3 (the root) to node 22.

SPLC values are pinned against the published numbers in Hummon &
Doreian (1989, p. 50). SPC and SPNP are pinned against hand-derived
values following Batagelj (2003, §3) — Batagelj explicitly notes
that the original paper's SPNP table has arithmetic errors, so we
use the algorithmic definition, not the printed table.
"""

from __future__ import annotations

import igraph as ig
import pytest

from citeclaw.mainpath.weights import (
    WEIGHT_REGISTRY,
    compute_spc,
    compute_splc,
    compute_spnp,
)


# Hummon & Doreian 1989 Figure 3 subgraph.
# Index mapping: 0=3, 1=5, 2=12, 3=15, 4=20, 5=21, 6=22.
_LABELS = ["3", "5", "12", "15", "20", "21", "22"]
_EDGES = [
    ("3", "5"),
    ("3", "21"),
    ("5", "12"),
    ("12", "15"),
    ("12", "20"),
    ("15", "22"),
    ("20", "21"),
    ("20", "22"),
    ("21", "22"),
]


def _hd_figure3() -> tuple[ig.Graph, dict[tuple[str, str], int]]:
    """Build the reference graph and return a label-edge → index lookup."""
    g = ig.Graph(directed=True)
    g.add_vertices(_LABELS)
    g.add_edges(_EDGES)
    edge_idx = {
        (_LABELS[e.source], _LABELS[e.target]): e.index for e in g.es
    }
    return g, edge_idx


class TestSPC:
    """SPC = number of distinct source→sink paths through each arc.

    Source = {3} (only in-degree-0), Sink = {22} (only out-degree-0).
    Total number of 3→22 paths is 4:
        3-5-12-15-22, 3-5-12-20-22, 3-5-12-20-21-22, 3-21-22.
    """

    def test_values(self):
        g, idx = _hd_figure3()
        w = compute_spc(g)
        expected = {
            ("3", "5"): 3,
            ("3", "21"): 1,
            ("5", "12"): 3,
            ("12", "15"): 1,
            ("12", "20"): 2,
            ("15", "22"): 1,
            ("20", "21"): 1,
            ("20", "22"): 1,
            ("21", "22"): 2,
        }
        for edge_key, expected_w in expected.items():
            assert w[idx[edge_key]] == expected_w, (
                f"SPC({edge_key}) = {w[idx[edge_key]]}, expected {expected_w}"
            )

    def test_kirchhoff_node_law(self):
        """Inflow == outflow at every intermediate vertex (Batagelj 2003, §4.3)."""
        g, _ = _hd_figure3()
        w = compute_spc(g)
        intermediates = [
            v for v in range(g.vcount())
            if g.indegree(v) > 0 and g.outdegree(v) > 0
        ]
        assert intermediates, "sanity: test graph should have intermediates"
        for v in intermediates:
            inflow = sum(w[e] for e in g.incident(v, mode="in"))
            outflow = sum(w[e] for e in g.incident(v, mode="out"))
            assert inflow == outflow, (
                f"Kirchhoff violation at vertex {v}: "
                f"in={inflow}, out={outflow}"
            )

    def test_total_flow_equals_source_to_sink_paths(self):
        """Total source-emitting flow = total sink-receiving flow = # s-t paths."""
        g, _ = _hd_figure3()
        w = compute_spc(g)
        total_out_of_sources = sum(
            w[e]
            for v in range(g.vcount()) if g.indegree(v) == 0
            for e in g.incident(v, mode="out")
        )
        total_into_sinks = sum(
            w[e]
            for v in range(g.vcount()) if g.outdegree(v) == 0
            for e in g.incident(v, mode="in")
        )
        assert total_out_of_sources == total_into_sinks == 4


class TestSPLC:
    """SPLC pinned against Hummon & Doreian (1989, p. 50)."""

    def test_values_match_hd1989_table(self):
        g, idx = _hd_figure3()
        w = compute_splc(g)
        # Direct from the paper.
        expected = {
            ("3", "5"): 3,
            ("3", "21"): 1,
            ("5", "12"): 6,
            ("12", "15"): 3,
            ("12", "20"): 6,
            ("15", "22"): 4,
            ("20", "21"): 4,
            ("20", "22"): 4,
            ("21", "22"): 6,
        }
        for edge_key, expected_w in expected.items():
            assert w[idx[edge_key]] == expected_w, (
                f"SPLC({edge_key}) = {w[idx[edge_key]]}, expected {expected_w}"
            )


class TestSPNP:
    """SPNP from the algorithmic definition (Batagelj 2003, §3.3).

    Every vertex is both a possible origin and a possible destination.
    Values are hand-derived from the topological DP and do NOT match
    the 1989 paper's printed table, which Batagelj flags as containing
    errors.
    """

    def test_values(self):
        g, idx = _hd_figure3()
        w = compute_spnp(g)
        expected = {
            ("3", "5"): 8,
            ("3", "21"): 2,
            ("5", "12"): 14,
            ("12", "15"): 6,
            ("12", "20"): 12,
            ("15", "22"): 4,
            ("20", "21"): 8,
            ("20", "22"): 4,
            ("21", "22"): 6,
        }
        for edge_key, expected_w in expected.items():
            assert w[idx[edge_key]] == expected_w, (
                f"SPNP({edge_key}) = {w[idx[edge_key]]}, expected {expected_w}"
            )


class TestWeightOrdering:
    """Batagelj 2003 §4.1: SPC <= SPLC <= SPNP pointwise on every arc."""

    def test_pointwise_ordering(self):
        g, _ = _hd_figure3()
        spc = compute_spc(g)
        splc = compute_splc(g)
        spnp = compute_spnp(g)
        for e in range(g.ecount()):
            assert spc[e] <= splc[e] <= spnp[e], (
                f"Ordering violated at edge {e}: "
                f"spc={spc[e]}, splc={splc[e]}, spnp={spnp[e]}"
            )


class TestRegistry:
    def test_registry_has_all_three(self):
        assert set(WEIGHT_REGISTRY) == {"spc", "splc", "spnp"}

    def test_registry_callables_produce_same_values(self):
        g, _ = _hd_figure3()
        assert WEIGHT_REGISTRY["spc"](g) == compute_spc(g)
        assert WEIGHT_REGISTRY["splc"](g) == compute_splc(g)
        assert WEIGHT_REGISTRY["spnp"](g) == compute_spnp(g)


class TestEdgeCases:
    def test_single_node_graph(self):
        g = ig.Graph(n=1, directed=True)
        assert compute_spc(g) == []
        assert compute_splc(g) == []
        assert compute_spnp(g) == []

    def test_single_edge(self):
        g = ig.Graph(n=2, directed=True)
        g.add_edge(0, 1)
        # Only path through arc (0,1) in any of the three extended
        # networks is s → 0 → 1 → t — count 1 regardless of variant.
        assert compute_spc(g) == [1]
        assert compute_splc(g) == [1]
        assert compute_spnp(g) == [1]

    def test_raises_on_non_dag(self):
        g = ig.Graph(n=2, directed=True)
        g.add_edges([(0, 1), (1, 0)])
        with pytest.raises(Exception):
            compute_spc(g)

    def test_disconnected_components(self):
        """Two independent chains: weights are computed independently."""
        g = ig.Graph(n=4, directed=True)
        g.add_edges([(0, 1), (2, 3)])
        spc = compute_spc(g)
        # Each chain is its own source → sink path, weight 1 each.
        assert spc == [1, 1]
