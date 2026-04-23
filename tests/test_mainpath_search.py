"""Tests for the five main-path search variants.

Uses the same Hummon & Doreian 1989 Figure 3 subgraph as
:mod:`test_mainpath_weights`. With SPC weights computed as:

    3→5=3, 3→21=1, 5→12=3, 12→15=1, 12→20=2,
    15→22=1, 20→21=1, 20→22=1, 21→22=2.

Greedy priority-first from node 3:
    3 (max out = 3→5) → 5 (only out 5→12) → 12 (max out = 12→20=2)
    → 20 (tied out 20→21=1, 20→22=1 — follow both)
    → 21 → 22 (for the 20→21 branch), and 22 directly (for 20→22).

So the local-forward main path is:
    {3→5, 5→12, 12→20, 20→21, 20→22, 21→22}  (6 edges, 6 nodes)

omitting {3→21, 12→15, 15→22} because of the greedy picks at vertices 3
and 12.
"""

from __future__ import annotations

import igraph as ig

from citeclaw.mainpath.search import (
    SEARCH_REGISTRY,
    global_cpm,
    key_route,
    local_backward,
    local_forward,
    multi_local,
)
from citeclaw.mainpath.weights import compute_spc


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


def _hd_figure3():
    g = ig.Graph(directed=True)
    g.add_vertices(_LABELS)
    g.add_edges(_EDGES)
    return g


def _edges_by_label(g: ig.Graph, edge_set: set[int]) -> set[tuple[str, str]]:
    return {
        (g.vs[g.es[e].source]["name"], g.vs[g.es[e].target]["name"])
        for e in edge_set
    }


class TestLocalForward:
    def test_spc_main_path(self):
        g = _hd_figure3()
        w = compute_spc(g)
        on_path = local_forward(g, w)
        labels = _edges_by_label(g, on_path)
        assert labels == {
            ("3", "5"),
            ("5", "12"),
            ("12", "20"),
            ("20", "21"),
            ("20", "22"),
            ("21", "22"),
        }

    def test_starts_from_every_source(self):
        """Two disconnected source chains both appear in the output."""
        g = ig.Graph(directed=True)
        g.add_vertices(["a", "b", "c", "d"])
        g.add_edges([("a", "b"), ("c", "d")])
        w = [1, 1]
        on_path = local_forward(g, w)
        assert len(on_path) == 2


class TestLocalBackward:
    def test_spc_main_path(self):
        """Backward from 22 should pick 21→22 (w=2) > 15→22, 20→22 (w=1)."""
        g = _hd_figure3()
        w = compute_spc(g)
        on_path = local_backward(g, w)
        labels = _edges_by_label(g, on_path)
        # Backward from 22: pick 21→22 (w=2 beats 15→22=1, 20→22=1).
        # From 21: pick 3→21 (w=1) AND 20→21 (w=1) — tied.
        # From 3: source, stop. From 20: pick 12→20 (w=2). From 12: pick 5→12.
        # From 5: pick 3→5.
        assert labels == {
            ("21", "22"),
            ("3", "21"),
            ("20", "21"),
            ("12", "20"),
            ("5", "12"),
            ("3", "5"),
        }


class TestGlobalCPM:
    def test_returns_max_weighted_path(self):
        """Critical path on SPC: find max-sum-weight path from 3 to 22.

        Paths and their SPC sums:
            3-5-12-15-22:       3 + 3 + 1 + 1 = 8
            3-5-12-20-22:       3 + 3 + 2 + 1 = 9
            3-5-12-20-21-22:    3 + 3 + 2 + 1 + 2 = 11  ← critical
            3-21-22:            1 + 2 = 3.
        """
        g = _hd_figure3()
        w = compute_spc(g)
        on_path = global_cpm(g, w)
        labels = _edges_by_label(g, on_path)
        assert labels == {
            ("3", "5"),
            ("5", "12"),
            ("12", "20"),
            ("20", "21"),
            ("21", "22"),
        }


class TestKeyRoute:
    def test_k1_seeds_from_top_arc_by_spc(self):
        """Top SPC arc is tied between 3→5 and 5→12 (both w=3).

        Stable sort on edge id keeps them in insertion order, so
        3→5 is picked as the first key-route. Forward from 5:
        5→12→20→{21,22}→22. Backward from 3: empty (source).
        """
        g = _hd_figure3()
        w = compute_spc(g)
        on_path = key_route(g, w, k=1)
        labels = _edges_by_label(g, on_path)
        assert labels == {
            ("3", "5"),
            ("5", "12"),
            ("12", "20"),
            ("20", "21"),
            ("20", "22"),
            ("21", "22"),
        }

    def test_k0_returns_empty(self):
        g = _hd_figure3()
        w = compute_spc(g)
        assert key_route(g, w, k=0) == set()

    def test_k_exceeds_edges_truncates(self):
        g = _hd_figure3()
        w = compute_spc(g)
        out = key_route(g, w, k=999)
        # All positive-weight edges get seeded, so output includes them all.
        assert len(out) == g.ecount()

    def test_guarantees_top_arc_on_path(self):
        """Core property of key-route per Liu & Lu 2012: top arc in output."""
        g = _hd_figure3()
        w = compute_spc(g)
        top_e = max(range(g.ecount()), key=lambda e: w[e])
        on_path = key_route(g, w, k=1)
        assert top_e in on_path


class TestMultiLocal:
    def test_zero_tolerance_matches_local_forward(self):
        g = _hd_figure3()
        w = compute_spc(g)
        assert multi_local(g, w, tolerance=0.0) == local_forward(g, w)

    def test_high_tolerance_includes_more_edges(self):
        """With tolerance=1.0, every positive-weight edge from each
        reachable vertex is included."""
        g = _hd_figure3()
        w = compute_spc(g)
        out = multi_local(g, w, tolerance=1.0)
        # All 9 edges have positive weight.
        assert len(out) == 9

    def test_fifty_percent_tolerance_behaviour(self):
        """At vertex 12 with outs 12→15 (w=1) and 12→20 (w=2), tolerance=0.5
        means threshold = 2 * 0.5 = 1, so BOTH edges qualify."""
        g = _hd_figure3()
        w = compute_spc(g)
        out = multi_local(g, w, tolerance=0.5)
        labels = _edges_by_label(g, out)
        # 12→15 should now be included alongside 12→20.
        assert ("12", "15") in labels
        assert ("12", "20") in labels


class TestSearchRegistry:
    def test_registry_has_all_five(self):
        assert set(SEARCH_REGISTRY) == {
            "local-forward",
            "local-backward",
            "global",
            "key-route",
            "multi-local",
        }

    def test_registry_callables_accept_kwargs(self):
        """Each search function accepts k=, tolerance= kwargs even if it
        doesn't use them, so the runner can call them uniformly."""
        g = _hd_figure3()
        w = compute_spc(g)
        for fn in SEARCH_REGISTRY.values():
            out = fn(g, w, k=1, tolerance=0.2)
            assert isinstance(out, set)
