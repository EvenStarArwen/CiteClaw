"""Tests for LeidenClusterer + the size-ranked relabelling it depends on.

Real igraph and real ``leidenalg`` throughout — no mocking. The headline
test is :class:`TestRun37Fixture`, which runs the clusterer's partition
function over the actual 354-node citation graph from the design
hand-off (``web/design-reference/uploads/run37/graph.json``) and checks
it against the ``community_leiden`` column the UI was designed on. That
file is read-only truth; nothing here writes to it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import igraph as ig
import pytest

from citeclaw.cluster import CLUSTERER_REGISTRY, build_clusterer
from citeclaw.cluster.agreement import adjusted_rand, normalized_mutual_info
from citeclaw.cluster.base import ClusterResult, relabel_by_size
from citeclaw.cluster.leiden import LeidenClusterer, leiden_partition
from citeclaw.context import Context
from citeclaw.models import PaperRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "web" / "design-reference" / "uploads" / "run37"


def _paper(pid: str, *, refs: list[str] | None = None) -> PaperRecord:
    return PaperRecord(paper_id=pid, references=refs or [])


def _three_cliques(ctx: Context) -> list[PaperRecord]:
    """Three disjoint cliques of 4, 3 and 2 papers — sizes 4 > 3 > 2."""
    groups = [
        ["A1", "A2", "A3", "A4"],
        ["B1", "B2", "B3"],
        ["C1", "C2"],
    ]
    papers: list[PaperRecord] = []
    for members in groups:
        for pid in members:
            papers.append(_paper(pid, refs=[m for m in members if m != pid]))
    for p in papers:
        ctx.collection[p.paper_id] = p
    return papers


# ---------------------------------------------------------------------------
# relabel_by_size
# ---------------------------------------------------------------------------


class TestRelabelBySize:
    def test_largest_cluster_becomes_zero(self):
        membership = {"a": 7, "b": 7, "c": 7, "d": 3, "e": 3, "f": 9}
        out = relabel_by_size(membership)
        assert out["a"] == out["b"] == out["c"] == 0
        assert out["d"] == out["e"] == 1
        assert out["f"] == 2

    def test_noise_is_passed_through_and_unranked(self):
        membership = {"a": -1, "b": -1, "c": -1, "d": 5, "e": 5}
        out = relabel_by_size(membership)
        assert out == {"a": -1, "b": -1, "c": -1, "d": 0, "e": 0}

    def test_ties_break_on_smallest_paper_id_not_dict_order(self):
        forward = {"z": 1, "a": 2}
        backward = {"a": 2, "z": 1}
        # Both clusters have size 1; "a" < "z" so cluster 2 must rank first
        # regardless of insertion order.
        assert relabel_by_size(forward) == relabel_by_size(backward) == {"a": 0, "z": 1}

    def test_empty_membership(self):
        assert relabel_by_size({}) == {}


# ---------------------------------------------------------------------------
# LeidenClusterer — small hand-built graphs
# ---------------------------------------------------------------------------


class TestLeidenClusterer:
    def test_registered_under_leiden(self):
        assert CLUSTERER_REGISTRY["leiden"] is LeidenClusterer
        clusterer = build_clusterer({"type": "leiden", "seed": 7})
        assert isinstance(clusterer, LeidenClusterer)
        assert clusterer.seed == 7

    def test_empty_collection_returns_empty_result(self, ctx: Context):
        result = LeidenClusterer().cluster([], ctx)
        assert isinstance(result, ClusterResult)
        assert result.membership == {}
        assert result.algorithm == "leiden"

    def test_disjoint_cliques_are_separated(self, ctx: Context):
        papers = _three_cliques(ctx)
        result = LeidenClusterer().cluster(papers, ctx)
        assert set(result.membership) == {p.paper_id for p in papers}
        for members in (("A1", "A2", "A3", "A4"), ("B1", "B2", "B3"), ("C1", "C2")):
            assert len({result.membership[m] for m in members}) == 1
        assert len(set(result.membership.values())) == 3

    def test_ids_are_ranked_by_descending_size(self, ctx: Context):
        papers = _three_cliques(ctx)
        result = LeidenClusterer().cluster(papers, ctx)
        assert result.membership["A1"] == 0   # the 4-clique
        assert result.membership["B1"] == 1   # the 3-clique
        assert result.membership["C1"] == 2   # the 2-clique

    def test_metadata_sizes_match_membership(self, ctx: Context):
        papers = _three_cliques(ctx)
        result = LeidenClusterer().cluster(papers, ctx)
        assert {cid: m.size for cid, m in result.metadata.items()} == {0: 4, 1: 3, 2: 2}

    def test_quality_reports_modularity_and_provenance(self, ctx: Context):
        papers = _three_cliques(ctx)
        result = LeidenClusterer().cluster(papers, ctx)
        # Three disjoint cliques is the textbook high-modularity case.
        assert result.quality["modularity"] > 0.4
        assert result.quality["n_communities"] == 3.0
        assert result.quality["seed"] == 42.0
        assert result.quality["backend_is_leidenalg"] == 1.0

    def test_deterministic_across_runs_with_same_seed(self, ctx: Context):
        papers = _three_cliques(ctx)
        first = LeidenClusterer(seed=11).cluster(papers, ctx)
        second = LeidenClusterer(seed=11).cluster(papers, ctx)
        assert first.membership == second.membership

    def test_membership_is_restricted_to_signal(self, ctx: Context):
        papers = _three_cliques(ctx)
        signal = [p for p in papers if p.paper_id.startswith("A")]
        result = LeidenClusterer().cluster(signal, ctx)
        assert set(result.membership) == {"A1", "A2", "A3", "A4"}
        # Ranked over the whole collection, so the A-clique keeps id 0
        # (it is the largest community in the corpus, not just the signal).
        assert set(result.membership.values()) == {0}

    def test_unknown_objective_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unknown Leiden objective"):
            LeidenClusterer(objective="modulaarity")

    def test_unknown_objective_rejected_by_partition_helper(self):
        g = ig.Graph.Famous("Zachary")
        with pytest.raises(ValueError, match="Unknown Leiden objective"):
            leiden_partition(g, objective="nope")

    def test_cpm_resolution_changes_granularity(self):
        """A high CPM resolution must fragment more than a low one."""
        g = ig.Graph.Famous("Zachary")
        coarse, _ = leiden_partition(g, objective="cpm", resolution=0.05, seed=42)
        fine, _ = leiden_partition(g, objective="cpm", resolution=0.9, seed=42)
        assert len(set(fine)) > len(set(coarse))

    def test_empty_graph_partition(self):
        membership, backend = leiden_partition(ig.Graph(n=0))
        assert membership == []
        assert backend == "leidenalg"

    def test_other_graph_clusterers_still_work(self, ctx: Context):
        """Guard rail: adding leiden must not disturb walktrap / louvain."""
        papers = _three_cliques(ctx)
        walktrap = build_clusterer({"type": "walktrap", "n_communities": 3})
        louvain = build_clusterer("louvain")
        for clusterer in (walktrap, louvain):
            result = clusterer.cluster(papers, ctx)
            assert set(result.membership) == {p.paper_id for p in papers}
            assert len({result.membership[m] for m in ("A1", "A2", "A3", "A4")}) == 1


# ---------------------------------------------------------------------------
# The run37 design fixture — 354 nodes, 1239 edges
# ---------------------------------------------------------------------------


def _load_fixture_graph() -> tuple[ig.Graph, list[str]]:
    """Build the run37 citation graph and return it with its paper ids.

    ``graph.json`` node ``id`` is the row index into ``accepted.csv``, so
    the vertex order here matches the CSV order and the two can be zipped.
    """
    data = json.loads((FIXTURE_DIR / "graph.json").read_text())
    nodes = data["nodes"]
    edges = [(e[0], e[1]) for e in data["edges"]]
    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    return g.as_undirected(mode="collapse"), [n["pid"] for n in nodes]


def _load_fixture_labels(column: str) -> list[int]:
    with (FIXTURE_DIR / "accepted.csv").open(encoding="utf-8") as fh:
        return [int(row[column]) for row in csv.DictReader(fh)]


pytestmark_fixture = pytest.mark.skipif(
    not (FIXTURE_DIR / "graph.json").exists(),
    reason="design-reference fixture not present in this checkout",
)


@pytestmark_fixture
class TestRun37Fixture:
    def test_graph_shape(self):
        g, pids = _load_fixture_graph()
        assert g.vcount() == 354
        assert len(pids) == len(set(pids)) == 354

    def test_reproduces_the_fixture_partition(self):
        """Seed 42 + modularity objective must land on the shipped labels.

        This is the evidence behind choosing ``leidenalg`` over igraph's
        built-in ``community_leiden`` (which scores ARI 0.53 here). If a
        future ``leidenalg`` release changes its RNG or tie-breaking this
        test fails loudly, which is the point.

        Note what is pinned: algorithm + seed + **vertex order**. Leiden's
        RNG trajectory depends on traversal order, and this test walks the
        graph in ``graph.json`` order. A production run goes through
        ``build_citation_graph``, which orders vertices by ``paper_id``,
        and therefore settles on a different (equally valid, in fact
        slightly higher-modularity) local optimum — see
        ``test_clusterer_end_to_end_over_the_fixture_corpus``, which
        asserts the properties that *do* hold there.
        """
        g, _ = _load_fixture_graph()
        membership, backend = leiden_partition(
            g, objective="modularity", seed=42, n_iterations=2,
        )
        assert backend == "leidenalg"
        expected = _load_fixture_labels("community_leiden")
        ranked = relabel_by_size(
            {str(i).zfill(4): c for i, c in enumerate(membership)}
        )
        actual = [ranked[str(i).zfill(4)] for i in range(len(membership))]
        assert normalized_mutual_info(actual, expected) == pytest.approx(1.0)
        assert adjusted_rand(actual, expected) == pytest.approx(1.0)
        assert actual == expected

    def test_partition_is_non_degenerate(self):
        """Not one giant blob, not 354 singletons, and modular."""
        g, _ = _load_fixture_graph()
        membership, _ = leiden_partition(g, seed=42)
        sizes = sorted(
            [membership.count(c) for c in set(membership)], reverse=True,
        )
        assert 2 <= len(sizes) <= 60
        assert sizes[0] < 0.5 * g.vcount()      # no dominant super-community
        assert sum(1 for s in sizes if s >= 5) >= 3   # several real groups
        assert g.modularity(membership) > 0.3

    def test_clusterer_end_to_end_over_the_fixture_corpus(self, ctx: Context):
        """Drive the real LeidenClusterer through a Context, not the helper."""
        data = json.loads((FIXTURE_DIR / "graph.json").read_text())
        pids = [n["pid"] for n in data["nodes"]]
        refs: dict[str, list[str]] = {pid: [] for pid in pids}
        # graph.json edge [a, b, w] means a cites b -> b is in a's references.
        for src, dst, *_ in data["edges"]:
            refs[pids[src]].append(pids[dst])
        papers = [_paper(pid, refs=refs[pid]) for pid in pids]
        for p in papers:
            ctx.collection[p.paper_id] = p

        result = LeidenClusterer(seed=42).cluster(papers, ctx)
        assert len(result.membership) == 354
        assert result.algorithm == "leiden"
        sizes = sorted(
            (m.size for m in result.metadata.values()), reverse=True,
        )
        assert 2 <= len(sizes) <= 60
        assert sizes[0] < 0.5 * 354
        assert result.quality["modularity"] > 0.3
        # Ranked ids are contiguous from 0.
        assert set(result.membership.values()) == set(range(len(sizes)))

    def test_agrees_strongly_with_the_shipped_topic_partition(self):
        """Sanity-check the agreement metrics on real, non-synthetic data."""
        leiden = _load_fixture_labels("community_leiden")
        topics = _load_fixture_labels("topic")
        # The fixture's own community_methods.csv records these values.
        assert normalized_mutual_info(leiden, topics) == pytest.approx(0.4897, abs=5e-5)
        assert adjusted_rand(leiden, topics) == pytest.approx(0.2562, abs=5e-5)
