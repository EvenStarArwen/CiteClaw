"""Tests for rerank metrics, diversity allocator, and clusterers."""

from __future__ import annotations

import pytest

from citeclaw.cluster import (
    LouvainClusterer,
    WalktrapClusterer,
    build_clusterer,
)
from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.models import PaperRecord
from citeclaw.rerank.diversity import cluster_diverse_top_k
from citeclaw.rerank.metrics import compute_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper(pid, cit=0, refs=None, year=2022):
    return PaperRecord(
        paper_id=pid, title=pid, year=year, citation_count=cit, references=refs or [],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestComputeMetric:
    def test_citation(self, ctx):
        signal = [_paper("A", cit=10), _paper("B", cit=20), _paper("C", cit=5)]
        scores = compute_metric("citation", signal, ctx)
        assert scores == {"A": 10.0, "B": 20.0, "C": 5.0}

    def test_citation_handles_none(self, ctx):
        signal = [_paper("A", cit=None)]
        scores = compute_metric("citation", signal, ctx)
        assert scores == {"A": 0.0}

    def test_pagerank(self, ctx):
        """PageRank needs actual edges — build a 3-node citation chain and
        verify relative ordering. The graph direction is cited → citer
        (rank flows downstream), so ``C`` is a pure sink and should dominate."""
        ctx.collection = {
            "A": _paper("A", refs=[]),
            "B": _paper("B", refs=["A"]),
            "C": _paper("C", refs=["A", "B"]),
        }
        scores = compute_metric("pagerank", list(ctx.collection.values()), ctx)
        assert set(scores.keys()) == {"A", "B", "C"}
        # C has two incoming edges and zero outgoing → strict PageRank max.
        assert scores["C"] > scores["A"]
        assert scores["C"] > scores["B"]
        assert all(v >= 0 for v in scores.values())

    def test_pagerank_empty_collection(self, ctx):
        ctx.collection = {}
        scores = compute_metric("pagerank", [], ctx)
        assert scores == {}

    def test_unknown_metric_raises(self, ctx):
        with pytest.raises(ValueError, match="Unknown rerank metric"):
            compute_metric("magic", [], ctx)


# ---------------------------------------------------------------------------
# Graph clusterers (walktrap, louvain)
# ---------------------------------------------------------------------------


class TestGraphClusterers:
    def _wire_two_cluster_collection(self, ctx):
        # Cluster 1: A-B-C (dense refs among themselves)
        # Cluster 2: D-E-F (dense refs among themselves)
        # No cross edges.
        coll = {
            "A": _paper("A", refs=[]),
            "B": _paper("B", refs=["A"]),
            "C": _paper("C", refs=["A", "B"]),
            "D": _paper("D", refs=[]),
            "E": _paper("E", refs=["D"]),
            "F": _paper("F", refs=["D", "E"]),
        }
        ctx.collection = coll
        return list(coll.values())

    def test_walktrap_returns_cluster_result(self, ctx):
        signal = self._wire_two_cluster_collection(ctx)
        d = WalktrapClusterer(n_communities=2)
        result = d.cluster(signal, ctx)
        assert isinstance(result, ClusterResult)
        assert result.algorithm == "walktrap"
        assert set(result.membership.keys()) == {p.paper_id for p in signal}
        # There should be at least two distinct communities with these inputs.
        assert len({c for c in result.membership.values() if c != -1}) >= 2
        # Per-cluster sizes are populated.
        for cid, md in result.metadata.items():
            assert isinstance(md, ClusterMetadata)
            assert md.size > 0

    def test_walktrap_empty_graph(self, ctx):
        ctx.collection = {}
        result = WalktrapClusterer().cluster([], ctx)
        assert isinstance(result, ClusterResult)
        assert result.membership == {}

    def test_louvain_returns_cluster_result(self, ctx):
        signal = self._wire_two_cluster_collection(ctx)
        d = LouvainClusterer()
        result = d.cluster(signal, ctx)
        assert isinstance(result, ClusterResult)
        assert result.algorithm == "louvain"
        assert set(result.membership.keys()) == {p.paper_id for p in signal}

    def test_louvain_empty_graph(self, ctx):
        ctx.collection = {}
        result = LouvainClusterer().cluster([], ctx)
        assert result.membership == {}


class TestBuildClusterer:
    def test_string_spec(self):
        assert isinstance(build_clusterer("walktrap"), WalktrapClusterer)
        assert isinstance(build_clusterer("louvain"), LouvainClusterer)

    def test_dict_spec(self):
        d = build_clusterer({"type": "walktrap", "n_communities": 5})
        assert isinstance(d, WalktrapClusterer)
        assert d.n_communities == 5

    def test_string_unknown(self):
        with pytest.raises(ValueError, match="Unknown clusterer"):
            build_clusterer("nonesuch")

    def test_dict_unknown(self):
        with pytest.raises(ValueError, match="Unknown clusterer"):
            build_clusterer({"type": "nonesuch"})

    def test_dict_missing_type(self):
        with pytest.raises(ValueError, match="missing required 'type'"):
            build_clusterer({"n_communities": 3})

    def test_bad_spec(self):
        with pytest.raises(ValueError, match="Bad clusterer spec"):
            build_clusterer(42)


# ---------------------------------------------------------------------------
# Diversity allocator
# ---------------------------------------------------------------------------


class _FixedMembershipClusterer:
    """Deterministic clusterer that hands back a pre-baked membership dict."""

    def __init__(self, membership: dict[str, int]):
        self._m = dict(membership)
        self.name = "fixed"

    def cluster(self, signal, ctx) -> ClusterResult:
        return ClusterResult(membership=self._m, algorithm=self.name)


class TestDiversityAllocator:
    def test_empty_membership_falls_back_to_top_k(self, ctx, monkeypatch):
        monkeypatch.setattr(
            "citeclaw.rerank.diversity.build_clusterer",
            lambda cfg: _FixedMembershipClusterer({}),
        )
        signal = [_paper("A", cit=10), _paper("B", cit=5), _paper("C", cit=1)]
        scores = {"A": 10.0, "B": 5.0, "C": 1.0}
        out = cluster_diverse_top_k(signal, scores, ctx, k=2, cfg={"type": "x"})
        assert [p.paper_id for p in out] == ["A", "B"]

    def test_floor_then_proportional(self, ctx, monkeypatch):
        """Two clusters: big (4 members) and small (2 members). k=4.
        Floor: 1 each. Surplus = 2. Proportional: big gets 2*4/6 ≈ 1, small
        gets 2*2/6 ≈ 1. After largest-remainder mop-up, the result should
        contain members from both clusters."""
        membership = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 1, "F": 1}
        monkeypatch.setattr(
            "citeclaw.rerank.diversity.build_clusterer",
            lambda cfg: _FixedMembershipClusterer(membership),
        )
        signal = [
            _paper("A", cit=100), _paper("B", cit=90), _paper("C", cit=80), _paper("D", cit=70),
            _paper("E", cit=60), _paper("F", cit=50),
        ]
        scores = {p.paper_id: float(p.citation_count) for p in signal}
        out = cluster_diverse_top_k(signal, scores, ctx, k=4, cfg={"type": "x"})
        assert len(out) == 4
        out_ids = {p.paper_id for p in out}
        # Must include members of both clusters.
        assert out_ids & {"E", "F"}
        assert out_ids & {"A", "B", "C", "D"}

    def test_k_smaller_than_clusters(self, ctx, monkeypatch):
        """With k < n_clusters, only the largest k clusters get one slot each."""
        membership = {"A": 0, "B": 0, "C": 1, "D": 2}
        monkeypatch.setattr(
            "citeclaw.rerank.diversity.build_clusterer",
            lambda cfg: _FixedMembershipClusterer(membership),
        )
        signal = [
            _paper("A", cit=10), _paper("B", cit=9),
            _paper("C", cit=100), _paper("D", cit=50),
        ]
        scores = {p.paper_id: float(p.citation_count) for p in signal}
        out = cluster_diverse_top_k(signal, scores, ctx, k=2, cfg={"type": "x"})
        assert len(out) == 2
        out_ids = {p.paper_id for p in out}
        # Cluster 0 is largest (2 members), then 1 and 2 tie (1 each).
        # Cluster 0 must get a slot; one of {1, 2} gets the other.
        assert "A" in out_ids or "B" in out_ids

    def test_leftover_fill_top_up_when_cluster_runs_out(self, ctx, monkeypatch):
        """Cluster 1 has only 1 member; k=3 forces the allocator to fall
        back to leftover-fill from cluster 0."""
        membership = {"A": 0, "B": 0, "C": 0, "D": 1}
        monkeypatch.setattr(
            "citeclaw.rerank.diversity.build_clusterer",
            lambda cfg: _FixedMembershipClusterer(membership),
        )
        signal = [
            _paper("A", cit=100), _paper("B", cit=90), _paper("C", cit=80),
            _paper("D", cit=1),
        ]
        scores = {p.paper_id: float(p.citation_count) for p in signal}
        out = cluster_diverse_top_k(signal, scores, ctx, k=3, cfg={"type": "x"})
        assert len(out) == 3
        out_ids = {p.paper_id for p in out}
        assert "D" in out_ids  # cluster 1's lone member
        assert {"A", "B"}.issubset(out_ids)  # top-scored survivors in cluster 0

    def test_noise_papers_skipped_from_floor(self, ctx, monkeypatch):
        """Papers in cluster -1 (noise) don't get a guaranteed slot, but they
        can still resurface in the leftover-fill loop."""
        membership = {"A": 0, "B": 0, "C": -1}
        monkeypatch.setattr(
            "citeclaw.rerank.diversity.build_clusterer",
            lambda cfg: _FixedMembershipClusterer(membership),
        )
        signal = [_paper("A", cit=100), _paper("B", cit=90), _paper("C", cit=10)]
        scores = {p.paper_id: float(p.citation_count) for p in signal}
        # k=2: A and B fill the cluster-0 quota; noise C is excluded.
        out = cluster_diverse_top_k(signal, scores, ctx, k=2, cfg={"type": "x"})
        assert {p.paper_id for p in out} == {"A", "B"}
        # k=3: cluster 0 only has 2 papers, so leftover-fill picks up C.
        out3 = cluster_diverse_top_k(signal, scores, ctx, k=3, cfg={"type": "x"})
        assert {p.paper_id for p in out3} == {"A", "B", "C"}

    def test_named_cluster_reference(self, ctx):
        """diversity={'cluster': 'foo'} reads from ctx.clusters."""
        ctx.clusters["foo"] = ClusterResult(
            membership={"A": 0, "B": 0, "C": 1},
            algorithm="precomputed",
        )
        signal = [_paper("A", cit=10), _paper("B", cit=5), _paper("C", cit=1)]
        scores = {p.paper_id: float(p.citation_count) for p in signal}
        out = cluster_diverse_top_k(signal, scores, ctx, k=2, cfg={"cluster": "foo"})
        assert len(out) == 2
        # Both clusters represented.
        out_ids = {p.paper_id for p in out}
        assert out_ids & {"A", "B"}
        assert out_ids & {"C"}

    def test_named_cluster_missing_raises(self, ctx):
        signal = [_paper("A"), _paper("B")]
        scores = {"A": 1.0, "B": 1.0}
        with pytest.raises(ValueError, match="unknown cluster 'nope'"):
            cluster_diverse_top_k(signal, scores, ctx, k=2, cfg={"cluster": "nope"})
