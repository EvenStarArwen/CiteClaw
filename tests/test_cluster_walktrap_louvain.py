"""Tests for graph-based clusterers (Walktrap + Louvain).

Both run igraph's community detection over the citation graph built
from ``ctx.collection``. The tests use real igraph (no mocking) over
small hand-built collections so the assertions exercise the actual
membership pipeline, including the degenerate fallbacks.
"""

from __future__ import annotations

import pytest

from citeclaw.cluster.base import ClusterResult
from citeclaw.cluster.louvain import LouvainClusterer
from citeclaw.cluster.walktrap import WalktrapClusterer
from citeclaw.context import Context
from citeclaw.models import PaperRecord


def _paper(pid: str, *, refs: list[str] | None = None) -> PaperRecord:
    return PaperRecord(paper_id=pid, references=refs or [])


def _two_clusters(ctx: Context) -> list[PaperRecord]:
    """Two cliques (A1-A2-A3) and (B1-B2-B3) with no inter-edge."""
    papers = [
        _paper("A1", refs=["A2", "A3"]),
        _paper("A2", refs=["A1", "A3"]),
        _paper("A3", refs=["A1", "A2"]),
        _paper("B1", refs=["B2", "B3"]),
        _paper("B2", refs=["B1", "B3"]),
        _paper("B3", refs=["B1", "B2"]),
    ]
    for p in papers:
        ctx.collection[p.paper_id] = p
    return papers


# ---------------------------------------------------------------------------
# Walktrap
# ---------------------------------------------------------------------------


class TestWalktrap:
    def test_empty_collection_returns_empty_result(self, ctx: Context):
        result = WalktrapClusterer(n_communities=2).cluster([], ctx)
        assert isinstance(result, ClusterResult)
        assert result.membership == {}
        assert result.algorithm == "walktrap"

    def test_two_disjoint_cliques_split(self, ctx: Context):
        papers = _two_clusters(ctx)
        result = WalktrapClusterer(n_communities=2).cluster(papers, ctx)
        assert set(result.membership) == {p.paper_id for p in papers}
        a_clusters = {result.membership[p] for p in ("A1", "A2", "A3")}
        b_clusters = {result.membership[p] for p in ("B1", "B2", "B3")}
        assert len(a_clusters) == 1
        assert len(b_clusters) == 1
        assert a_clusters != b_clusters

    def test_membership_only_covers_signal(self, ctx: Context):
        all_papers = _two_clusters(ctx)
        signal_subset = all_papers[:3]
        result = WalktrapClusterer(n_communities=2).cluster(signal_subset, ctx)
        assert set(result.membership) == {"A1", "A2", "A3"}

    def test_metadata_size_matches_membership(self, ctx: Context):
        papers = _two_clusters(ctx)
        result = WalktrapClusterer(n_communities=2).cluster(papers, ctx)
        for cid, meta in result.metadata.items():
            actual_size = sum(1 for v in result.membership.values() if v == cid)
            assert meta.size == actual_size

    def test_zero_communities_falls_back_safely(self, ctx: Context):
        papers = _two_clusters(ctx)
        result = WalktrapClusterer(n_communities=0).cluster(papers, ctx)
        # n=0 gates the partition fallback to all-zero.
        assert set(result.membership.values()) <= {0}


# ---------------------------------------------------------------------------
# Louvain
# ---------------------------------------------------------------------------


class TestLouvain:
    def test_empty_collection_returns_empty_result(self, ctx: Context):
        result = LouvainClusterer().cluster([], ctx)
        assert result.membership == {}
        assert result.algorithm == "louvain"

    def test_two_disjoint_cliques_split(self, ctx: Context):
        papers = _two_clusters(ctx)
        result = LouvainClusterer().cluster(papers, ctx)
        assert set(result.membership) == {p.paper_id for p in papers}
        a_clusters = {result.membership[p] for p in ("A1", "A2", "A3")}
        b_clusters = {result.membership[p] for p in ("B1", "B2", "B3")}
        assert len(a_clusters) == 1
        assert len(b_clusters) == 1
        assert a_clusters != b_clusters

    def test_membership_only_covers_signal(self, ctx: Context):
        all_papers = _two_clusters(ctx)
        signal_subset = all_papers[3:]
        result = LouvainClusterer().cluster(signal_subset, ctx)
        assert set(result.membership) == {"B1", "B2", "B3"}

    def test_metadata_size_matches_membership(self, ctx: Context):
        papers = _two_clusters(ctx)
        result = LouvainClusterer().cluster(papers, ctx)
        for cid, meta in result.metadata.items():
            actual_size = sum(1 for v in result.membership.values() if v == cid)
            assert meta.size == actual_size

    def test_n_communities_is_advisory(self):
        c1 = LouvainClusterer()
        c2 = LouvainClusterer(n_communities=5)
        assert c1.n_communities is None
        assert c2.n_communities == 5
