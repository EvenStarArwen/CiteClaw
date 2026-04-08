"""Tests for the embedding-based ``TopicModelClusterer``.

Most of these tests need ``umap-learn`` and ``hdbscan`` (the optional
``topic_model`` extras). When the libraries aren't installed, the test
class is skipped — except for the "missing extras raises a clear error"
test which monkeypatches the imports to simulate the missing-package
case and runs unconditionally.
"""

from __future__ import annotations

import sys

import pytest

from citeclaw.cluster import TopicModelClusterer
from citeclaw.cluster.base import ClusterResult
from citeclaw.models import PaperRecord
from tests.fakes import FakeS2Client, make_paper

# Probe the optional deps once at import time so we can skip cleanly.
try:
    import hdbscan  # noqa: F401
    import numpy  # noqa: F401
    import umap  # noqa: F401

    _HAS_TOPIC_EXTRAS = True
except ImportError:
    _HAS_TOPIC_EXTRAS = False


def _paper(pid: str) -> PaperRecord:
    return PaperRecord(paper_id=pid, title=f"title-{pid}")


class TestExtrasMissingError:
    """Verify the constructor raises a clear error when extras are missing.

    Run unconditionally — we monkeypatch ``sys.modules`` to simulate the
    case where ``umap`` is unavailable, regardless of what's actually
    installed in the test environment.
    """

    def test_missing_umap_raises_runtime_error(self, monkeypatch):
        # Hide umap from the import system; lazy-import inside cluster()
        # should then fail and the constructor should re-raise as RuntimeError
        # with the extras-install hint.
        monkeypatch.setitem(sys.modules, "umap", None)
        c = TopicModelClusterer(min_cluster_size=2)
        with pytest.raises(RuntimeError, match="topic_model"):
            c.cluster([_paper("p1")], _StubCtx())

    def test_missing_hdbscan_raises_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "hdbscan", None)
        c = TopicModelClusterer(min_cluster_size=2)
        with pytest.raises(RuntimeError, match="topic_model"):
            c.cluster([_paper("p1")], _StubCtx())


class _StubCtx:
    """Minimal context-like object for the missing-extras tests.

    Only needs ``s2.fetch_embeddings_batch`` so the constructor's lazy
    imports run before the embedding fetch.
    """
    class _S2:
        def fetch_embeddings_batch(self, ids):
            return {pid: None for pid in ids}
    s2 = _S2()


# ---------------------------------------------------------------------------
# Integration tests — only run when the extras are installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TOPIC_EXTRAS, reason="topic_model extras not installed")
class TestTopicModelClustering:
    """End-to-end UMAP+HDBSCAN clustering on a synthetic embedding corpus."""

    def _build_corpus(self, n_per_cluster=10) -> tuple[list[PaperRecord], FakeS2Client, _Ctx]:
        """Build three obvious clusters of papers via synthetic 8-dim embeddings."""
        import random
        random.seed(0)

        client = FakeS2Client()
        papers: list[PaperRecord] = []

        # Cluster A: vectors near [1, 0, 0, 0, 0, 0, 0, 0]
        for i in range(n_per_cluster):
            base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            jitter = [random.uniform(-0.05, 0.05) for _ in range(8)]
            v = [a + j for a, j in zip(base, jitter)]
            pid = f"A{i}"
            client.add(make_paper(pid, embedding=v))
            papers.append(_paper(pid))

        # Cluster B: vectors near [0, 1, 0, 0, 0, 0, 0, 0]
        for i in range(n_per_cluster):
            base = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            jitter = [random.uniform(-0.05, 0.05) for _ in range(8)]
            v = [a + j for a, j in zip(base, jitter)]
            pid = f"B{i}"
            client.add(make_paper(pid, embedding=v))
            papers.append(_paper(pid))

        # Cluster C: vectors near [0, 0, 1, 0, 0, 0, 0, 0]
        for i in range(n_per_cluster):
            base = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            jitter = [random.uniform(-0.05, 0.05) for _ in range(8)]
            v = [a + j for a, j in zip(base, jitter)]
            pid = f"C{i}"
            client.add(make_paper(pid, embedding=v))
            papers.append(_paper(pid))

        ctx = _Ctx(s2=client)
        return papers, client, ctx

    def test_finds_multiple_clusters(self):
        papers, _client, ctx = self._build_corpus(n_per_cluster=10)
        c = TopicModelClusterer(
            min_cluster_size=3,
            n_neighbors=5,
            random_state=42,
        )
        result = c.cluster(papers, ctx)
        assert isinstance(result, ClusterResult)
        assert result.algorithm == "topic_model"
        # 30 papers, all should be in the membership map.
        assert set(result.membership.keys()) == {p.paper_id for p in papers}
        # Should find at least 2 clusters (might be 2 or 3 depending on UMAP).
        non_noise = {c for c in result.membership.values() if c != -1}
        assert len(non_noise) >= 2
        # Per-cluster size metadata is filled in.
        for cid, md in result.metadata.items():
            assert md.size > 0

    def test_deterministic_across_runs(self):
        papers, _client, ctx = self._build_corpus(n_per_cluster=10)
        c1 = TopicModelClusterer(min_cluster_size=3, n_neighbors=5, random_state=42)
        c2 = TopicModelClusterer(min_cluster_size=3, n_neighbors=5, random_state=42)
        r1 = c1.cluster(papers, ctx)
        r2 = c2.cluster(papers, ctx)
        assert r1.membership == r2.membership

    def test_papers_without_embeddings_get_noise(self):
        papers, client, ctx = self._build_corpus(n_per_cluster=10)
        # Add 3 extra papers with NO embeddings.
        for i in range(3):
            pid = f"NOEMB{i}"
            client.add(make_paper(pid))  # no embedding kwarg
            papers.append(_paper(pid))

        c = TopicModelClusterer(min_cluster_size=3, n_neighbors=5, random_state=42)
        result = c.cluster(papers, ctx)
        for i in range(3):
            assert result.membership[f"NOEMB{i}"] == -1

    def test_too_few_papers_returns_all_noise(self):
        """If fewer than min_cluster_size papers have embeddings, every
        paper lands in cluster -1 (and HDBSCAN is never invoked)."""
        client = FakeS2Client()
        client.add(make_paper("p1", embedding=[1.0, 0.0]))
        client.add(make_paper("p2", embedding=[0.0, 1.0]))
        ctx = _Ctx(s2=client)
        papers = [_paper("p1"), _paper("p2")]

        c = TopicModelClusterer(min_cluster_size=10, n_neighbors=5, random_state=42)
        result = c.cluster(papers, ctx)
        assert all(cid == -1 for cid in result.membership.values())

    def test_empty_signal(self):
        ctx = _Ctx(s2=FakeS2Client())
        c = TopicModelClusterer(min_cluster_size=3)
        result = c.cluster([], ctx)
        assert result.membership == {}
        assert result.algorithm == "topic_model"


class _Ctx:
    """Minimal context for topic_model.cluster() — only needs ``s2``."""
    def __init__(self, s2):
        self.s2 = s2
