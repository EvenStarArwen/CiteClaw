"""End-to-end tests for the ``Cluster`` step."""

from __future__ import annotations

import pytest

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.models import PaperRecord
from citeclaw.steps.cluster import Cluster

try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _paper(pid: str, refs: list[str] | None = None, title: str = "") -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=title or f"title-{pid}",
        abstract=f"abstract for {pid}",
        references=refs or [],
    )


class _FakeClusterer:
    """Tiny clusterer that returns a pre-baked membership."""

    name = "fake"

    def __init__(self, membership: dict[str, int]):
        self._m = membership

    def cluster(self, signal, ctx) -> ClusterResult:
        sizes: dict[int, int] = {}
        for cid in self._m.values():
            if cid != -1:
                sizes[cid] = sizes.get(cid, 0) + 1
        return ClusterResult(
            membership=dict(self._m),
            metadata={cid: ClusterMetadata(size=n) for cid, n in sizes.items()},
            algorithm="fake",
        )


class TestClusterStepConstruction:
    def test_requires_store_as(self):
        with pytest.raises(ValueError, match="store_as"):
            Cluster(store_as="", algorithm={"type": "walktrap"})

    def test_requires_algorithm(self):
        with pytest.raises(ValueError, match="algorithm"):
            Cluster(store_as="x", algorithm=None)

    def test_invalid_naming_mode(self):
        with pytest.raises(ValueError, match="naming.mode"):
            Cluster(
                store_as="x",
                algorithm={"type": "walktrap"},
                naming={"mode": "fancy"},
            )


class TestClusterStepRun:
    def _wire_two_cluster_collection(self, ctx):
        # Two cluster citation graph: A-B-C and D-E-F.
        coll = {
            "A": _paper("A"),
            "B": _paper("B", refs=["A"]),
            "C": _paper("C", refs=["A", "B"]),
            "D": _paper("D"),
            "E": _paper("E", refs=["D"]),
            "F": _paper("F", refs=["D", "E"]),
        }
        ctx.collection = coll
        return list(coll.values())

    def test_walktrap_no_naming(self, ctx):
        signal = self._wire_two_cluster_collection(ctx)
        step = Cluster(
            store_as="comm",
            algorithm={"type": "walktrap", "n_communities": 2},
        )
        result = step.run(signal, ctx)
        assert "comm" in ctx.clusters
        cr = ctx.clusters["comm"]
        assert isinstance(cr, ClusterResult)
        assert set(cr.membership.keys()) == {p.paper_id for p in signal}
        # Signal is unchanged (no drop_noise).
        assert [p.paper_id for p in result.signal] == [p.paper_id for p in signal]
        # Stats reflect cluster count.
        assert result.stats["n_clusters"] >= 2
        assert result.stats["store_as"] == "comm"
        assert result.stats["named"] is False

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_tfidf_naming_fills_keywords(self, ctx):
        # Make papers with cluster-distinguishing vocab.
        papers = [
            PaperRecord(paper_id="A", title="neural network", abstract="neural neural"),
            PaperRecord(paper_id="B", title="neural deep", abstract="deep deep", references=["A"]),
            PaperRecord(paper_id="C", title="deep learning", abstract="deep neural", references=["A"]),
            PaperRecord(paper_id="D", title="phylogenetic tree", abstract="phylogenetic"),
            PaperRecord(paper_id="E", title="phylogenetic species", abstract="phylogenetic", references=["D"]),
            PaperRecord(paper_id="F", title="phylogenetic analysis", abstract="phylogenetic", references=["D"]),
        ]
        ctx.collection = {p.paper_id: p for p in papers}

        # Inject our fake clusterer via build_clusterer monkeypatch.
        membership = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        import citeclaw.steps.cluster as cluster_step_mod

        def _fake_build(spec):
            return _FakeClusterer(membership)

        original = cluster_step_mod.build_clusterer
        cluster_step_mod.build_clusterer = _fake_build  # type: ignore[assignment]
        try:
            step = Cluster(
                store_as="topics",
                algorithm={"type": "fake"},
                naming={"mode": "tfidf", "n_keywords": 5},
            )
            step.run(papers, ctx)
        finally:
            cluster_step_mod.build_clusterer = original  # type: ignore[assignment]

        cr = ctx.clusters["topics"]
        assert cr.metadata[0].keywords  # non-empty
        assert cr.metadata[1].keywords
        assert "neural" in cr.metadata[0].keywords or "deep" in cr.metadata[0].keywords
        assert "phylogenetic" in cr.metadata[1].keywords

    def test_drop_noise_filters_signal(self, ctx):
        signal = [_paper("p1"), _paper("p2"), _paper("p3")]
        ctx.collection = {p.paper_id: p for p in signal}
        membership = {"p1": 0, "p2": 0, "p3": -1}

        import citeclaw.steps.cluster as cluster_step_mod
        original = cluster_step_mod.build_clusterer
        cluster_step_mod.build_clusterer = lambda spec: _FakeClusterer(membership)  # type: ignore[assignment]
        try:
            step = Cluster(
                store_as="x",
                algorithm={"type": "fake"},
                drop_noise=True,
            )
            result = step.run(signal, ctx)
        finally:
            cluster_step_mod.build_clusterer = original  # type: ignore[assignment]
        out_ids = {p.paper_id for p in result.signal}
        assert out_ids == {"p1", "p2"}
        assert result.stats["drop_noise"] is True
        assert result.stats["n_noise"] == 1

    def test_overwrite_replaces_existing(self, ctx):
        """Re-running a Cluster step with the same store_as overwrites the
        previous result. (A warning is also logged for visibility, but
        log capture is unreliable here because citeclaw's logger has
        propagate=False once setup_logging has run.)"""
        signal = [_paper("p1")]
        ctx.collection = {"p1": signal[0]}
        ctx.clusters["x"] = ClusterResult(membership={"p1": 0}, algorithm="seed")

        import citeclaw.steps.cluster as cluster_step_mod
        original = cluster_step_mod.build_clusterer
        cluster_step_mod.build_clusterer = (  # type: ignore[assignment]
            lambda spec: _FakeClusterer({"p1": 0})
        )
        try:
            step = Cluster(store_as="x", algorithm={"type": "fake"})
            step.run(signal, ctx)
        finally:
            cluster_step_mod.build_clusterer = original  # type: ignore[assignment]
        # The new result replaced the seeded one.
        assert ctx.clusters["x"].algorithm == "fake"

    def test_empty_signal(self, ctx):
        step = Cluster(store_as="empty", algorithm={"type": "walktrap"})
        result = step.run([], ctx)
        assert result.signal == []
        assert result.stats["n_clusters"] == 0
