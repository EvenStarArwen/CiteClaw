"""Tests for the Clusterer Protocol, ClusterResult dataclass, and registry."""

from __future__ import annotations

import pytest

from citeclaw.cluster import (
    CLUSTERER_REGISTRY,
    Clusterer,
    LouvainClusterer,
    TopicModelClusterer,
    WalktrapClusterer,
    build_clusterer,
)
from citeclaw.cluster.base import ClusterMetadata, ClusterResult


class TestClusterResult:
    def test_default_construction(self):
        r = ClusterResult(membership={"a": 0, "b": 0, "c": 1})
        assert r.membership == {"a": 0, "b": 0, "c": 1}
        assert r.metadata == {}
        assert r.algorithm == ""

    def test_full_construction(self):
        md = ClusterMetadata(
            label="machine learning",
            keywords=["neural", "network"],
            summary="Deep learning papers.",
            size=12,
            representative_papers=["a", "b"],
        )
        r = ClusterResult(
            membership={"a": 0},
            metadata={0: md},
            algorithm="topic_model",
        )
        assert r.metadata[0].label == "machine learning"
        assert r.metadata[0].keywords == ["neural", "network"]
        assert r.algorithm == "topic_model"


class TestClusterMetadata:
    def test_defaults(self):
        m = ClusterMetadata()
        assert m.label == ""
        assert m.keywords == []
        assert m.summary == ""
        assert m.size == 0
        assert m.representative_papers == []


class TestClustererProtocol:
    def test_walktrap_satisfies_protocol(self):
        # runtime_checkable Protocol — isinstance does duck-typed check.
        assert isinstance(WalktrapClusterer(), Clusterer)

    def test_louvain_satisfies_protocol(self):
        assert isinstance(LouvainClusterer(), Clusterer)

    def test_topic_model_satisfies_protocol(self):
        assert isinstance(TopicModelClusterer(), Clusterer)


class TestRegistry:
    def test_registry_contents(self):
        assert "walktrap" in CLUSTERER_REGISTRY
        assert "louvain" in CLUSTERER_REGISTRY
        assert "topic_model" in CLUSTERER_REGISTRY
        # No legacy "bertopic" placeholder.
        assert "bertopic" not in CLUSTERER_REGISTRY

    def test_build_string_walktrap(self):
        assert isinstance(build_clusterer("walktrap"), WalktrapClusterer)

    def test_build_string_louvain(self):
        assert isinstance(build_clusterer("louvain"), LouvainClusterer)

    def test_build_string_topic_model(self):
        c = build_clusterer("topic_model")
        assert isinstance(c, TopicModelClusterer)
        # Adaptive defaults (resolved at cluster() time based on signal size,
        # not at construction). The attribute staying None is the signal that
        # adaptive resolution should fire.
        assert c.n_neighbors is None
        assert c.min_cluster_size is None
        assert c.min_samples is None

    def test_build_dict_walktrap_with_kwargs(self):
        c = build_clusterer({"type": "walktrap", "n_communities": 7})
        assert isinstance(c, WalktrapClusterer)
        assert c.n_communities == 7

    def test_build_dict_topic_model_with_kwargs(self):
        c = build_clusterer({
            "type": "topic_model",
            "min_cluster_size": 5,
            "n_neighbors": 10,
            "random_state": 1234,
        })
        assert isinstance(c, TopicModelClusterer)
        assert c.min_cluster_size == 5
        assert c.n_neighbors == 10
        assert c.random_state == 1234

    def test_dict_missing_type(self):
        with pytest.raises(ValueError, match="missing required 'type'"):
            build_clusterer({"n_communities": 3})

    def test_string_unknown(self):
        with pytest.raises(ValueError, match="Unknown clusterer 'foo'"):
            build_clusterer("foo")

    def test_dict_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown clusterer 'foo'"):
            build_clusterer({"type": "foo"})

    def test_bad_spec_type(self):
        with pytest.raises(ValueError, match="Bad clusterer spec"):
            build_clusterer(42)
