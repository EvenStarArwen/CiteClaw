"""Tests for the cluster naming pipeline (c-TF-IDF + LLM)."""

from __future__ import annotations

import pytest

from citeclaw.cluster.base import ClusterMetadata
from citeclaw.cluster.representation import (
    _parse_naming_response,
    extract_keywords_ctfidf,
    name_topics_via_llm,
    select_representative_papers,
)
from citeclaw.models import PaperRecord

# c-TF-IDF needs sklearn (a transitive dep of hdbscan in the topic_model
# extras). Skip the keyword tests if sklearn isn't installed.
try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _paper(pid: str, title: str = "", abstract: str = "") -> PaperRecord:
    return PaperRecord(paper_id=pid, title=title, abstract=abstract)


# ---------------------------------------------------------------------------
# c-TF-IDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
class TestExtractKeywordsCtfidf:
    def test_three_clusters_distinct_vocab(self):
        """Each cluster has a clearly dominant token; c-TF-IDF should surface
        that token in the per-cluster top-N."""
        papers = {
            "a1": _paper("a1", "neural network deep learning", "neural"),
            "a2": _paper("a2", "neural neural neural", "neural neural"),
            "a3": _paper("a3", "deep neural networks", "neural"),
            "b1": _paper("b1", "phylogenetic tree analysis", "phylogenetic"),
            "b2": _paper("b2", "phylogenetic phylogenetic", "phylogenetic"),
            "b3": _paper("b3", "phylogenetic species tree", "phylogenetic"),
            "c1": _paper("c1", "kinase enzyme assay", "kinase"),
            "c2": _paper("c2", "kinase kinase enzyme", "kinase"),
            "c3": _paper("c3", "kinase substrate binding", "kinase"),
        }
        membership = {
            "a1": 0, "a2": 0, "a3": 0,
            "b1": 1, "b2": 1, "b3": 1,
            "c1": 2, "c2": 2, "c3": 2,
        }
        out = extract_keywords_ctfidf(membership, papers, n_keywords=5)
        assert set(out.keys()) == {0, 1, 2}
        # Each cluster's top-1 keyword should be its distinguishing word.
        assert out[0][0] == "neural"
        assert out[1][0] == "phylogenetic"
        assert out[2][0] == "kinase"

    def test_excludes_noise_cluster(self):
        papers = {"x": _paper("x", "anything")}
        membership = {"x": -1}
        assert extract_keywords_ctfidf(membership, papers) == {}

    def test_empty_membership(self):
        assert extract_keywords_ctfidf({}, {}) == {}

    def test_papers_without_text_silently_skipped(self):
        papers = {
            "a": _paper("a"),  # no title, no abstract
            "b": _paper("b", "real title here"),
        }
        membership = {"a": 0, "b": 0}
        out = extract_keywords_ctfidf(membership, papers, n_keywords=3)
        # Cluster 0 still ranks SOMETHING (from b's title) — at least one
        # keyword.
        assert 0 in out
        assert len(out[0]) >= 1


# ---------------------------------------------------------------------------
# select_representative_papers
# ---------------------------------------------------------------------------


class TestSelectRepresentativePapers:
    def test_picks_centroid_closest(self):
        """Cluster 0: 3 papers around the centroid, with one obvious outlier.
        The two non-outliers should be returned, and the outlier excluded."""
        embeddings = {
            "tight":   [1.0, 0.0],
            "near":    [0.95, 0.05],
            "outlier": [0.6, 0.4],
        }
        membership = {"tight": 0, "near": 0, "outlier": 0}
        out = select_representative_papers(membership, embeddings, n=2)
        # The outlier (0.6, 0.4) is much farther from the centroid in cosine
        # space than either "tight" or "near", so the top-2 must exclude it.
        assert "outlier" not in out[0]
        assert set(out[0]) == {"tight", "near"}
        assert len(out[0]) == 2

    def test_excludes_noise(self):
        embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        membership = {"a": -1, "b": -1}
        out = select_representative_papers(membership, embeddings, n=5)
        assert out == {}

    def test_drops_papers_without_embedding(self):
        embeddings = {"a": [1.0, 0.0], "b": None, "c": [0.0, 1.0]}
        membership = {"a": 0, "b": 0, "c": 0}
        out = select_representative_papers(membership, embeddings, n=5)
        # b is dropped silently.
        assert set(out[0]) == {"a", "c"}

    def test_separate_clusters_get_separate_centroids(self):
        embeddings = {
            "a1": [1.0, 0.0],
            "a2": [0.95, 0.05],
            "b1": [0.0, 1.0],
            "b2": [0.05, 0.95],
        }
        membership = {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
        out = select_representative_papers(membership, embeddings, n=5)
        assert set(out[0]) == {"a1", "a2"}
        assert set(out[1]) == {"b1", "b2"}


# ---------------------------------------------------------------------------
# _parse_naming_response
# ---------------------------------------------------------------------------


class TestParseNamingResponse:
    def test_clean_json(self):
        text = '{"topic_label": "deep learning", "summary": "Papers about NN"}'
        assert _parse_naming_response(text) == ("deep learning", "Papers about NN")

    def test_label_alias(self):
        text = '{"label": "old key", "summary": "ok"}'
        assert _parse_naming_response(text) == ("old key", "ok")

    def test_code_fenced_json(self):
        text = '```json\n{"topic_label": "ml", "summary": "x"}\n```'
        assert _parse_naming_response(text) == ("ml", "x")

    def test_invalid_json_returns_none(self):
        assert _parse_naming_response("not json at all") is None

    def test_non_dict_returns_none(self):
        assert _parse_naming_response("[1, 2, 3]") is None

    def test_empty_fields_returns_none(self):
        assert _parse_naming_response('{"topic_label": "", "summary": ""}') is None


# ---------------------------------------------------------------------------
# name_topics_via_llm — drives the stub LLM through LLMClient protocol
# ---------------------------------------------------------------------------


class TestNameTopicsViaLlm:
    def _build_client(self, ctx):
        from citeclaw.clients.llm import build_llm_client
        return build_llm_client(ctx.config, ctx.budget)

    def test_stub_fills_label_and_summary(self, ctx):
        """The stub recognises the topic_naming prompt and returns
        ``{"topic_label": "stub-topic", "summary": "stub summary"}``."""
        papers = {
            "p1": _paper("p1", "neural networks"),
            "p2": _paper("p2", "deep learning"),
        }
        metadata = {
            0: ClusterMetadata(
                size=2,
                keywords=["neural", "deep"],
                representative_papers=["p1", "p2"],
            )
        }
        client = self._build_client(ctx)
        name_topics_via_llm(metadata, papers, client=client)
        assert metadata[0].label == "stub-topic"
        assert metadata[0].summary == "stub summary"

    def test_clusters_without_keywords_or_repr_are_skipped(self, ctx):
        metadata = {0: ClusterMetadata(size=5)}  # nothing to feed the LLM
        client = self._build_client(ctx)
        name_topics_via_llm(metadata, {}, client=client)
        assert metadata[0].label == ""
        assert metadata[0].summary == ""

    def test_noise_cluster_skipped(self, ctx):
        metadata = {-1: ClusterMetadata(size=5, keywords=["a"], representative_papers=["x"])}
        client = self._build_client(ctx)
        name_topics_via_llm(metadata, {"x": _paper("x", "x")}, client=client)
        assert metadata[-1].label == ""

    def test_empty_metadata_is_noop(self, ctx):
        client = self._build_client(ctx)
        name_topics_via_llm({}, {}, client=client)  # must not raise

    def test_parse_failure_falls_back_to_keywords(self, ctx, monkeypatch):
        """If the LLM returns junk, the cluster falls back to a
        space-joined keyword label."""
        metadata = {
            0: ClusterMetadata(
                size=2,
                keywords=["alpha", "beta", "gamma"],
                representative_papers=["p1"],
            )
        }
        # Force the parser to return None.
        monkeypatch.setattr(
            "citeclaw.cluster.representation._parse_naming_response",
            lambda text: None,
        )
        client = self._build_client(ctx)
        name_topics_via_llm(metadata, {"p1": _paper("p1", "x")}, client=client)
        # Fallback label is the top keywords joined by spaces.
        assert metadata[0].label == "alpha beta gamma"
