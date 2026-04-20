"""Tests for ``citeclaw.clients.s2.cache_layer.S2CacheLayer``.

The layer is a thin shim around :class:`Cache` whose only added value
is bumping ``BudgetTracker._s2_cache[<bucket>]`` on every read-through
hit. Tests verify both behaviours: the wrapped read returns the right
data (or None on miss) AND the cache-hit counter is recorded under the
correct per-method bucket.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citeclaw.budget import BudgetTracker
from citeclaw.cache import Cache
from citeclaw.clients.s2.cache_layer import S2CacheLayer


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    c = Cache(tmp_path / "cache.db")
    yield c
    c.close()


@pytest.fixture
def budget() -> BudgetTracker:
    return BudgetTracker()


@pytest.fixture
def layer(cache: Cache, budget: BudgetTracker) -> S2CacheLayer:
    return S2CacheLayer(cache, budget)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_miss_returns_none_no_hit_recorded(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        assert layer.get_metadata("missing") is None
        assert budget._s2_cache.get("metadata", 0) == 0

    def test_put_then_get_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        layer.put_metadata("p1", {"paperId": "p1", "title": "Foo"})
        assert layer.get_metadata("p1") == {"paperId": "p1", "title": "Foo"}
        assert budget._s2_cache["metadata"] == 1

    def test_repeated_hits_increment_bucket(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        layer.put_metadata("p1", {"paperId": "p1"})
        layer.get_metadata("p1")
        layer.get_metadata("p1")
        layer.get_metadata("p1")
        assert budget._s2_cache["metadata"] == 3


# ---------------------------------------------------------------------------
# References + citations + has_*
# ---------------------------------------------------------------------------


class TestReferences:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        edges = [{"citedPaper": {"paperId": "ref1"}}]
        layer.put_references("p1", edges)
        assert layer.get_references("p1") == edges
        assert budget._s2_cache["references"] == 1
        assert layer.has_references("p1") is True

    def test_has_references_false_on_miss(self, layer: S2CacheLayer):
        assert layer.has_references("missing") is False


class TestCitations:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        edges = [{"citingPaper": {"paperId": "cit1"}}]
        layer.put_citations("p1", edges)
        assert layer.get_citations("p1") == edges
        assert budget._s2_cache["citations"] == 1

    def test_has_citations_false_on_miss(self, layer: S2CacheLayer):
        assert layer.has_citations("missing") is False


# ---------------------------------------------------------------------------
# Embeddings — sentinel handling
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        layer.put_embedding("p1", [0.1, 0.2, 0.3])
        assert layer.get_embedding("p1") == [0.1, 0.2, 0.3]
        assert budget._s2_cache["embeddings"] == 1

    def test_empty_list_sentinel_records_hit_returns_none(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        # Storing [] is the sentinel for "S2 confirmed no embedding".
        # ``get_embedding`` returns None but the lookup still counts as
        # a hit so we don't refetch.
        layer.put_embedding("p1", [])
        assert layer.get_embedding("p1") is None
        assert layer.has_embedding("p1") is True
        assert budget._s2_cache["embeddings"] == 1

    def test_no_hit_recorded_when_uncached(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        assert layer.get_embedding("missing") is None
        assert budget._s2_cache.get("embeddings", 0) == 0


# ---------------------------------------------------------------------------
# Author metadata + papers
# ---------------------------------------------------------------------------


class TestAuthorMetadata:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        layer.put_author_metadata("A1", {"hIndex": 30})
        assert layer.get_author_metadata("A1") == {"hIndex": 30}
        assert budget._s2_cache["author_metadata"] == 1

    def test_has_author_metadata_false_on_miss(self, layer: S2CacheLayer):
        assert layer.has_author_metadata("missing") is False


class TestAuthorPapers:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        papers = [{"paperId": "p1"}, {"paperId": "p2"}]
        layer.put_author_papers("A1", papers)
        assert layer.get_author_papers("A1") == papers
        assert budget._s2_cache["author_papers"] == 1


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


class TestSearchResults:
    def test_round_trip_records_hit(
        self, layer: S2CacheLayer, budget: BudgetTracker,
    ):
        layer.put_search_results(
            "abc123", {"q": "test"}, {"total": 5, "data": []},
        )
        assert layer.get_search_results("abc123") == {"total": 5, "data": []}
        assert budget._s2_cache["search"] == 1

    def test_miss_returns_none(self, layer: S2CacheLayer, budget: BudgetTracker):
        assert layer.get_search_results("nope") is None
        assert budget._s2_cache.get("search", 0) == 0
