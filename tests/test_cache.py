"""Tests for :class:`citeclaw.cache.Cache` — the SQLite read-through cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from citeclaw.cache import Cache


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "subdir" / "cache.db"


def test_cache_creates_parent_dirs(cache_path: Path):
    assert not cache_path.parent.exists()
    c = Cache(cache_path)
    assert cache_path.parent.exists()
    assert cache_path.exists()
    c.close()


class TestMetadata:
    def test_get_miss(self, cache_path):
        c = Cache(cache_path)
        assert c.get_metadata("nope") is None
        c.close()

    def test_put_and_get(self, cache_path):
        c = Cache(cache_path)
        data = {"paperId": "p1", "title": "Hello", "year": 2020}
        c.put_metadata("p1", data)
        got = c.get_metadata("p1")
        assert got == data
        c.close()

    def test_replace_existing(self, cache_path):
        c = Cache(cache_path)
        c.put_metadata("p1", {"title": "v1"})
        c.put_metadata("p1", {"title": "v2"})
        assert c.get_metadata("p1") == {"title": "v2"}
        c.close()

    def test_persistence_across_instances(self, cache_path):
        c1 = Cache(cache_path)
        c1.put_metadata("p1", {"title": "persists"})
        c1.close()
        c2 = Cache(cache_path)
        assert c2.get_metadata("p1") == {"title": "persists"}
        c2.close()


class TestReferences:
    def test_put_get_has(self, cache_path):
        c = Cache(cache_path)
        assert not c.has_references("p1")
        edges = [{"citedPaper": {"paperId": "r1"}}, {"citedPaper": {"paperId": "r2"}}]
        c.put_references("p1", edges)
        assert c.has_references("p1")
        assert c.get_references("p1") == edges
        c.close()

    def test_missing_is_none(self, cache_path):
        c = Cache(cache_path)
        assert c.get_references("nope") is None
        assert not c.has_references("nope")
        c.close()


class TestCitations:
    def test_roundtrip(self, cache_path):
        c = Cache(cache_path)
        edges = [{"citingPaper": {"paperId": "c1", "citationCount": 10}}]
        c.put_citations("p1", edges)
        assert c.has_citations("p1")
        assert c.get_citations("p1") == edges
        c.close()


class TestEmbeddings:
    def test_store_and_fetch(self, cache_path):
        c = Cache(cache_path)
        vec = [0.1, 0.2, -0.3]
        c.put_embedding("p1", vec)
        assert c.has_embedding("p1")
        assert c.get_embedding("p1") == vec
        c.close()

    def test_empty_sentinel_returns_none(self, cache_path):
        """An empty list is the 'confirmed no embedding' sentinel — it must
        persist so the caller records that we looked, but reads still come
        back as ``None`` so downstream logic is uniform."""
        c = Cache(cache_path)
        c.put_embedding("p1", [])
        assert c.has_embedding("p1")
        assert c.get_embedding("p1") is None
        c.close()

    def test_missing_is_none(self, cache_path):
        c = Cache(cache_path)
        assert c.get_embedding("nope") is None
        assert not c.has_embedding("nope")
        c.close()


class TestAuthorMetadata:
    def test_roundtrip(self, cache_path):
        c = Cache(cache_path)
        assert not c.has_author_metadata("A1")
        data = {"name": "Alice", "hIndex": 10}
        c.put_author_metadata("A1", data)
        assert c.has_author_metadata("A1")
        assert c.get_author_metadata("A1") == data
        c.close()

    def test_missing_is_none(self, cache_path):
        c = Cache(cache_path)
        assert c.get_author_metadata("nope") is None
        c.close()


class TestSearchQueries:
    """PA-04 search-results table — keyed by query_hash, TTL-aware."""

    def test_put_get_roundtrip(self, cache_path):
        c = Cache(cache_path)
        query = {"q": "transformers", "filters": {"year": "2018-2025"}}
        result = {"data": [{"paperId": "p1", "title": "T1"}], "total": 1}
        c.put_search_results("hash-abc", query, result)
        assert c.get_search_results("hash-abc") == result
        c.close()

    def test_get_miss_returns_none(self, cache_path):
        c = Cache(cache_path)
        assert c.get_search_results("nope") is None
        c.close()

    def test_has_search_results_true_after_put(self, cache_path):
        c = Cache(cache_path)
        c.put_search_results("h1", {"q": "x"}, {"data": []})
        assert c.has_search_results("h1") is True
        c.close()

    def test_has_search_results_false_when_missing(self, cache_path):
        c = Cache(cache_path)
        assert c.has_search_results("nope") is False
        c.close()

    def test_replace_existing_query_hash(self, cache_path):
        c = Cache(cache_path)
        c.put_search_results("h1", {"q": "x"}, {"data": [], "v": 1})
        c.put_search_results("h1", {"q": "x"}, {"data": [], "v": 2})
        assert c.get_search_results("h1") == {"data": [], "v": 2}
        c.close()

    def test_persistence_across_instances(self, cache_path):
        c1 = Cache(cache_path)
        c1.put_search_results("h1", {"q": "x"}, {"data": [{"id": 1}]})
        c1.close()
        c2 = Cache(cache_path)
        assert c2.get_search_results("h1") == {"data": [{"id": 1}]}
        c2.close()

    def test_ttl_expired_entry_returns_none(self, cache_path):
        """Backdate ``fetched_at`` and confirm the cache returns None
        once the entry is older than the TTL window."""
        c = Cache(cache_path)
        c.put_search_results("h1", {"q": "x"}, {"data": []})
        # Backdate by 60 days
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        c._conn.execute(
            "UPDATE search_queries SET fetched_at = ? WHERE query_hash = ?",
            (old, "h1"),
        )
        c._conn.commit()
        assert c.get_search_results("h1", ttl_days=30) is None
        assert c.has_search_results("h1", ttl_days=30) is False
        # And confirm a generous ttl recovers it
        assert c.get_search_results("h1", ttl_days=365) == {"data": []}
        c.close()

    def test_ttl_default_is_30_days(self, cache_path):
        """A 31-day-old entry should be considered stale under default TTL."""
        c = Cache(cache_path)
        c.put_search_results("h1", {"q": "x"}, {"data": []})
        old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        c._conn.execute(
            "UPDATE search_queries SET fetched_at = ? WHERE query_hash = ?",
            (old, "h1"),
        )
        c._conn.commit()
        assert c.get_search_results("h1") is None
        c.close()

    def test_query_dict_round_trips_through_json_column(self, cache_path):
        """The ``query_json`` column should preserve the original query
        dict so it can be inspected later for debugging."""
        c = Cache(cache_path)
        query = {"q": "x", "filters": {"year": [2020, 2024]}}
        c.put_search_results("h1", query, {"data": []})
        # Read it back via raw SQL since the public API only returns
        # the result payload.
        with c._cursor() as cur:
            cur.execute("SELECT query_json FROM search_queries WHERE query_hash = ?", ("h1",))
            row = cur.fetchone()
        assert json.loads(row[0]) == query
        c.close()


class TestAuthorPapers:
    """PA-04 author_papers table — full S2 paper list per author."""

    def test_put_get_roundtrip(self, cache_path):
        c = Cache(cache_path)
        papers = [
            {"paperId": "p1", "title": "Paper 1", "year": 2020},
            {"paperId": "p2", "title": "Paper 2", "year": 2021},
        ]
        c.put_author_papers("A1", papers)
        assert c.get_author_papers("A1") == papers
        c.close()

    def test_get_miss_returns_none(self, cache_path):
        c = Cache(cache_path)
        assert c.get_author_papers("nope") is None
        c.close()

    def test_replace_existing_author(self, cache_path):
        c = Cache(cache_path)
        c.put_author_papers("A1", [{"paperId": "old"}])
        c.put_author_papers("A1", [{"paperId": "new"}])
        assert c.get_author_papers("A1") == [{"paperId": "new"}]
        c.close()

    def test_persistence_across_instances(self, cache_path):
        c1 = Cache(cache_path)
        c1.put_author_papers("A1", [{"paperId": "p1"}])
        c1.close()
        c2 = Cache(cache_path)
        assert c2.get_author_papers("A1") == [{"paperId": "p1"}]
        c2.close()

    def test_empty_list_is_distinct_from_missing(self, cache_path):
        """An author with no papers should round-trip as ``[]``, not None
        — that's how the cache distinguishes 'we looked, none found' from
        'we never looked'."""
        c = Cache(cache_path)
        c.put_author_papers("A1", [])
        assert c.get_author_papers("A1") == []
        assert c.get_author_papers("A2") is None
        c.close()
