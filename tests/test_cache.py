"""Tests for :class:`citeclaw.cache.Cache` — the SQLite read-through cache."""

from __future__ import annotations

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
