"""Tests for the search-surface methods on :class:`SemanticScholarClient`.

These tests monkey-patch ``client._http.get`` directly so no network
traffic occurs and so each test can assert exactly which path / params /
``req_type`` the API method passed to the HTTP layer.

Cache wiring lands in PA-05; PA-01 only adds the bare HTTP surface, so
none of these tests touch the cache. They DO construct a real ``Cache``
because the client constructor requires one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from citeclaw.cache import Cache
from citeclaw.clients.s2 import SemanticScholarClient
from citeclaw.config import BudgetTracker, Settings


@pytest.fixture
def client(tmp_path: Path) -> SemanticScholarClient:
    cfg = Settings(screening_model="stub", data_dir=tmp_path, s2_api_key="")
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    c = SemanticScholarClient(cfg, cache, budget)
    yield c
    c.close()
    cache.close()


class _Recorder:
    """Reusable monkey-patch helper for ``client._http.get``.

    Replaces the bound method with a plain callable that captures every
    invocation and returns a canned response. Tests can then assert on
    ``rec.calls[i]`` for path/params/req_type and on ``rec.last`` for the
    most recent invocation.
    """

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def install(self, client: SemanticScholarClient) -> None:
        def fake_get(path: str, params: dict[str, Any] | None = None, *, req_type: str = "other"):
            self.calls.append({"path": path, "params": params, "req_type": req_type})
            if isinstance(self._response, BaseException):
                raise self._response
            return self._response

        client._http.get = fake_get  # type: ignore[method-assign]

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "no http.get call captured"
        return self.calls[-1]


# ---------------------------------------------------------------------------
# search_bulk
# ---------------------------------------------------------------------------


class TestSearchBulk:
    def test_basic_call_shape(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [{"paperId": "p1", "title": "Foo"}], "total": 1, "token": None})
        rec.install(client)

        result = client.search_bulk("transformer architectures")

        assert rec.last["path"] == "/paper/search/bulk"
        assert rec.last["req_type"] == "search"
        params = rec.last["params"]
        assert params["query"] == "transformer architectures"
        assert params["fields"] == "paperId,title"
        assert params["limit"] == 1000
        assert result["total"] == 1
        assert result["data"][0]["paperId"] == "p1"

    def test_forwards_all_whitelisted_filters(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [], "total": 0, "token": None})
        rec.install(client)

        client.search_bulk(
            "deep learning",
            filters={
                "year": "2018-2025",
                "venue": "Nature,Science",
                "fieldsOfStudy": "Computer Science,Biology",
                "minCitationCount": 50,
                "publicationTypes": "JournalArticle",
                "publicationDateOrYear": "2024-01-01:",
                "openAccessPdf": "",
            },
        )

        params = rec.last["params"]
        assert params["year"] == "2018-2025"
        assert params["venue"] == "Nature,Science"
        assert params["fieldsOfStudy"] == "Computer Science,Biology"
        assert params["minCitationCount"] == 50
        assert params["publicationTypes"] == "JournalArticle"
        assert params["publicationDateOrYear"] == "2024-01-01:"
        # openAccessPdf is allowed to be the empty string (presence flag)
        assert params["openAccessPdf"] == ""

    def test_drops_unknown_filter_keys(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", filters={"year": 2024, "bogus_key": "ignored"})

        params = rec.last["params"]
        assert params["year"] == 2024
        assert "bogus_key" not in params

    def test_drops_none_valued_filters(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", filters={"year": None, "venue": "Nature"})

        params = rec.last["params"]
        assert "year" not in params
        assert params["venue"] == "Nature"

    def test_sort_and_token_forwarded_when_set(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [], "token": "next-page"})
        rec.install(client)

        client.search_bulk("q", sort="citationCount:desc", token="prev-page", limit=500)

        params = rec.last["params"]
        assert params["sort"] == "citationCount:desc"
        assert params["token"] == "prev-page"
        assert params["limit"] == 500

    def test_sort_and_token_omitted_when_unset(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q")

        params = rec.last["params"]
        assert "sort" not in params
        assert "token" not in params

    def test_no_filters_argument_works(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", filters=None)

        params = rec.last["params"]
        # None of the whitelisted keys should be present
        for k in (
            "year", "venue", "fieldsOfStudy", "minCitationCount",
            "publicationTypes", "publicationDateOrYear", "openAccessPdf",
        ):
            assert k not in params


# ---------------------------------------------------------------------------
# search_match
# ---------------------------------------------------------------------------


class TestSearchMatch:
    def test_returns_first_data_item(self, client: SemanticScholarClient):
        rec = _Recorder(
            {
                "data": [
                    {"paperId": "match-1", "title": "Attention Is All You Need"},
                    {"paperId": "match-2", "title": "Other"},
                ]
            }
        )
        rec.install(client)

        result = client.search_match("Attention Is All You Need")

        assert rec.last["path"] == "/paper/search/match"
        assert rec.last["req_type"] == "search_match"
        params = rec.last["params"]
        assert params["query"] == "Attention Is All You Need"
        # full PAPER_FIELDS so the caller can inspect more than the id
        assert "paperId" in params["fields"]
        assert result is not None
        assert result["paperId"] == "match-1"

    def test_returns_none_when_data_is_empty(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        assert client.search_match("Nonexistent paper") is None

    def test_returns_none_when_response_has_no_data_key(self, client: SemanticScholarClient):
        rec = _Recorder({})
        rec.install(client)

        assert client.search_match("Nonexistent paper") is None

    def test_returns_none_on_404(self, client: SemanticScholarClient):
        request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match")
        response = httpx.Response(404, request=request, json={"error": "Title match not found"})
        rec = _Recorder(httpx.HTTPStatusError("404", request=request, response=response))
        rec.install(client)

        assert client.search_match("Nonexistent paper") is None

    def test_reraises_non_404_http_errors(self, client: SemanticScholarClient):
        request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match")
        response = httpx.Response(500, request=request, json={"error": "boom"})
        rec = _Recorder(httpx.HTTPStatusError("500", request=request, response=response))
        rec.install(client)

        with pytest.raises(httpx.HTTPStatusError):
            client.search_match("anything")


# ---------------------------------------------------------------------------
# search_relevance
# ---------------------------------------------------------------------------


class TestSearchRelevance:
    def test_basic_call_shape(self, client: SemanticScholarClient):
        rec = _Recorder(
            {
                "total": 2,
                "offset": 0,
                "data": [
                    {"paperId": "r1", "title": "Result 1"},
                    {"paperId": "r2", "title": "Result 2"},
                ],
            }
        )
        rec.install(client)

        result = client.search_relevance("graph neural networks")

        assert rec.last["path"] == "/paper/search"
        assert rec.last["req_type"] == "search"
        params = rec.last["params"]
        assert params["query"] == "graph neural networks"
        assert params["fields"] == "paperId,title"
        assert params["limit"] == 100  # default
        assert params["offset"] == 0  # default
        assert result["total"] == 2
        assert len(result["data"]) == 2

    def test_custom_limit_and_offset(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [], "total": 0, "offset": 50})
        rec.install(client)

        client.search_relevance("topic", limit=50, offset=50)

        params = rec.last["params"]
        assert params["limit"] == 50
        assert params["offset"] == 50
