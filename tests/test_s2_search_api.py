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
    """Reusable monkey-patch helper for ``client._http`` methods.

    Replaces a bound method with a plain callable that captures every
    invocation and returns a canned response. Tests can then assert on
    ``rec.calls[i]`` for the captured args and on ``rec.last`` for the
    most recent invocation. Three install methods are provided so tests
    can patch ``get`` (relative path), ``get_url`` (full URL), or
    ``post`` (full URL with json body).
    """

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def _maybe_raise(self) -> Any:
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    def install(self, client: SemanticScholarClient) -> None:
        def fake_get(path: str, params: dict[str, Any] | None = None, *, req_type: str = "other"):
            self.calls.append({"path": path, "params": params, "req_type": req_type})
            return self._maybe_raise()

        client._http.get = fake_get  # type: ignore[method-assign]

    def install_get_url(self, client: SemanticScholarClient) -> None:
        def fake_get_url(url: str, params: dict[str, Any] | None = None, *, req_type: str = "other"):
            self.calls.append({"url": url, "params": params, "req_type": req_type})
            return self._maybe_raise()

        client._http.get_url = fake_get_url  # type: ignore[method-assign]

    def install_post(self, client: SemanticScholarClient) -> None:
        def fake_post(
            url: str,
            params: dict[str, Any] | None = None,
            json_body: Any = None,
            *,
            req_type: str = "batch",
        ):
            self.calls.append(
                {"url": url, "params": params, "json_body": json_body, "req_type": req_type}
            )
            return self._maybe_raise()

        client._http.post = fake_post  # type: ignore[method-assign]

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "no http call captured"
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


# ---------------------------------------------------------------------------
# fetch_recommendations (PA-02)
# ---------------------------------------------------------------------------


class TestFetchRecommendations:
    def test_basic_post_call_shape(self, client: SemanticScholarClient):
        rec = _Recorder(
            {
                "recommendedPapers": [
                    {"paperId": "rec-1", "title": "Recommended 1"},
                    {"paperId": "rec-2", "title": "Recommended 2"},
                ]
            }
        )
        rec.install_post(client)

        result = client.fetch_recommendations(["anchor-1", "anchor-2"])

        assert (
            rec.last["url"]
            == "https://api.semanticscholar.org/recommendations/v1/papers"
        )
        assert rec.last["req_type"] == "recommendations"
        assert rec.last["json_body"] == {"positivePaperIds": ["anchor-1", "anchor-2"]}
        params = rec.last["params"]
        assert params["fields"] == "paperId,title"
        assert params["limit"] == 100
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["paperId"] == "rec-1"

    def test_negative_ids_added_when_provided(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": []})
        rec.install_post(client)

        client.fetch_recommendations(
            ["anchor-1"], negative_ids=["neg-1", "neg-2"], limit=25,
        )

        body = rec.last["json_body"]
        assert body["positivePaperIds"] == ["anchor-1"]
        assert body["negativePaperIds"] == ["neg-1", "neg-2"]
        assert rec.last["params"]["limit"] == 25

    def test_negative_ids_omitted_when_unset(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": []})
        rec.install_post(client)

        client.fetch_recommendations(["anchor-1"])

        body = rec.last["json_body"]
        assert "negativePaperIds" not in body

    def test_custom_fields_forwarded(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": []})
        rec.install_post(client)

        client.fetch_recommendations(
            ["a"], fields="paperId,title,year,abstract",
        )

        assert rec.last["params"]["fields"] == "paperId,title,year,abstract"

    def test_unwraps_recommended_papers_envelope(self, client: SemanticScholarClient):
        rec = _Recorder(
            {"recommendedPapers": [{"paperId": "p"}]}
        )
        rec.install_post(client)

        result = client.fetch_recommendations(["a"])

        # Result should be the inner list, not the wrapper dict
        assert isinstance(result, list)
        assert result == [{"paperId": "p"}]

    def test_returns_empty_list_when_envelope_missing(self, client: SemanticScholarClient):
        rec = _Recorder({})
        rec.install_post(client)

        result = client.fetch_recommendations(["a"])

        assert result == []

    def test_returns_empty_list_when_response_not_dict(self, client: SemanticScholarClient):
        rec = _Recorder([{"paperId": "p"}])  # unexpected list response
        rec.install_post(client)

        result = client.fetch_recommendations(["a"])

        assert result == []

    def test_positive_ids_are_copied_not_mutated(self, client: SemanticScholarClient):
        """Pass-by-reference protection: caller's list shouldn't end up in
        the request body verbatim, in case the caller later mutates it."""
        rec = _Recorder({"recommendedPapers": []})
        rec.install_post(client)
        ids = ["a", "b"]

        client.fetch_recommendations(ids)

        sent = rec.last["json_body"]["positivePaperIds"]
        assert sent == ids
        assert sent is not ids  # was list-copied


# ---------------------------------------------------------------------------
# fetch_recommendations_for_paper (PA-02)
# ---------------------------------------------------------------------------


class TestFetchRecommendationsForPaper:
    def test_basic_get_call_shape(self, client: SemanticScholarClient):
        rec = _Recorder(
            {
                "recommendedPapers": [
                    {"paperId": "fp-1", "title": "Forpaper 1"},
                    {"paperId": "fp-2", "title": "Forpaper 2"},
                ]
            }
        )
        rec.install_get_url(client)

        result = client.fetch_recommendations_for_paper("anchor")

        assert (
            rec.last["url"]
            == "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/anchor"
        )
        assert rec.last["req_type"] == "recommendations"
        params = rec.last["params"]
        assert params["fields"] == "paperId,title"
        assert params["limit"] == 100
        assert len(result) == 2
        assert result[0]["paperId"] == "fp-1"

    def test_custom_limit_and_fields(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": []})
        rec.install_get_url(client)

        client.fetch_recommendations_for_paper(
            "anchor", limit=50, fields="paperId,title,abstract",
        )

        params = rec.last["params"]
        assert params["limit"] == 50
        assert params["fields"] == "paperId,title,abstract"

    def test_url_includes_paper_id(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": []})
        rec.install_get_url(client)

        client.fetch_recommendations_for_paper("DOI:10.1234/abc")

        assert rec.last["url"].endswith("/forpaper/DOI:10.1234/abc")

    def test_unwraps_envelope(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": [{"paperId": "p"}]})
        rec.install_get_url(client)

        result = client.fetch_recommendations_for_paper("a")

        assert result == [{"paperId": "p"}]

    def test_returns_empty_list_when_envelope_missing(self, client: SemanticScholarClient):
        rec = _Recorder({})
        rec.install_get_url(client)

        assert client.fetch_recommendations_for_paper("a") == []


# ---------------------------------------------------------------------------
# fetch_author_papers (PA-03)
# ---------------------------------------------------------------------------


def _install_paginated_get(client: SemanticScholarClient, pages: list[Any]) -> list[dict[str, Any]]:
    """Monkey-patch ``client._http.get`` to return successive entries
    from ``pages`` on each call. Returns a list that the test can read
    after running to inspect captured args."""
    iter_pages = iter(pages)
    calls: list[dict[str, Any]] = []

    def fake_get(path: str, params: dict[str, Any] | None = None, *, req_type: str = "other"):
        calls.append({"path": path, "params": params, "req_type": req_type})
        try:
            return next(iter_pages)
        except StopIteration:  # safety net for over-call
            return {"data": []}

    client._http.get = fake_get  # type: ignore[method-assign]
    return calls


class TestFetchAuthorPapers:
    def test_basic_single_page(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [{"paperId": "p1", "title": "T1"}, {"paperId": "p2"}]})
        rec.install(client)

        result = client.fetch_author_papers("A1")

        assert rec.last["path"] == "/author/A1/papers"
        assert rec.last["req_type"] == "author_papers"
        params = rec.last["params"]
        assert params["fields"] == "paperId,title,year,venue,citationCount"
        assert params["limit"] == 100  # S2 page size, not the user's `limit`
        assert params["offset"] == 0
        assert len(result) == 2
        assert result[0]["paperId"] == "p1"

    def test_persists_to_cache_after_fetch(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [{"paperId": "p1"}]})
        rec.install(client)

        client.fetch_author_papers("A1")
        # Underlying Cache should now have an entry
        assert client._cache.get_author_papers("A1") == [{"paperId": "p1"}]

    def test_cache_hit_skips_http(self, client: SemanticScholarClient):
        # Pre-populate the cache directly
        client._cache.put_author_papers("A1", [{"paperId": "cached-1"}, {"paperId": "cached-2"}])
        rec = _Recorder({"data": [{"paperId": "fresh-from-api"}]})
        rec.install(client)

        result = client.fetch_author_papers("A1")

        assert result == [{"paperId": "cached-1"}, {"paperId": "cached-2"}]
        assert rec.calls == []  # http.get should NOT have been invoked

    def test_cache_hit_slices_to_limit(self, client: SemanticScholarClient):
        client._cache.put_author_papers(
            "A1", [{"paperId": f"p{i}"} for i in range(10)],
        )
        rec = _Recorder({"data": []})
        rec.install(client)

        result = client.fetch_author_papers("A1", limit=3)

        assert len(result) == 3
        assert [p["paperId"] for p in result] == ["p0", "p1", "p2"]
        assert rec.calls == []

    def test_paginates_when_first_page_full(self, client: SemanticScholarClient):
        full_page = [{"paperId": f"p{i}"} for i in range(100)]
        second_page = [{"paperId": f"p{i}"} for i in range(100, 150)]
        calls = _install_paginated_get(
            client, [{"data": full_page}, {"data": second_page}, {"data": []}],
        )

        result = client.fetch_author_papers("A1", limit=200)

        assert len(result) == 150
        # Two HTTP calls: offset=0 then offset=100
        assert len(calls) == 2
        assert calls[0]["params"]["offset"] == 0
        assert calls[1]["params"]["offset"] == 100

    def test_stops_paginating_when_limit_reached(self, client: SemanticScholarClient):
        full_page_1 = [{"paperId": f"p{i}"} for i in range(100)]
        full_page_2 = [{"paperId": f"p{i}"} for i in range(100, 200)]
        calls = _install_paginated_get(client, [{"data": full_page_1}, {"data": full_page_2}])

        result = client.fetch_author_papers("A1", limit=150)

        assert len(result) == 150
        assert result[-1]["paperId"] == "p149"
        # Two pages were needed; a third page would have been wasted
        assert len(calls) == 2

    def test_stops_paginating_on_short_page(self, client: SemanticScholarClient):
        """Less than a full page implies no more results — bail out."""
        calls = _install_paginated_get(
            client, [{"data": [{"paperId": f"p{i}"} for i in range(40)]}],
        )

        result = client.fetch_author_papers("A1", limit=200)

        assert len(result) == 40
        assert len(calls) == 1  # only one HTTP call

    def test_empty_author_returns_empty_list(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        result = client.fetch_author_papers("EmptyAuthor")

        assert result == []
        # Empty list is still cached so we don't re-fetch
        assert client._cache.get_author_papers("EmptyAuthor") == []

    def test_custom_fields_forwarded(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.fetch_author_papers("A1", fields="paperId,title,abstract")

        assert rec.last["params"]["fields"] == "paperId,title,abstract"

    def test_cache_hit_records_budget(self, client: SemanticScholarClient):
        """A cache hit should bump the cached counter, not the api counter."""
        client._cache.put_author_papers("A1", [{"paperId": "p1"}])
        rec = _Recorder({"data": []})
        rec.install(client)

        client.fetch_author_papers("A1")

        assert client._budget._s2_cache.get("author_papers", 0) == 1
        assert client._budget._s2_api.get("author_papers", 0) == 0

    def test_caches_truncated_result_when_limit_caps(self, client: SemanticScholarClient):
        """If pagination is capped by ``limit``, the cached list reflects
        what we actually fetched (not the full author corpus)."""
        full_page = [{"paperId": f"p{i}"} for i in range(100)]
        _install_paginated_get(client, [{"data": full_page}])

        client.fetch_author_papers("A1", limit=10)

        assert len(client._cache.get_author_papers("A1")) == 10


# ---------------------------------------------------------------------------
# PA-05: cache wiring for search_bulk + fetch_author_papers; negative
# coverage for the deliberately-uncached paths.
# ---------------------------------------------------------------------------


class TestSearchBulkCacheWiring:
    def test_second_identical_call_serves_from_cache(self, client: SemanticScholarClient):
        """Two identical calls → one HTTP call, one cache hit."""
        rec = _Recorder({"data": [{"paperId": "p1"}], "total": 1})
        rec.install(client)

        first = client.search_bulk("transformers", filters={"year": "2020-2025"})
        second = client.search_bulk("transformers", filters={"year": "2020-2025"})

        assert first == second
        assert len(rec.calls) == 1
        assert client._budget._s2_api.get("search", 0) == 0  # we monkey-patched http.get
        assert client._budget._s2_cache.get("search", 0) == 1

    def test_different_filters_miss_cache(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", filters={"year": "2020"})
        client.search_bulk("q", filters={"year": "2021"})

        assert len(rec.calls) == 2
        assert client._budget._s2_cache.get("search", 0) == 0

    def test_different_query_text_misses_cache(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("topic A")
        client.search_bulk("topic B")

        assert len(rec.calls) == 2

    def test_different_sort_misses_cache(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", sort="citationCount:desc")
        client.search_bulk("q", sort="year:desc")

        assert len(rec.calls) == 2

    def test_different_token_misses_cache(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", token=None)
        client.search_bulk("q", token="page-2")

        assert len(rec.calls) == 2

    def test_filter_dict_key_order_does_not_affect_hash(self, client: SemanticScholarClient):
        """sort_keys=True in the hash payload means dict ordering is irrelevant."""
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", filters={"year": "2020", "venue": "Nature"})
        client.search_bulk("q", filters={"venue": "Nature", "year": "2020"})

        assert len(rec.calls) == 1  # second is a cache hit

    def test_cache_persists_full_response_payload(self, client: SemanticScholarClient):
        """The cached value should be the EXACT response, including
        any S2-side metadata like ``token``, ``total``, etc."""
        rec = _Recorder({
            "data": [{"paperId": "p1"}, {"paperId": "p2"}],
            "total": 2,
            "token": "next-page-cursor",
        })
        rec.install(client)

        client.search_bulk("q")
        # Now read the cache directly to confirm everything round-tripped
        cached = client._cache.get_search_results(_query_hash_for("q", None, None, None))
        assert cached == {
            "data": [{"paperId": "p1"}, {"paperId": "p2"}],
            "total": 2,
            "token": "next-page-cursor",
        }


def _query_hash_for(
    query: str, filters: Any, sort: Any, token: Any,
) -> str:
    """Mirror the hash recipe used inside ``search_bulk`` so a test can
    verify the exact cache key without reaching into the SUT."""
    import hashlib
    import json as _json

    payload = {"q": query, "filters": filters, "sort": sort, "token": token}
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class TestUncachedSurfaces:
    """The spec deliberately leaves ``search_match`` and
    ``fetch_recommendations`` uncached because freshness matters there.
    Confirm two identical calls always reach the network."""

    def test_search_match_is_not_cached(self, client: SemanticScholarClient):
        rec = _Recorder({"data": [{"paperId": "match-1"}]})
        rec.install(client)

        client.search_match("Some Title")
        client.search_match("Some Title")

        assert len(rec.calls) == 2

    def test_fetch_recommendations_post_is_not_cached(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": [{"paperId": "rec-1"}]})
        rec.install_post(client)

        client.fetch_recommendations(["anchor"])
        client.fetch_recommendations(["anchor"])

        assert len(rec.calls) == 2

    def test_fetch_recommendations_for_paper_is_not_cached(self, client: SemanticScholarClient):
        rec = _Recorder({"recommendedPapers": [{"paperId": "fp-1"}]})
        rec.install_get_url(client)

        client.fetch_recommendations_for_paper("anchor")
        client.fetch_recommendations_for_paper("anchor")

        assert len(rec.calls) == 2
