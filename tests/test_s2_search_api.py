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
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings


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

    def test_sort_relevance_dropped_silently(self, client: SemanticScholarClient):
        """``sort=relevance`` is the LLM agent's default guess but is invalid
        for ``/paper/search/bulk`` (the bulk endpoint only supports paperId /
        publicationDate / citationCount). The sanitiser drops it rather
        than letting S2 400 the whole call."""
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", sort="relevance")

        params = rec.last["params"]
        assert "sort" not in params

    def test_sort_invalid_field_dropped(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", sort="hIndex:desc")

        params = rec.last["params"]
        assert "sort" not in params

    def test_sort_invalid_direction_keeps_field_only(self, client: SemanticScholarClient):
        """Bad direction (e.g. ``citationCount:high``) is not worth a 400 —
        keep the field, drop the direction so we still get a sorted result."""
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", sort="citationCount:high")

        params = rec.last["params"]
        assert params["sort"] == "citationCount"

    def test_sort_valid_field_passes_through(self, client: SemanticScholarClient):
        rec = _Recorder({"data": []})
        rec.install(client)

        client.search_bulk("q", sort="publicationDate:asc")

        params = rec.last["params"]
        assert params["sort"] == "publicationDate:asc"

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

# ---------------------------------------------------------------------------
# enrich_batch cache wiring (PH-09)
# ---------------------------------------------------------------------------


class TestEnrichBatchCache:
    """``enrich_batch`` is called from ExpandForward / ExpandBackward / the
    expansion family on EVERY citer / reference / search hit. The original
    implementation always called the network even when every id was already
    cached, which made second-run wall time identical to first-run wall
    time on a large pipeline. PH-09 added cache-first hydration; this
    suite confirms it.
    """

    def _seed_metadata(self, client: SemanticScholarClient, paper_id: str) -> None:
        """Plant a metadata row directly in the cache so subsequent
        enrich_batch calls can read it without a network round-trip."""
        client._cache.put_metadata(paper_id, {
            "paperId": paper_id,
            "title": f"Title of {paper_id}",
            "abstract": f"Abstract of {paper_id}",
            "year": 2024,
            "venue": "Nature",
            "citationCount": 42,
        })

    def test_fully_cached_batch_makes_zero_network_calls(
        self, client: SemanticScholarClient,
    ):
        """When every id is already in the cache, enrich_batch must NOT
        hit the network. This is the regression test for the bug where
        a second run of the same pipeline re-fetched every paper."""
        for pid in ["p1", "p2", "p3"]:
            self._seed_metadata(client, pid)

        rec = _Recorder([])  # would 'fail' if any network call happened
        rec.install_post(client)

        result = client.enrich_batch([
            {"paper_id": "p1"},
            {"paper_id": "p2"},
            {"paper_id": "p3"},
        ])

        assert len(rec.calls) == 0  # zero network calls
        assert len(result) == 3
        assert {r.paper_id for r in result} == {"p1", "p2", "p3"}
        assert all(r.title for r in result)
        assert all(r.abstract for r in result)

    def test_fully_cached_batch_records_cache_hits_in_budget(
        self, client: SemanticScholarClient,
    ):
        """The cache layer's get_metadata bumps BudgetTracker.s2_cache;
        confirm enrich_batch flows through that wrapper so the run-end
        summary correctly attributes the saved calls."""
        for pid in ["p1", "p2"]:
            self._seed_metadata(client, pid)

        before_hits = client._budget.s2_cache_hits
        before_api = client._budget.s2_requests
        client.enrich_batch([{"paper_id": "p1"}, {"paper_id": "p2"}])

        # 2 cache hits, 0 API calls.
        assert client._budget.s2_cache_hits == before_hits + 2
        assert client._budget.s2_requests == before_api

    def test_partially_cached_batch_only_fetches_misses(
        self, client: SemanticScholarClient,
    ):
        """Mixed: 2 papers in cache, 1 missing. Only the 1 missing
        should trigger a network call; the 2 cached should be served
        locally."""
        self._seed_metadata(client, "p1")
        self._seed_metadata(client, "p2")
        # p3 is NOT in the cache

        rec = _Recorder([
            {
                "paperId": "p3",
                "title": "Fetched p3",
                "abstract": "from network",
                "year": 2025,
                "venue": "Cell",
                "citationCount": 7,
            },
        ])
        rec.install_post(client)

        result = client.enrich_batch([
            {"paper_id": "p1"},
            {"paper_id": "p2"},
            {"paper_id": "p3"},
        ])

        # Exactly one network call, and its body included only the
        # missing id (not the two cached ones).
        assert len(rec.calls) == 1
        assert rec.last["json_body"]["ids"] == ["p3"]
        assert {r.paper_id for r in result} == {"p1", "p2", "p3"}

    def test_uncached_batch_falls_back_to_network(
        self, client: SemanticScholarClient,
    ):
        """Cold cache: every paper is a miss → one batch network call
        with all ids. The cache is then populated for next time."""
        rec = _Recorder([
            {"paperId": "p1", "title": "T1", "abstract": "A1"},
            {"paperId": "p2", "title": "T2", "abstract": "A2"},
        ])
        rec.install_post(client)

        client.enrich_batch([{"paper_id": "p1"}, {"paper_id": "p2"}])

        assert len(rec.calls) == 1
        assert sorted(rec.last["json_body"]["ids"]) == ["p1", "p2"]
        # Cache was populated.
        assert client._cache.get_metadata("p1") is not None
        assert client._cache.get_metadata("p2") is not None

    def test_second_call_after_first_is_zero_network(
        self, client: SemanticScholarClient,
    ):
        """End-to-end: a second call to enrich_batch with the same ids
        should be a no-op against the network because the first call
        populated the cache. This is the regression for the original
        complaint."""
        rec = _Recorder([
            {"paperId": "p1", "title": "T1", "abstract": "A1"},
            {"paperId": "p2", "title": "T2", "abstract": "A2"},
        ])
        rec.install_post(client)

        client.enrich_batch([{"paper_id": "p1"}, {"paper_id": "p2"}])
        assert len(rec.calls) == 1  # first call hits the network

        client.enrich_batch([{"paper_id": "p1"}, {"paper_id": "p2"}])
        assert len(rec.calls) == 1  # second call is fully cached


# ---------------------------------------------------------------------------
# HTTP retry policy: _is_retryable
# ---------------------------------------------------------------------------


class TestHttpRetryPolicy:
    """The S2 HTTP layer retries transient failures (5xx, 429, transport)
    but must NOT retry permanent client errors (400, 401, 403, 404). Each
    retry of a permanent failure wastes 6 requests against the 1-rps S2
    budget while the caller waits 30+ seconds for backoff to give up.

    Tests :func:`citeclaw.clients.s2.http._is_retryable` directly because
    it's a pure function — exercising the full retry decorator would
    pull in tenacity timing and add minutes to the test suite.
    """

    def _make_status_error(self, status: int) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/foo")
        response = httpx.Response(status, request=request)
        return httpx.HTTPStatusError(f"{status}", request=request, response=response)

    def test_5xx_is_retryable(self):
        from citeclaw.clients.s2.http import _is_retryable
        assert _is_retryable(self._make_status_error(500)) is True
        assert _is_retryable(self._make_status_error(502)) is True
        assert _is_retryable(self._make_status_error(503)) is True
        assert _is_retryable(self._make_status_error(504)) is True

    def test_429_is_retryable(self):
        """Rate-limit responses should retry — the backoff handles the wait."""
        from citeclaw.clients.s2.http import _is_retryable
        assert _is_retryable(self._make_status_error(429)) is True

    def test_4xx_other_than_429_is_not_retryable(self):
        """400 / 401 / 403 / 404 are permanent — retrying wastes our 1-rps
        budget. The original 400 from a bad sort key was getting retried
        6 times before tenacity gave up; that's the bug this test guards."""
        from citeclaw.clients.s2.http import _is_retryable
        assert _is_retryable(self._make_status_error(400)) is False
        assert _is_retryable(self._make_status_error(401)) is False
        assert _is_retryable(self._make_status_error(403)) is False
        assert _is_retryable(self._make_status_error(404)) is False
        assert _is_retryable(self._make_status_error(422)) is False

    def test_transport_error_is_retryable(self):
        from citeclaw.clients.s2.http import _is_retryable
        assert _is_retryable(httpx.ConnectError("dns failed")) is True
        assert _is_retryable(httpx.ReadTimeout("read timed out")) is True

    def test_arbitrary_exceptions_are_not_retryable(self):
        from citeclaw.clients.s2.http import _is_retryable
        assert _is_retryable(ValueError("bad input")) is False
        assert _is_retryable(KeyError("nope")) is False
