"""Tests for the OpenAlex abstract + references fallback client.

Drives :class:`OpenAlexClient` against a stub ``httpx.Client`` so no
network traffic is required. Covers abstract reconstruction from
``abstract_inverted_index``, DOI normalisation on the wire, and the
graceful-None behaviour on 404 / HTTP errors.
"""

from __future__ import annotations

import httpx
import pytest

from citeclaw.clients.openalex import (
    OpenAlexClient,
    _reconstruct_abstract,
)
from citeclaw.config import Settings


class _Canned:
    """Stub httpx.Client that matches the tiny surface OpenAlexClient uses."""

    def __init__(self, responses: dict[str, dict | int]):
        """responses maps path -> dict (JSON) or int (status code)."""
        self._responses = responses
        self.requests: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None):
        self.requests.append((url, params or {}))
        # Strip host prefix; keep path+anything after
        base = "https://api.openalex.org"
        path = url[len(base):] if url.startswith(base) else url
        body = self._responses.get(path)
        if isinstance(body, int):
            return _FakeResp(body, {})
        if body is None:
            return _FakeResp(404, {})
        return _FakeResp(200, body)

    def close(self):
        pass


class _FakeResp:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://api.openalex.org"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)


def _client_with(responses: dict) -> OpenAlexClient:
    cfg = Settings(openalex_rps=1000.0)  # effectively un-throttled
    client = OpenAlexClient(cfg)
    client._http = _Canned(responses)
    return client


class TestReconstructAbstract:
    def test_simple(self):
        idx = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(idx) == "Hello world"

    def test_order_preserved_across_positions(self):
        idx = {"world": [1], "Hello": [0], "!": [2]}
        assert _reconstruct_abstract(idx) == "Hello world !"

    def test_repeated_words(self):
        idx = {"the": [0, 2], "cat": [1, 3]}
        assert _reconstruct_abstract(idx) == "the cat the cat"

    def test_none_input(self):
        assert _reconstruct_abstract(None) is None

    def test_empty_dict(self):
        assert _reconstruct_abstract({}) is None

    def test_malformed_positions_ignored(self):
        idx = {"OK": [0], "bad": "nope", "also_bad": [-1]}
        # Only "OK" at pos 0 is valid. "also_bad" has negative position.
        assert _reconstruct_abstract(idx) == "OK"


class TestFetchAbstractByDoi:
    def test_successful_fetch(self):
        work = {
            "id": "https://openalex.org/W123",
            "abstract_inverted_index": {"hello": [0], "world": [1]},
        }
        c = _client_with({"/works/doi:10.1038/nature12345": work})
        abs_ = c.fetch_abstract_by_doi("10.1038/nature12345")
        assert abs_ == "hello world"

    def test_strips_url_prefix(self):
        work = {"abstract_inverted_index": {"hi": [0]}}
        c = _client_with({"/works/doi:10.1038/abc": work})
        assert c.fetch_abstract_by_doi("https://doi.org/10.1038/abc") == "hi"

    def test_returns_none_on_404(self):
        c = _client_with({})  # 404 for any path
        assert c.fetch_abstract_by_doi("10.9999/nope") is None

    def test_returns_none_on_missing_abstract_field(self):
        c = _client_with({"/works/doi:10.1/abcde": {"id": "x"}})  # no inverted_index
        assert c.fetch_abstract_by_doi("10.1/abcde") is None

    def test_empty_doi_returns_none(self):
        c = _client_with({})
        assert c.fetch_abstract_by_doi("") is None
        assert c.fetch_abstract_by_doi(None) is None


class TestFetchReferencesByDoi:
    def test_resolves_referenced_works_to_dois(self):
        work = {"referenced_works": ["https://openalex.org/W11", "https://openalex.org/W22"]}
        responses = {
            "/works/doi:10.1/parent": work,
            "/works/W11": {"doi": "https://doi.org/10.1/child1"},
            "/works/W22": {"doi": "https://doi.org/10.1/child2"},
        }
        c = _client_with(responses)
        refs = c.fetch_references_by_doi("10.1/parent")
        assert refs == ["10.1/child1", "10.1/child2"]

    def test_skips_entries_missing_doi(self):
        work = {"referenced_works": ["https://openalex.org/W11"]}
        responses = {
            "/works/doi:10.1/parent": work,
            "/works/W11": {"id": "x"},  # no doi
        }
        c = _client_with(responses)
        assert c.fetch_references_by_doi("10.1/parent") == []

    def test_empty_references(self):
        c = _client_with({"/works/doi:10.1/p": {}})
        assert c.fetch_references_by_doi("10.1/p") == []


class TestHeaders:
    def test_email_included_in_user_agent(self):
        cfg = Settings(openalex_email="user@example.com")
        c = OpenAlexClient(cfg)
        try:
            ua = c._http.headers.get("User-Agent", "")
            assert "user@example.com" in ua
            assert "CiteClaw" in ua
        finally:
            c.close()

    def test_api_key_included_as_bearer(self):
        cfg = Settings(openalex_api_key="secret123")
        c = OpenAlexClient(cfg)
        try:
            auth = c._http.headers.get("Authorization", "")
            assert auth == "Bearer secret123"
        finally:
            c.close()

    def test_no_auth_when_empty(self):
        cfg = Settings()
        c = OpenAlexClient(cfg)
        try:
            assert "Authorization" not in c._http.headers
        finally:
            c.close()
