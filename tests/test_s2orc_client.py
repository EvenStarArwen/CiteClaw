"""Unit tests for the S2ORC mirror client (offline, httpx MockTransport)."""

from __future__ import annotations

import json
from urllib.parse import unquote

import httpx
import pytest

from citeclaw.clients.s2orc import (
    S2orcClient,
    build_s2orc_client,
    mirror_id_for,
)

# corpusid 123 is "in S2ORC"; everything else is a miss.
_HIT = {
    "corpusid": 123, "text": "FULL BODY TEXT", "license": "CCBY",
    "status": "GOLD", "openAccessUrl": "https://ex.org/p.pdf",
    "annotations": {"paragraph": "[]"},
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = unquote(request.url.path)
    if path.endswith("/paper/batch"):
        body = json.loads(request.content)
        return httpx.Response(200, json=[
            {"corpusid": 123, "license": "CCBY"} if i == "CorpusId:123" else None
            for i in body["ids"]
        ])
    ident = path.split("/paper/", 1)[-1]
    if ident in ("CorpusId:123", "DOI:10.1/present"):
        return httpx.Response(200, json=_HIT)
    return httpx.Response(404, json={"error": "not found in s2orc"})


class _Paper:
    def __init__(self, pid, ext):
        self.paper_id = pid
        self.external_ids = ext


class _FakeCache:
    def __init__(self):
        self.store: dict = {}

    def get_full_text(self, pid):
        return self.store.get(pid)

    def put_full_text(self, pid, *, text=None, error=None):
        self.store[pid] = {"text": text, "error": error}


def _client():
    return S2orcClient("https://mirror.example.com", "k",
                       transport=httpx.MockTransport(_handler))


def test_mirror_id_priority():
    assert mirror_id_for(_Paper("x", {"CorpusId": "123", "DOI": "10.1/a"})) == "CorpusId:123"
    assert mirror_id_for(_Paper("x", {"DOI": "10.1/a", "ArXiv": "2101.1"})) == "DOI:10.1/a"
    assert mirror_id_for(_Paper("x", {"ArXiv": "2101.1"})) == "ARXIV:2101.1"
    assert mirror_id_for(_Paper("x", {"PubMed": "999"})) == "PMID:999"
    assert mirror_id_for(_Paper("x", {})) is None
    assert mirror_id_for(_Paper("x", None)) is None


def test_fetch_hit_writes_through_cache():
    cache = _FakeCache()
    res = _client().fetch_full_text(_Paper("pA", {"CorpusId": "123"}), cache=cache)
    assert res is not None
    assert res.text == "FULL BODY TEXT" and res.source == "s2orc"
    assert res.corpusid == 123 and res.license == "CCBY"
    assert res.chars == len("FULL BODY TEXT")
    # written through to the shared cache
    assert cache.store["pA"]["text"] == "FULL BODY TEXT"


def test_fetch_second_call_served_from_cache():
    cache = _FakeCache()
    cache.store["pA"] = {"text": "CACHED BODY", "error": None}
    res = _client().fetch_full_text(_Paper("pA", {"CorpusId": "123"}), cache=cache)
    assert res.source == "cache" and res.text == "CACHED BODY"


def test_miss_returns_none_and_does_not_poison_cache():
    cache = _FakeCache()
    res = _client().fetch_full_text(_Paper("pB", {"CorpusId": "999"}), cache=cache)
    assert res is None
    assert "pB" not in cache.store          # no error row written -> PDF path stays free


def test_no_usable_id_is_a_miss():
    assert _client().fetch_full_text(_Paper("pC", {}), cache=_FakeCache()) is None


def test_resolve_by_doi():
    res = _client().fetch_full_text(_Paper("pD", {"DOI": "10.1/present"}))
    assert res is not None and res.corpusid == 123


def test_availability_batch():
    papers = [
        _Paper("pA", {"CorpusId": "123"}),
        _Paper("pB", {"CorpusId": "999"}),
        _Paper("pC", {}),                    # no id -> False, never sent
    ]
    avail = _client().availability(papers)
    assert avail == {"pA": True, "pB": False, "pC": False}


def test_build_returns_none_without_url():
    class S:
        s2orc_mirror_url = ""
        s2orc_mirror_key = ""
    assert build_s2orc_client(S()) is None

    class S2:
        s2orc_mirror_url = "https://m.example.com"
        s2orc_mirror_key = "k"
    c = build_s2orc_client(S2())
    assert isinstance(c, S2orcClient) and c.base.endswith("/s2orc/v1")
