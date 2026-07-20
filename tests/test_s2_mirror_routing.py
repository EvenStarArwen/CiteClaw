"""S2Http routing when a self-hosted graph mirror is configured:
graph traffic -> mirror (mirror key, un-throttled, 1000-row pages);
search + recommendations -> real S2 (S2 key, throttled)."""

from __future__ import annotations

import httpx
import pytest

from citeclaw.budget import BudgetTracker
from citeclaw.clients.s2.http import BASE_URL, S2Http, _normalize_graph_base
from citeclaw.config import Settings

MIRROR = "https://mirror.example.test"


def _mock_client(seen: list, payload=None, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload if payload is not None else {"ok": True})
    return httpx.Client(transport=httpx.MockTransport(handler), headers=headers or {})


def _swap_mirror(http, seen, payload=None):
    http._mirror_http = _mock_client(seen, payload, dict(http._mirror_http.headers))


def _swap_real(http, seen, payload=None):
    http._http = _mock_client(seen, payload, dict(http._http.headers))


@pytest.fixture()
def http():
    cfg = Settings(
        s2_api_key="real-s2-key",
        s2_mirror_url=MIRROR,
        s2_mirror_key="mirror-key",
        s2_requests_per_second=0.001,  # 1000s between real calls
    )
    h = S2Http(cfg, BudgetTracker())
    yield h
    h.close()


def test_normalize_graph_base():
    assert _normalize_graph_base("") == ""
    assert _normalize_graph_base("https://m.test") == "https://m.test/graph/v1"
    assert _normalize_graph_base("https://m.test/") == "https://m.test/graph/v1"
    assert _normalize_graph_base("https://m.test/graph/v1") == "https://m.test/graph/v1"


def test_graph_get_routes_to_mirror_with_mirror_key(http):
    seen: list[httpx.Request] = []
    _swap_mirror(http, seen)
    http.get("/paper/abc123", {"fields": "title"})
    assert seen[0].url.host == "mirror.example.test"
    assert str(seen[0].url).startswith(f"{MIRROR}/graph/v1/paper/abc123")
    assert seen[0].headers["x-api-key"] == "mirror-key"


def test_search_stays_on_real_s2(http):
    seen: list[httpx.Request] = []
    _swap_real(http, seen)
    # pre-warm the throttle window so the (huge) min interval doesn't stall the test
    http._last_request_time = 0.0
    http.get("/paper/search/match", {"query": "alpha"})
    assert seen[0].url.host == "api.semanticscholar.org"
    assert seen[0].headers["x-api-key"] == "real-s2-key"


def test_batch_urls_derive_from_mirror(http):
    assert http.batch_url == f"{MIRROR}/graph/v1/paper/batch"
    assert http.author_batch_url == f"{MIRROR}/graph/v1/author/batch"
    seen: list[httpx.Request] = []
    _swap_mirror(http, seen, payload=[])
    http.post(http.batch_url, json_body={"ids": ["x"]})
    assert seen[0].url.host == "mirror.example.test"


def test_mirror_calls_skip_throttle(http):
    seen: list[httpx.Request] = []
    _swap_mirror(http, seen)
    # with s2_rps=0.001 two throttled calls would need ~1000s; these must not block
    import time
    t0 = time.monotonic()
    http.get("/paper/a")
    http.get("/paper/b")
    assert time.monotonic() - t0 < 2.0
    assert len(seen) == 2


def test_mirror_page_size(http):
    assert http._page_size == 1000
    seen: list[httpx.Request] = []
    _swap_mirror(http, seen, payload={"data": []})
    http.paginate("abc", "citations", fields="citingPaper.paperId")
    assert seen[0].url.params["limit"] == "1000"


def test_no_mirror_defaults_unchanged():
    cfg = Settings(s2_api_key="k")
    h = S2Http(cfg, BudgetTracker())
    try:
        assert h.graph_base == BASE_URL
        assert h.batch_url.endswith("/graph/v1/paper/batch")
        assert h._page_size == 100
        assert h._mirror_http is None
    finally:
        h.close()


def test_recommendations_full_url_stays_real(http):
    seen: list[httpx.Request] = []
    _swap_real(http, seen, payload={"recommendedPapers": []})
    http._last_request_time = 0.0
    http.get_url("https://api.semanticscholar.org/recommendations/v1/papers/forpaper/x",
                 {"limit": 5})
    assert seen[0].url.host == "api.semanticscholar.org"


def _edge_pages_transport(total):
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", 0))
        limit = int(request.url.params.get("limit", 100))
        items = [{"citingPaper": {"paperId": f"p{i}"}}
                 for i in range(offset, min(offset + limit, total))]
        return httpx.Response(200, json={"offset": offset, "data": items})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_windowed_pagination_matches_sequential_output(http):
    http._mirror_http = _edge_pages_transport(2350)
    calls = []
    out = http.paginate("abc", "citations", fields="citingPaper.paperId",
                        progress_cb=calls.append)
    assert [e["citingPaper"]["paperId"] for e in out] == [f"p{i}" for i in range(2350)]
    assert calls == [1000, 1000, 350]  # same order + sizes as the serial walk


def test_windowed_pagination_max_items(http):
    http._mirror_http = _edge_pages_transport(5000)
    out = http.paginate("abc", "citations", fields="citingPaper.paperId",
                        max_items=1500)
    assert [e["citingPaper"]["paperId"] for e in out] == [f"p{i}" for i in range(1500)]


def test_windowed_pagination_empty(http):
    http._mirror_http = _edge_pages_transport(0)
    calls = []
    out = http.paginate("abc", "citations", fields="citingPaper.paperId",
                        progress_cb=calls.append)
    assert out == [] and calls == []


def test_mirror_key_forbidden_in_yaml():
    from citeclaw.config import _normalize_yaml
    with pytest.raises(ValueError):
        _normalize_yaml({"s2_mirror_key": "sneaky"})
    assert _normalize_yaml({"s2_mirror_url": MIRROR})["s2_mirror_url"] == MIRROR
