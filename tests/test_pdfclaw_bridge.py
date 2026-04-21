"""Tier 1 offline tests for :mod:`citeclaw.clients.pdfclaw_bridge`.

The bridge composes three optional layers (cache → HTTP → pdfclaw
browser recipes); this file mocks each layer so the bridge runs without
a real cache, real network, or the optional pdfclaw package installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citeclaw.clients import pdfclaw_bridge as bridge_mod
from citeclaw.clients.pdfclaw_bridge import PdfClawBridge


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeCache:
    """In-memory stand-in for :class:`citeclaw.cache.Cache`'s full-text rows."""

    def __init__(self):
        self._rows: dict[str, dict] = {}

    def get_full_text(self, paper_id):
        return self._rows.get(paper_id)

    def put_full_text(self, paper_id, *, text=None, error=None):
        self._rows[paper_id] = {"text": text, "error": error}


def _make_paper(paper_id="p1", *, doi=None, arxiv=None, pdf_url=None):
    """Minimal PaperRecord-shaped object."""
    ext: dict[str, str] = {}
    if doi:
        ext["DOI"] = doi
    if arxiv:
        ext["ArXiv"] = arxiv
    return SimpleNamespace(
        paper_id=paper_id,
        external_ids=ext,
        pdf_url=pdf_url,
    )


# ---------------------------------------------------------------------------
# fetch_text — three-layer cache → HTTP → pdfclaw fall-through
# ---------------------------------------------------------------------------


class TestFetchTextCachePath:
    def test_returns_cached_text_when_present(self):
        cache = _FakeCache()
        cache.put_full_text("p1", text="hello world")
        b = PdfClawBridge(cache)
        try:
            assert b.fetch_text(_make_paper("p1")) == "hello world"
        finally:
            b.close()

    def test_truncates_cached_text_to_max_chars(self):
        cache = _FakeCache()
        cache.put_full_text("p1", text="abcdefghij")
        b = PdfClawBridge(cache, max_text_chars=4)
        try:
            assert b.fetch_text(_make_paper("p1")) == "abcd"
        finally:
            b.close()

    def test_cached_parse_failed_short_circuits_to_none(self):
        """Cached parse_failed/too_large skip the pdfclaw retry — won't help."""
        cache = _FakeCache()
        cache.put_full_text("p1", error="parse_failed")
        b = PdfClawBridge(cache)
        try:
            assert b.fetch_text(_make_paper("p1")) is None
        finally:
            b.close()

    def test_returns_none_when_no_paper_id(self):
        b = PdfClawBridge(_FakeCache())
        try:
            assert b.fetch_text(_make_paper(paper_id="")) is None
        finally:
            b.close()


class TestFetchTextHttpPath:
    def test_http_success_caches_and_returns_text(self, monkeypatch):
        cache = _FakeCache()
        monkeypatch.setattr(bridge_mod, "download_pdf_bytes",
                            lambda http, url: (b"%PDF-fake", None))
        monkeypatch.setattr(bridge_mod, "parse_pdf_bytes",
                            lambda body, max_chars=None: "extracted body")
        b = PdfClawBridge(cache)
        try:
            paper = _make_paper("p1", pdf_url="https://example.com/x.pdf")
            result = b.fetch_text(paper)
        finally:
            b.close()
        assert result == "extracted body"
        # Cached for the next call
        assert cache._rows["p1"]["text"] == "extracted body"

    def test_http_failure_falls_through_then_caches_download_failed(self, monkeypatch):
        cache = _FakeCache()
        monkeypatch.setattr(bridge_mod, "download_pdf_bytes",
                            lambda http, url: (None, "download_failed"))
        # Force pdfclaw layer disabled so we exit cleanly to the failure cache
        b = PdfClawBridge(cache)
        b._pdfclaw_available = False
        try:
            paper = _make_paper("p1", pdf_url="https://example.com/x.pdf")
            assert b.fetch_text(paper) is None
        finally:
            b.close()
        assert cache._rows["p1"]["error"] == "download_failed"


# ---------------------------------------------------------------------------
# _extract_doi — DOI takes priority, ArXiv synthesises a 10.48550/arXiv.<id>
# ---------------------------------------------------------------------------


class TestExtractDoi:
    def test_returns_doi_when_present(self):
        b = PdfClawBridge(_FakeCache())
        try:
            assert b._extract_doi(_make_paper(doi="10.1/abc")) == "10.1/abc"
        finally:
            b.close()

    def test_falls_back_to_arxiv(self):
        b = PdfClawBridge(_FakeCache())
        try:
            assert b._extract_doi(_make_paper(arxiv="2301.12345")) == "10.48550/arXiv.2301.12345"
        finally:
            b.close()

    def test_doi_wins_over_arxiv(self):
        b = PdfClawBridge(_FakeCache())
        try:
            paper = _make_paper(doi="10.1/abc", arxiv="2301.12345")
            assert b._extract_doi(paper) == "10.1/abc"
        finally:
            b.close()

    def test_returns_none_when_neither_present(self):
        b = PdfClawBridge(_FakeCache())
        try:
            assert b._extract_doi(_make_paper()) is None
        finally:
            b.close()


# ---------------------------------------------------------------------------
# _extract_text — body_text wins over pdf_bytes, both can be empty
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_prefers_body_text(self, monkeypatch):
        # parse_pdf_bytes shouldn't be called when body_text is present
        called = {"n": 0}
        def _pp(body, max_chars=None):
            called["n"] += 1
            return "should not be used"
        monkeypatch.setattr(bridge_mod, "parse_pdf_bytes", _pp)
        b = PdfClawBridge(_FakeCache())
        try:
            result = SimpleNamespace(body_text="from body_text", pdf_bytes=b"%PDF")
            assert b._extract_text(result) == "from body_text"
            assert called["n"] == 0
        finally:
            b.close()

    def test_falls_back_to_parsed_pdf_bytes(self, monkeypatch):
        monkeypatch.setattr(bridge_mod, "parse_pdf_bytes",
                            lambda body, max_chars=None: "parsed text")
        b = PdfClawBridge(_FakeCache())
        try:
            result = SimpleNamespace(body_text=None, pdf_bytes=b"%PDF-fake")
            assert b._extract_text(result) == "parsed text"
        finally:
            b.close()

    def test_returns_none_when_neither_field_set(self):
        b = PdfClawBridge(_FakeCache())
        try:
            result = SimpleNamespace(body_text=None, pdf_bytes=None)
            assert b._extract_text(result) is None
        finally:
            b.close()

    def test_truncates_to_max_chars(self):
        b = PdfClawBridge(_FakeCache(), max_text_chars=4)
        try:
            result = SimpleNamespace(body_text="abcdefghij", pdf_bytes=None)
            assert b._extract_text(result) == "abcd"
        finally:
            b.close()


# ---------------------------------------------------------------------------
# _bump_failures — 3-strike rule suppresses the recipe for the rest of the run
# ---------------------------------------------------------------------------


class TestBumpFailures:
    def test_first_two_failures_dont_suppress(self):
        b = PdfClawBridge(_FakeCache())
        try:
            b._bump_failures("nature")
            b._bump_failures("nature")
            assert "nature" not in b._auth_failed
            assert b._consecutive_failures["nature"] == 2
        finally:
            b.close()

    def test_third_failure_adds_to_auth_failed(self):
        b = PdfClawBridge(_FakeCache())
        try:
            for _ in range(3):
                b._bump_failures("nature")
            assert "nature" in b._auth_failed
        finally:
            b.close()

    def test_independent_counters_per_recipe(self):
        b = PdfClawBridge(_FakeCache())
        try:
            b._bump_failures("nature")
            b._bump_failures("springer")
            b._bump_failures("springer")
            assert b._consecutive_failures == {"nature": 1, "springer": 2}
            assert "nature" not in b._auth_failed
            assert "springer" not in b._auth_failed
        finally:
            b.close()


# ---------------------------------------------------------------------------
# close — defensive against partially-init / re-close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_is_idempotent(self):
        b = PdfClawBridge(_FakeCache())
        b.close()
        b.close()  # second close should not raise

    def test_close_swallows_browser_exit_failure(self, caplog):
        """Even if the browser exit raises, close() must not propagate."""

        class _BadCtx:
            def __exit__(self, *exc):
                raise RuntimeError("simulated browser exit failure")

        b = PdfClawBridge(_FakeCache())
        b._browser_ctx_manager = _BadCtx()
        # Must not raise:
        b.close()
        # Browser handles cleared
        assert b._browser_ctx_manager is None
        assert b._browser_page is None

    def test_close_swallows_http_close_failure(self):
        """Even if http.close() raises, close() must not propagate."""

        class _BadHttp:
            def close(self):
                raise RuntimeError("simulated http close failure")

        b = PdfClawBridge(_FakeCache())
        b._http = _BadHttp()
        # Must not raise:
        b.close()


# ---------------------------------------------------------------------------
# _ensure_pdfclaw — gracefully handles the missing-package path
# ---------------------------------------------------------------------------


class TestEnsurePdfclaw:
    def test_caches_negative_result_on_import_error(self, monkeypatch):
        """When pdfclaw isn't installed, the negative result is memoised."""
        b = PdfClawBridge(_FakeCache())
        try:
            # Force the import inside _ensure_pdfclaw to fail by stubbing
            # find_module via sys.modules — easier: patch importlib.
            import builtins
            original_import = builtins.__import__

            def _no_pdfclaw(name, *a, **kw):
                if name.startswith("pdfclaw"):
                    raise ImportError("simulated missing pdfclaw")
                return original_import(name, *a, **kw)

            monkeypatch.setattr(builtins, "__import__", _no_pdfclaw)

            assert b._ensure_pdfclaw() is False
            # Second call: short-circuits without re-raising
            assert b._ensure_pdfclaw() is False
        finally:
            b.close()
