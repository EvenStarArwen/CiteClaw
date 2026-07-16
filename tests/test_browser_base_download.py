"""Unit tests for the two-tier download in :class:`BrowserRecipeBase`.

The reliability fix these cover: when the bare ``context.request.get``
is blocked by publisher anti-bot (Cloudflare/Akamai) and returns a 403
or a non-PDF challenge page, the recipe must fall back to an in-page
``fetch`` that runs in the live browser JS context and clears the check.

No real browser is involved — a tiny fake Playwright ``Page`` drives the
two tiers directly.
"""

from __future__ import annotations

import base64

from pdfclaw.publishers._browser_base import BrowserRecipeBase

_PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n%%EOF\n"
_PDF_B64 = base64.b64encode(_PDF).decode()


class _FakeResponse:
    def __init__(self, *, ok: bool, status: int, url: str, body: bytes = b""):
        self.ok = ok
        self.status = status
        self.url = url
        self._body = body

    def body(self) -> bytes:
        return self._body


class _FakeRequestContext:
    """Stand-in for ``page.context.request``."""

    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises

    def get(self, url, timeout=None, headers=None):  # noqa: ARG002
        if self._raises is not None:
            raise self._raises
        return self._response


class _FakeContext:
    def __init__(self, request_ctx):
        self.request = request_ctx


class _FakePage:
    """Minimal fake with just what _download_href / _fetch_via_page touch."""

    def __init__(self, *, url, request_response=None, request_raises=None,
                 evaluate_return=None, evaluate_raises=None):
        self.url = url
        self.context = _FakeContext(
            _FakeRequestContext(request_response, request_raises)
        )
        self._evaluate_return = evaluate_return
        self._evaluate_raises = evaluate_raises

    def evaluate(self, _js, _arg=None):
        if self._evaluate_raises is not None:
            raise self._evaluate_raises
        return self._evaluate_return


def _recipe() -> BrowserRecipeBase:
    return BrowserRecipeBase()


# ---------------------------------------------------------------------------
# _download_href — the two-tier strategy
# ---------------------------------------------------------------------------


def test_tier1_context_request_success():
    """Fast path: context request returns a PDF — no in-page fetch needed."""
    page = _FakePage(
        url="https://pub.example/article/1",
        request_response=_FakeResponse(ok=True, status=200, url="x", body=_PDF),
        evaluate_return=None,  # must not be consulted
    )
    body, method, sso = _recipe()._download_href(page, "https://pub.example/pdf/1")
    assert body == _PDF
    assert method == "context_request"
    assert sso is None


def test_tier2_page_fetch_recovers_from_403():
    """The core fix: bare request 403s (bot block), in-page fetch wins."""
    page = _FakePage(
        url="https://academic.oup.com/nar/article/1",
        request_response=_FakeResponse(ok=False, status=403, url="https://academic.oup.com/nar/article-pdf/x.pdf"),
        evaluate_return=_PDF_B64,
    )
    body, method, sso = _recipe()._download_href(
        page, "https://academic.oup.com/nar/article-pdf/x.pdf"
    )
    assert body == _PDF
    assert method == "page_fetch"
    assert sso is None


def test_tier2_recovers_from_non_pdf_challenge_html():
    """Context request returns 200 but an HTML challenge page; page fetch wins."""
    page = _FakePage(
        url="https://www.mdpi.com/1/1",
        request_response=_FakeResponse(ok=True, status=200, url="x",
                                       body=b"<html>Just a moment...</html>"),
        evaluate_return=_PDF_B64,
    )
    body, method, sso = _recipe()._download_href(page, "https://www.mdpi.com/1/1/pdf")
    assert body == _PDF
    assert method == "page_fetch"


def test_sso_bounce_is_surfaced():
    """A not-ok response redirected to an SSO host returns the sso marker."""
    page = _FakePage(
        url="https://pub.example/article/1",
        request_response=_FakeResponse(ok=False, status=302,
                                       url="https://idp.institution.edu/login"),
        evaluate_return=None,
    )
    body, method, sso = _recipe()._download_href(page, "https://pub.example/pdf/1")
    assert body is None
    assert method == "sso"
    assert "idp." in sso


def test_both_tiers_fail_returns_reason():
    """Bare request 403 and in-page fetch yields nothing -> failure + reason."""
    page = _FakePage(
        url="https://pub.example/article/1",
        request_response=_FakeResponse(ok=False, status=403, url="https://pub.example/pdf/1"),
        evaluate_return=None,
    )
    body, method, sso = _recipe()._download_href(page, "https://pub.example/pdf/1")
    assert body is None
    assert sso is None
    assert "403" in method


def test_context_request_exception_falls_through_to_page_fetch():
    """A raised context.request.get still lets the in-page fetch recover."""
    page = _FakePage(
        url="https://pub.example/article/1",
        request_raises=RuntimeError("connection reset"),
        evaluate_return=_PDF_B64,
    )
    body, method, _ = _recipe()._download_href(page, "https://pub.example/pdf/1")
    assert body == _PDF
    assert method == "page_fetch"


# ---------------------------------------------------------------------------
# _fetch_via_page — the JS-context fetch helper
# ---------------------------------------------------------------------------


def test_fetch_via_page_decodes_pdf():
    page = _FakePage(url="x", evaluate_return=_PDF_B64)
    assert _recipe()._fetch_via_page(page, "https://x/y.pdf") == _PDF


def test_fetch_via_page_rejects_non_pdf_bytes():
    page = _FakePage(url="x", evaluate_return=base64.b64encode(b"<html>nope").decode())
    assert _recipe()._fetch_via_page(page, "https://x/y") is None


def test_fetch_via_page_handles_null_and_errors():
    assert _recipe()._fetch_via_page(_FakePage(url="x", evaluate_return=None), "u") is None
    page = _FakePage(url="x", evaluate_raises=RuntimeError("execution context destroyed"))
    assert _recipe()._fetch_via_page(page, "u") is None
