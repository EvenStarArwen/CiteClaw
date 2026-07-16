"""Shared fetch flow for any publisher recipe that needs a real Chrome.

Almost every publisher works the same way under the hood:

  1. ``page.goto(https://doi.org/<doi>)``
  2. Wait for the article landing page to settle
  3. Check whether we got bounced to an SSO host (cookies missing/expired)
  4. Try a list of CSS / XPath selectors that match the publisher's
     "Download PDF" link, resolve its ``href``
  5. Download that href through an escalating three-tier strategy, each
     tier defeating a failure mode the previous one can't:
       a. ``context.request.get`` — fast, carries cookies; works when the
          PDF endpoint has no active bot protection.
       b. in-page ``fetch`` — runs in the live JS context, clearing a
          Cloudflare/Akamai check that 403's the bare request (same-origin
          endpoints only).
       c. top-level navigation (``page.goto`` the href) — the only tier
          that solves an *active* JS challenge AND follows the cross-origin
          redirect publishers use to hand off to a tokenised CDN (e.g. OUP
          -> ``watermark*.silverchair.com``). Chrome downloads or renders
          the PDF inline; either is captured.
  6. Verify the bytes start with ``%PDF-``

The only thing that differs between publishers is **the DOI prefix and
the selector list**. Everything else is shared. This base class
captures the shared flow so each new publisher recipe is 10-20 lines
instead of 100.

Subclasses set ONLY these class attributes:

  * ``name`` — short identifier for logs (e.g. ``"nature_browser"``)
  * ``DOI_PREFIX`` — string or tuple of strings the DOI must start with
  * ``DOWNLOAD_SELECTORS`` — ordered list of locators to try
  * (optional) ``SSO_HOSTS`` — extra substrings to detect auth bounce
  * (optional) ``EXTRA_WAIT_MS`` — extra wait after page settle
  * (optional) ``URL_BUILDER`` — callable taking ``doi`` and returning
    the navigation URL (default: ``https://doi.org/<doi>``)

When ALL selectors fail, the recipe writes a snapshot of the failed
page (HTML + a list of all anchor candidates with ``pdf`` in their
href or visible text) to ``/tmp/pdfclaw_failures/<recipe>_<paper_id>.{html,txt}``
so you can grep for the right selector after the run.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pdfclaw.publishers.base import (
    STATUS_AUTH,
    STATUS_ERROR,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

log = logging.getLogger("pdfclaw.publishers._browser_base")


# In-page PDF fetch. Runs inside the live Chrome JS context via
# ``page.evaluate`` — unlike ``page.context.request.get`` (a bare HTTP
# client), this inherits the page's already-solved Cloudflare/Akamai
# challenge, its cookies, and same-origin credentials, so it retrieves
# PDFs from endpoints that 403 a plain request. Bytes are chunk-encoded
# to base64 (32 KB windows) so multi-megabyte PDFs don't blow the
# argument limit of ``String.fromCharCode.apply`` or hang on per-byte
# string concatenation. Returns base64 text, or ``null`` on any failure
# (non-OK response, CORS block, network error).
_PDF_FETCH_JS = """
async (url) => {
    try {
        const resp = await fetch(url, {credentials: 'include'});
        if (!resp.ok) return null;
        const bytes = new Uint8Array(await resp.arrayBuffer());
        let binary = '';
        const CHUNK = 0x8000;
        for (let i = 0; i < bytes.length; i += CHUNK) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
        }
        return btoa(binary);
    } catch (e) { return null; }
}
"""


# Substrings that, if present in the URL after redirect, indicate the
# session needs to log in via institutional SSO. Shared across publishers.
DEFAULT_SSO_HOSTS: tuple[str, ...] = (
    "shibboleth",
    "wayfless",
    "openathens",
    "saml",
    "/sso/",
    "idp.",
    "login.",
    "/login",
    "auth.",
)


# Generic last-resort selectors appended to every recipe's DOWNLOAD_SELECTORS
# automatically. These catch publishers that have unusual or rebranded PDF
# anchors as long as some accessible attribute (text / aria-label / title /
# href) contains "PDF" or "Download". They run AFTER the publisher-specific
# selectors so the specific ones still take priority when present.
GENERIC_FALLBACK_SELECTORS: tuple[str, ...] = (
    'a[aria-label*="download pdf" i]',
    'a[aria-label*="full text pdf" i]',
    'a[aria-label*="pdf" i]',
    'a[title*="download pdf" i]',
    'a[title*="pdf" i]',
    'a[data-track-action*="pdf" i]',
    'a[data-test*="pdf" i]',
    'a[href*="/pdf"][href*="download"]',
    'a[href*=".pdf"]:not([href*=".pdfx"])',
    'a:has-text("Download PDF")',
    'a:has-text("Full Text PDF")',
    'a:has-text("View PDF")',
    'a:has-text("PDF Download")',
    'button[aria-label*="pdf" i]',
)


class BrowserRecipeBase:
    """Mix-in implementing the standard navigate -> click -> download flow."""

    needs_browser = True
    name: str = "TBD"
    DOI_PREFIX: str | tuple[str, ...] = ()
    DOWNLOAD_SELECTORS: list[str] = []
    SSO_HOSTS: tuple[str, ...] = DEFAULT_SSO_HOSTS
    EXTRA_WAIT_MS: int = 0
    URL_BUILDER: Callable[[str], str] | None = None
    GOTO_TIMEOUT_MS: int = 45_000
    NETWORKIDLE_TIMEOUT_MS: int = 15_000
    DOWNLOAD_TIMEOUT_MS: int = 20_000
    CLICK_TIMEOUT_MS: int = 10_000
    # How long to wait for a download event to *start* after navigating to
    # a PDF href. A download fires within a second or two of navigation
    # commit; when none fires the PDF rendered inline and we extract it
    # from the viewer instead. Kept short so the inline path isn't taxed
    # by the full DOWNLOAD_TIMEOUT_MS.
    NAV_DOWNLOAD_WAIT_MS: int = 6_000
    # If set, try fetching this URL (with {doi} placeholder) BEFORE
    # scanning for selectors. Many publishers have a predictable
    # /doi/pdf/{doi} endpoint that returns raw PDF bytes — much more
    # reliable than finding + clicking buttons on their JS-rendered pages.
    PDF_URL_TEMPLATE: str | None = None

    def matches(self, doi: str) -> bool:
        d = doi.lower()
        if isinstance(self.DOI_PREFIX, str):
            return d.startswith(self.DOI_PREFIX)
        return any(d.startswith(p) for p in self.DOI_PREFIX)

    def _all_selectors(self) -> list[str]:
        """Recipe-specific selectors followed by the generic fallbacks."""
        seen: set[str] = set()
        out: list[str] = []
        for s in list(self.DOWNLOAD_SELECTORS) + list(GENERIC_FALLBACK_SELECTORS):
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _build_url(self, doi: str) -> str:
        if self.URL_BUILDER is not None:
            return self.URL_BUILDER(doi)
        return f"https://doi.org/{doi}"

    def _fetch_via_page(self, page: "Page", url: str) -> bytes | None:
        """Download ``url`` from inside the page's JS context.

        This is the anti-bot fallback: an in-page ``fetch`` inherits the
        live browser's solved Cloudflare/Akamai challenge and cookies,
        so it succeeds where the out-of-band ``context.request.get`` is
        re-challenged and 403s. Returns verified PDF bytes or ``None``.
        """
        try:
            raw = page.evaluate(_PDF_FETCH_JS, url)
        except Exception as exc:  # noqa: BLE001 — best-effort fallback
            log.debug("%s: in-page fetch(%s) failed: %s", self.name, url[:80], exc)
            return None
        if not raw:
            return None
        try:
            body = base64.b64decode(raw)
        except Exception:  # noqa: BLE001
            return None
        return body if body.startswith(b"%PDF-") else None

    def _download_href(
        self, page: "Page", url: str,
    ) -> tuple[bytes | None, str, str | None]:
        """Fetch a PDF from ``url`` with a two-tier strategy.

        1. ``page.context.request.get`` — fast, carries cookies, works
           for endpoints without active bot protection.
        2. In-page JS ``fetch`` — slower but clears the Cloudflare /
           Akamai check that 403s tier 1.

        Returns ``(pdf_bytes | None, method_or_reason, sso_url | None)``.
        On success ``method`` is ``"context_request"`` or ``"page_fetch"``;
        on failure ``method`` carries the reason for the snapshot log. A
        non-empty ``sso_url`` signals an auth bounce the caller should
        surface as :data:`STATUS_AUTH`.
        """
        reason = "no attempt"
        try:
            api_resp = page.context.request.get(
                url, timeout=self.DOWNLOAD_TIMEOUT_MS,
                headers={"Referer": page.url},
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"context.request.get failed: {exc}"
        else:
            resp_url = (getattr(api_resp, "url", "") or "").lower()
            if not api_resp.ok and any(h in resp_url for h in self.SSO_HOSTS):
                return None, "sso", api_resp.url
            if api_resp.ok:
                body = api_resp.body()
                if body.startswith(b"%PDF-"):
                    return body, "context_request", None
                reason = "non-PDF response (bot-challenge or login HTML)"
            else:
                reason = f"HTTP {api_resp.status}"

        # Tier 2: in-page fetch defeats the anti-bot check on the PDF endpoint.
        body = self._fetch_via_page(page, url)
        if body:
            return body, "page_fetch", None
        return None, reason, None

    def _download_via_navigation(self, page: "Page", url: str) -> bytes | None:
        """Last-tier download: navigate the real browser to ``url``.

        A top-level navigation is the only technique that both (a) solves
        an *active* Cloudflare/Akamai JS challenge on the PDF endpoint and
        (b) follows the cross-origin redirect publishers use to hand off
        to a tokenised CDN (e.g. OUP -> ``watermark*.silverchair.com``).
        Neither ``context.request.get`` (403'd by the challenge) nor an
        in-page ``fetch`` (CORS-blocked by the cross-origin redirect) can
        do this — verified empirically against OUP.

        Chrome then either downloads the PDF (captured here) or renders it
        inline, in which case ``page.url`` is the settled, same-origin,
        challenge-cleared PDF and an in-page fetch of it succeeds.
        """
        try:
            with page.expect_download(timeout=self.NAV_DOWNLOAD_WAIT_MS) as dl_info:
                try:
                    page.goto(url, wait_until="commit", timeout=self.GOTO_TIMEOUT_MS)
                except Exception:  # noqa: BLE001 — a download nav raises ERR_ABORTED
                    pass
            try:
                path = dl_info.value.path()
            except Exception:  # noqa: BLE001
                path = None
            if path:
                data = Path(path).read_bytes()
                if data.startswith(b"%PDF-"):
                    return data
        except Exception:  # noqa: BLE001 — no download event: inline render
            pass

        # Inline-render path: settle, then extract from where we landed.
        try:
            page.wait_for_load_state(
                "networkidle", timeout=self.NETWORKIDLE_TIMEOUT_MS,
            )
        except Exception:  # noqa: BLE001
            pass
        return self._fetch_via_page(page, page.url)

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page: "Page | None" = None,
        http=None,  # noqa: ARG002
    ) -> FetchResult:
        if browser_page is None:
            raise ValueError(f"{self.name} needs a Playwright Page; pass browser_page=...")

        page = browser_page
        target_url = self._build_url(doi)

        # Step 0: clean up — navigate to about:blank first to release
        # resources from whatever article the page was showing before.
        # Without this, each paper leaves a heavy publisher page loaded
        # in the tab, and after 50+ papers the memory usage gets bad.
        try:
            page.goto("about:blank", wait_until="commit", timeout=5_000)
        except Exception:  # noqa: BLE001
            pass

        # Step 1: navigate
        try:
            page.goto(
                target_url,
                wait_until="domcontentloaded",
                timeout=self.GOTO_TIMEOUT_MS,
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"goto {target_url} failed: {exc}",
            )

        # Step 2: detect SSO bounce
        url_lower = page.url.lower()
        if any(host in url_lower for host in self.SSO_HOSTS):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_AUTH,
                fetched_via=self.name,
                error=(
                    f"Hit SSO at {page.url}. Run `python -m pdfclaw login` "
                    "and sign in via your institution, then re-run this fetch."
                ),
            )

        # Step 3: wait for the article body to render
        try:
            page.wait_for_load_state(
                "networkidle", timeout=self.NETWORKIDLE_TIMEOUT_MS,
            )
        except Exception:  # noqa: BLE001
            # networkidle is best-effort — many publishers have long-poll
            # pings that prevent it from ever firing.
            pass

        if self.EXTRA_WAIT_MS > 0:
            page.wait_for_timeout(self.EXTRA_WAIT_MS)

        # Step 3.5: dismiss cookie consent banners
        for cookie_sel in [
            'button:has-text("Accept All")', 'button:has-text("Accept all")',
            'button:has-text("Accept")', 'button:has-text("Agree")',
            '#onetrust-accept-btn-handler', '.cookie-accept',
        ]:
            try:
                loc = page.locator(cookie_sel)
                if loc.count() > 0:
                    loc.first.click(timeout=2_000)
                    page.wait_for_timeout(500)
                    break
            except Exception:  # noqa: BLE001
                continue

        # Step 4a: try the known PDF URL template if this recipe has one.
        # This bypasses selector discovery entirely for publishers with
        # a predictable /doi/pdf/{doi} endpoint. Uses the same two-tier
        # (context request -> in-page fetch) download so a bot-protected
        # template endpoint still resolves. On miss we fall through to
        # selector discovery rather than failing.
        if self.PDF_URL_TEMPLATE:
            pdf_url = self.PDF_URL_TEMPLATE.format(doi=doi)
            body, method, _sso = self._download_href(page, pdf_url)
            if body:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={"href": pdf_url, "method": f"pdf_url_template:{method}",
                           "n_bytes": len(body)},
                )
            log.debug("%s: PDF_URL_TEMPLATE miss (%s); trying selectors",
                      self.name, method)

        # Step 4b: discover the PDF link href and download it. ``nav_href``
        # remembers the first resolved href so the navigation tier below
        # can retry it if the direct download tiers are blocked.
        from urllib.parse import urlparse  # noqa: PLC0415

        last_err: str | None = None
        nav_href: str | None = None
        for selector in self._all_selectors():
            try:
                loc = page.locator(selector)
                if loc.count() == 0:
                    last_err = f"selector not present: {selector}"
                    continue

                href = loc.first.get_attribute("href")
                if not href:
                    last_err = f"{selector}: no href attribute"
                    continue

                # Make absolute
                if href.startswith("/"):
                    parts = urlparse(page.url)
                    href = f"{parts.scheme}://{parts.netloc}{href}"
                elif not href.startswith("http"):
                    href = f"https://{href}"
                if nav_href is None:
                    nav_href = href

                # Download via the two-tier strategy: fast context
                # request first, then an in-page fetch that clears the
                # anti-bot check when the PDF endpoint 403s the bare request.
                body, method, sso_url = self._download_href(page, href)
                if sso_url:
                    return FetchResult(
                        paper_id=paper_id, doi=doi, status=STATUS_AUTH,
                        fetched_via=self.name,
                        error=f"PDF fetch redirected to SSO: {sso_url}",
                    )
                if body:
                    return FetchResult(
                        paper_id=paper_id, doi=doi, status=STATUS_OK,
                        pdf_bytes=body, fetched_via=self.name,
                        extra={
                            "selector": selector,
                            "href": href,
                            "method": method,
                            "n_bytes": len(body),
                            "final_url": page.url,
                        },
                    )
                last_err = f"{selector}: {method} for {href}"
                continue
            except Exception as exc:  # noqa: BLE001
                last_err = f"{selector}: {exc}"
                continue

        # Last resort A: if the page URL already looks like a PDF URL
        # (Chrome opened it in the viewer), extract bytes via in-page fetch.
        page_url = page.url.lower()
        if ".pdf" in page_url or "/pdf" in page_url:
            body = self._fetch_via_page(page, page.url)
            if body:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={"method": "viewer_js_extract", "n_bytes": len(body)},
                )

        # Last resort B: navigate the real browser straight to the PDF
        # href. This clears active Cloudflare challenges and follows
        # cross-origin CDN redirects that defeat the direct-download tiers.
        if nav_href:
            body = self._download_via_navigation(page, nav_href)
            if body:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={"method": "navigation", "href": nav_href,
                           "n_bytes": len(body), "final_url": page.url},
                )

        snapshot = self._dump_failure(page, paper_id)
        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_ERROR,
            fetched_via=self.name,
            error=(
                f"All download selectors failed for {self.name}; "
                f"last error: {last_err}. Page snapshot saved to {snapshot}. "
                "Inspect the .txt for candidate anchors and update "
                "DOWNLOAD_SELECTORS in the recipe."
            ),
            extra={"failure_snapshot": snapshot, "final_url": page.url},
        )

    def _dump_failure(self, page: "Page", paper_id: str) -> str:
        """Save HTML + a list of pdf-ish anchors so the next iteration
        of the selector list can be informed."""
        from pathlib import Path
        out_dir = Path("/tmp/pdfclaw_failures")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = page.content()
            (out_dir / f"{self.name}_{paper_id}.html").write_text(
                html, encoding="utf-8", errors="ignore",
            )
        except Exception:  # noqa: BLE001
            pass

        # Extract all anchors that look PDF-related
        try:
            anchors = page.evaluate(
                """() => Array.from(document.querySelectorAll('a')).map(a => ({
                    href: a.getAttribute('href') || '',
                    text: (a.innerText || '').trim().slice(0, 100),
                    cls: a.getAttribute('class') || '',
                    aria: a.getAttribute('aria-label') || '',
                    title: a.getAttribute('title') || '',
                    data_track: a.getAttribute('data-track-action') || '',
                })).filter(a =>
                    /pdf/i.test(a.href) || /pdf/i.test(a.text) ||
                    /pdf/i.test(a.aria) || /pdf/i.test(a.title) ||
                    /download/i.test(a.text) || /download/i.test(a.aria)
                )"""
            )
        except Exception:  # noqa: BLE001
            anchors = []

        try:
            lines = [f"# Snapshot for {self.name} / {paper_id}",
                     f"# Final URL: {page.url}",
                     f"# Title: {page.title()}",
                     f"# {len(anchors)} candidate anchors found:",
                     ""]
            for a in anchors:
                lines.append(
                    f"href: {a.get('href','')[:120]}\n"
                    f"  text: {a.get('text','')[:80]}\n"
                    f"  class: {a.get('cls','')[:60]}\n"
                    f"  aria: {a.get('aria','')[:60]}\n"
                    f"  title: {a.get('title','')[:60]}\n"
                    f"  data-track-action: {a.get('data_track','')[:60]}\n"
                )
            (out_dir / f"{self.name}_{paper_id}.txt").write_text(
                "\n".join(lines), encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

        return str(out_dir / f"{self.name}_{paper_id}.txt")
