"""Shared fetch flow for any publisher recipe that needs a real Chrome.

Almost every publisher works the same way under the hood:

  1. ``page.goto(https://doi.org/<doi>)``
  2. Wait for the article landing page to settle
  3. Check whether we got bounced to an SSO host (cookies missing/expired)
  4. Try a list of CSS / XPath selectors that match the publisher's
     "Download PDF" link
  5. Click the first matching selector inside ``page.expect_download(...)``
  6. Read the resulting PDF bytes and verify they start with ``%PDF-``

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

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pdfclaw.publishers.base import (
    STATUS_AUTH,
    STATUS_ERROR,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page


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
    NETWORKIDLE_TIMEOUT_MS: int = 20_000
    DOWNLOAD_TIMEOUT_MS: int = 60_000
    CLICK_TIMEOUT_MS: int = 10_000

    def matches(self, doi: str) -> bool:
        d = doi.lower()
        if isinstance(self.DOI_PREFIX, str):
            return d.startswith(self.DOI_PREFIX)
        return any(d.startswith(p) for p in self.DOI_PREFIX)

    def _build_url(self, doi: str) -> str:
        if self.URL_BUILDER is not None:
            return self.URL_BUILDER(doi)
        return f"https://doi.org/{doi}"

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

        # Step 4: try each download selector in order
        last_err: str | None = None
        for selector in self.DOWNLOAD_SELECTORS:
            try:
                if page.locator(selector).count() == 0:
                    last_err = f"selector not present: {selector}"
                    continue
                with page.expect_download(timeout=self.DOWNLOAD_TIMEOUT_MS) as dl_info:
                    page.locator(selector).first.click(timeout=self.CLICK_TIMEOUT_MS)
                download = dl_info.value

                tmpdir = Path(tempfile.mkdtemp(prefix=f"pdfclaw_{self.name}_"))
                tmp_pdf = tmpdir / "tmp.pdf"
                try:
                    download.save_as(str(tmp_pdf))
                    body = tmp_pdf.read_bytes()
                finally:
                    if tmp_pdf.exists():
                        tmp_pdf.unlink()
                    tmpdir.rmdir()

                if not body.startswith(b"%PDF-"):
                    return FetchResult(
                        paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                        fetched_via=self.name,
                        error=f"Downloaded body isn't PDF (first bytes: {body[:8]!r})",
                    )

                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={
                        "selector": selector,
                        "n_bytes": len(body),
                        "final_url": page.url,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{selector}: {exc}"
                continue

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
