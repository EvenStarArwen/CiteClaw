"""Bridge between citeclaw's pipeline and pdfclaw's browser-based PDF fetcher.

The bridge provides a single entry point —
:meth:`PdfClawBridge.fetch_text` — that tries three layers in order:

  1. **Cache** — ``paper_full_text`` in the SQLite cache (instant).
  2. **HTTP** — citeclaw's built-in :func:`download_pdf_bytes` against
     S2's ``openAccessPdf.url`` (fast, no browser).
  3. **PDFClaw recipes** — the full publisher-aware fallback chain from
     ``pdfclaw.publishers``, including browser-based SSO recipes.

Layer 3 is only attempted when ``pdfclaw`` is importable (it's an
optional dependency).  The browser is opened **lazily** on the first
paper that needs it and **reused** across all subsequent papers in the
same bridge lifetime.  Call :meth:`close` (or use the bridge as a
context manager) to shut down the browser cleanly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from citeclaw.clients.pdf import (
    download_pdf_bytes,
    make_pdf_http_client,
    parse_pdf_bytes,
)

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.clients.pdfclaw_bridge")

_DEFAULT_MAX_TEXT_CHARS = 80_000
_DEFAULT_PROFILE_PATH = Path.home() / ".pdfclaw-chrome-profile"


class PdfClawBridge:
    """Cache-aware PDF fetcher that falls through to pdfclaw browser recipes.

    Designed to be instantiated **once per ExpandByPDF run** and reused
    across all papers in the signal.  The browser is opened lazily (only
    when a browser recipe is actually needed) and closed by :meth:`close`.
    """

    def __init__(
        self,
        cache: "Cache",
        *,
        max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
        profile_path: Path | None = None,
        headless: bool = True,
        sleep_between: float = 3.0,
    ) -> None:
        self._cache = cache
        self._max_text_chars = max_text_chars
        self._profile_path = (profile_path or _DEFAULT_PROFILE_PATH).expanduser()
        self._headless = headless
        self._sleep_between = sleep_between

        # Lazy-init state
        self._http = make_pdf_http_client()
        self._pdfclaw_available: bool | None = None
        self._registry: list | None = None
        self._browser_ctx_manager = None
        self._browser_page = None
        # Per-run recipe suppression (mirrors pdfclaw's Fetcher logic)
        self._auth_failed: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_text(self, paper: "PaperRecord") -> str | None:
        """Return parsed PDF body text for *paper*, or ``None`` on failure.

        Tries cache → HTTP → pdfclaw browser in order.  Successes and
        categorised failures are cached so repeated calls for the same
        paper never re-fetch.
        """
        if not paper.paper_id:
            return None

        # 1. Cache hit (text or known failure)
        cached = self._cache.get_full_text(paper.paper_id)
        if cached is not None:
            text = cached.get("text")
            if text:
                return text[: self._max_text_chars] if len(text) > self._max_text_chars else text
            # Cached failure from HTTP — still try pdfclaw browser below.
            # But if the error is "parse_failed" or "too_large", pdfclaw
            # won't help either.
            if cached.get("error") in ("parse_failed", "too_large"):
                return None

        # 2. HTTP fetch (uses S2's openAccessPdf.url)
        if cached is None and paper.pdf_url:
            text = self._try_http(paper)
            if text:
                self._cache.put_full_text(paper.paper_id, text=text)
                return text

        # 3. PDFClaw browser-based recipes
        text = self._try_pdfclaw(paper)
        if text:
            self._cache.put_full_text(paper.paper_id, text=text)
            return text

        # All attempts failed — cache the failure.
        if cached is None:
            self._cache.put_full_text(paper.paper_id, error="download_failed")
        return None

    def close(self) -> None:
        """Release browser and HTTP resources.

        Both the browser context exit and the http-client close can
        legitimately raise during interpreter shutdown (playwright and
        httpx can each fail to talk to their event loops at that
        point). The bridge's contract is "must not propagate" —
        consumers rely on ``with PdfClawBridge(...) as b: ...`` or a
        bare ``b.close()`` in a finally block, neither of which tolerate
        a close failure. DEBUG logs give a diagnostic trail without
        breaking shutdown.
        """
        if self._browser_ctx_manager is not None:
            try:
                self._browser_ctx_manager.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw browser context exit failed: %s", exc)
            self._browser_ctx_manager = None
            self._browser_page = None
        if self._http is not None:
            try:
                self._http.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw bridge http client close failed: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Layer 2: HTTP
    # ------------------------------------------------------------------

    def _try_http(self, paper: "PaperRecord") -> str | None:
        url = paper.pdf_url
        if not url:
            return None
        body, err = download_pdf_bytes(self._http, url)
        if err is not None:
            return None
        text = parse_pdf_bytes(body, max_chars=self._max_text_chars)
        return text

    # ------------------------------------------------------------------
    # Layer 3: PDFClaw browser recipes
    # ------------------------------------------------------------------

    def _try_pdfclaw(self, paper: "PaperRecord") -> str | None:
        if not self._ensure_pdfclaw():
            return None

        doi = self._extract_doi(paper)
        if not doi:
            return None

        from pdfclaw.publishers import find_recipes
        from pdfclaw.publishers.base import STATUS_AUTH

        recipes = find_recipes(doi, self._registry)
        if not recipes:
            return None

        for recipe in recipes:
            if recipe.name in self._auth_failed:
                continue

            # Lazy browser open
            if recipe.needs_browser:
                page = self._ensure_browser()
                if page is None:
                    continue
            else:
                page = None

            try:
                result = recipe.fetch(
                    paper.paper_id,
                    doi,
                    browser_page=page if recipe.needs_browser else None,
                    http=self._http if not recipe.needs_browser else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfclaw recipe %s raised: %s", recipe.name, exc)
                if recipe.needs_browser:
                    self._bump_failures(recipe.name)
                continue

            if result.ok:
                self._consecutive_failures[recipe.name] = 0
                return self._extract_text(result)

            if result.status == STATUS_AUTH:
                self._auth_failed.add(recipe.name)
                log.info(
                    "pdfclaw: recipe %s needs auth; suppressed for this run",
                    recipe.name,
                )
                continue

            if recipe.needs_browser and result.status in ("error", "blocked"):
                self._bump_failures(recipe.name)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_doi(self, paper: "PaperRecord") -> str | None:
        """Get DOI from paper metadata.  Try external_ids first, then ArXiv fallback."""
        doi = paper.external_ids.get("DOI")
        if doi:
            return doi
        arxiv = paper.external_ids.get("ArXiv")
        if arxiv:
            return f"10.48550/arXiv.{arxiv}"
        return None

    def _extract_text(self, result) -> str | None:
        """Parse a successful FetchResult into body text."""
        if result.body_text:
            text = result.body_text
        elif result.pdf_bytes:
            text = parse_pdf_bytes(result.pdf_bytes, max_chars=self._max_text_chars)
        else:
            return None
        if text and len(text) > self._max_text_chars:
            text = text[: self._max_text_chars]
        return text or None

    def _ensure_pdfclaw(self) -> bool:
        """Check whether pdfclaw is importable; cache the result."""
        if self._pdfclaw_available is not None:
            return self._pdfclaw_available
        try:
            from pdfclaw.publishers import build_default_registry

            self._registry = build_default_registry()
            self._pdfclaw_available = True
        except ImportError:
            log.info("pdfclaw not installed — browser-based PDF fetching disabled")
            self._pdfclaw_available = False
        return self._pdfclaw_available

    def _ensure_browser(self):
        """Lazily open a persistent browser context; return the Page or None."""
        if self._browser_page is not None:
            return self._browser_page
        try:
            from pdfclaw.browser import open_browser_context

            self._browser_ctx_manager = open_browser_context(
                self._profile_path,
                headless=self._headless,
            )
            _ctx, self._browser_page = self._browser_ctx_manager.__enter__()
            log.info("pdfclaw browser opened (headless=%s)", self._headless)
            return self._browser_page
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to open pdfclaw browser: %s", exc)
            self._pdfclaw_available = False  # Don't retry browser recipes
            return None

    def _bump_failures(self, recipe_name: str) -> None:
        count = self._consecutive_failures.get(recipe_name, 0) + 1
        self._consecutive_failures[recipe_name] = count
        if count >= 3 and recipe_name not in self._auth_failed:
            self._auth_failed.add(recipe_name)
            log.warning(
                "pdfclaw: recipe %s hit %d consecutive failures; suppressed",
                recipe_name,
                count,
            )

    def sleep(self) -> None:
        """Polite delay between fetches."""
        if self._sleep_between > 0:
            time.sleep(self._sleep_between)
