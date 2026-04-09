"""Open-access PDF download + parse for the ``full_text`` LLMFilter scope.

PH-06. Given a list of :class:`PaperRecord` instances, this module:

  1. Looks each paper up in the ``paper_full_text`` cache.
  2. For cache misses with a non-empty ``pdf_url``, downloads the PDF
     via httpx and parses it via ``pypdf``.
  3. Stores the outcome (parsed text on success, error category on
     failure) in cache.db so subsequent runs don't re-fetch the same
     papers.

Failure modes are categorised, not raised:

  - ``"no_pdf"``         — paper.pdf_url is empty (closed-access)
  - ``"download_failed"`` — httpx error (timeout, 4xx/5xx, transport)
  - ``"parse_failed"``    — pypdf raised on the downloaded bytes
  - ``"too_large"``       — body exceeded ``max_size_mb``

The caller (``llm_runner._dispatch_simple`` for scope=full_text) treats
"no text available" as a benign skip and falls back to title+abstract
content for that paper, so a closed-access paper still gets screened
under the full-text rule via its abstract.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.pdf")

# Practical maxima — academic PDFs are usually 1-15 MB and 8-40 pages
# of body text, which translates to 30-200K characters after parsing.
# These caps protect against pathological PDFs (huge appendices, OCR'd
# scans, malformed files) without dropping legitimate papers.
#
# 50 MB ceiling: most bioRxiv preprints with full supplementary
# material end up in the 20-40 MB range, and dropping them at 20 MB
# (the original cap) excluded a meaningful fraction of legitimate
# preprints during PH-06 live testing. 50 MB is the practical sweet
# spot — well above honest preprint sizes, well below pathological
# scanned-image PDFs.
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_SIZE_MB = 50
# 80K chars ≈ 25K BPE tokens. Sized to fit a single paper into Gemma 4
# 31B's 32K context with room for system prompt + user wrapper +
# structured-output JSON. Bump if you serve a higher-context model.
DEFAULT_MAX_TEXT_CHARS = 80_000


class PdfFetcher:
    """Cache-aware PDF download + parse.

    Stateless w.r.t. its inputs — the only state is the shared
    ``Cache`` and an internal ``httpx.Client``. Safe to construct once
    per CiteClaw run and reuse across many papers.
    """

    def __init__(
        self,
        cache: "Cache",
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    ) -> None:
        self._cache = cache
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_text_chars = max_text_chars
        # ``follow_redirects`` is essential — most preprint servers
        # serve PDFs from a final-CDN host that's a redirect away from
        # the metadata URL.
        self._http = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={
                # Some hosts (notably bioRxiv) require a UA or they 403.
                "User-Agent": (
                    "Mozilla/5.0 CiteClaw/0.1 (literature acquisition agent)"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text_for(self, paper: "PaperRecord") -> str | None:
        """Return the cached or freshly-fetched body text for one paper.

        Returns ``None`` if the paper has no PDF or the fetch/parse
        failed (the failure is cached so the caller never has to retry
        manually). The caller can then fall back to abstract-only
        content for that paper.
        """
        if not paper.paper_id:
            return None
        cached = self._cache.get_full_text(paper.paper_id)
        if cached is not None:
            return cached.get("text")  # may be None on cached failure
        text, error = self._fetch_and_parse(paper)
        self._cache.put_full_text(paper.paper_id, text=text, error=error)
        return text

    def prefetch(
        self,
        papers: list["PaperRecord"],
        *,
        max_workers: int = 4,
    ) -> dict[str, str | None]:
        """Download + parse PDFs for ``papers`` in parallel.

        Returns ``{paper_id: text_or_None}`` for every input paper. The
        ``max_workers`` cap limits concurrent HTTP requests so we don't
        flood preprint servers — bioRxiv in particular rate-limits
        aggressive parallel clients. 4 is a polite default.

        Cache hits are returned without re-downloading; only cache
        misses with a non-empty ``pdf_url`` cost a network request.
        """
        out: dict[str, str | None] = {}
        misses: list["PaperRecord"] = []
        for p in papers:
            if not p.paper_id:
                continue
            cached = self._cache.get_full_text(p.paper_id)
            if cached is not None:
                out[p.paper_id] = cached.get("text")
            else:
                misses.append(p)

        if not misses:
            return out

        log.info(
            "PdfFetcher: %d cache hits, %d cache misses to fetch (max_workers=%d)",
            len(out), len(misses), max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._fetch_and_parse, p): p
                for p in misses
            }
            for fut in as_completed(futures):
                paper = futures[fut]
                try:
                    text, error = fut.result()
                except Exception as exc:  # pragma: no cover - safety net
                    log.warning(
                        "PdfFetcher: unexpected error for %s: %s",
                        paper.paper_id, exc,
                    )
                    text, error = None, "download_failed"
                self._cache.put_full_text(paper.paper_id, text=text, error=error)
                out[paper.paper_id] = text
        return out

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_and_parse(
        self, paper: "PaperRecord",
    ) -> tuple[str | None, str | None]:
        """Return ``(text, error)`` for one paper. Exactly one is None."""
        url = paper.pdf_url
        if not url:
            return None, "no_pdf"

        try:
            resp = self._http.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.info(
                "PdfFetcher: download failed for %s (%s): %s",
                paper.paper_id, url, exc,
            )
            return None, "download_failed"

        body = resp.content
        if len(body) > self._max_size_bytes:
            log.info(
                "PdfFetcher: PDF too large for %s (%d bytes > cap)",
                paper.paper_id, len(body),
            )
            return None, "too_large"

        text = self._parse_pdf_bytes(body)
        if text is None:
            return None, "parse_failed"

        if len(text) > self._max_text_chars:
            text = text[: self._max_text_chars]
        return text, None

    def _parse_pdf_bytes(self, body: bytes) -> str | None:
        """Run pypdf over a PDF byte-string. Lazy-import so installs
        without the ``pdf`` extra still work for every other scope."""
        try:
            import pypdf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Full-text PDF parsing requires the 'pdf' extra. "
                "Install with: pip install 'citeclaw[pdf]'"
            ) from exc

        try:
            reader = pypdf.PdfReader(io.BytesIO(body))
            chunks: list[str] = []
            for page in reader.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    # Single-page extraction failures are common in
                    # mixed-encoding PDFs — skip the bad page rather
                    # than aborting the whole parse.
                    continue
            text = "\n".join(chunks).strip()
            return text or None
        except Exception as exc:
            log.info("PdfFetcher: pypdf parse failed: %s", exc)
            return None
