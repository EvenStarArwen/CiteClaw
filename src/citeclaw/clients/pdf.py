"""Open-access PDF download + parse — used by the ``full_text`` LLMFilter
scope (in-memory, cached) and by the ``fetch-pdfs`` CLI subcommand
(download bundles to disk for downstream agents).

S2's ``openAccessPdf.url`` is inconsistent in practice: sometimes a
direct ``...pdf`` URL, sometimes a DOI redirect to a publisher landing
page (Nature, PMC, etc.). When the response body is HTML rather than
PDF bytes, ``download_pdf_bytes`` does ONE retry against the page's
``<meta name="citation_pdf_url">`` tag (Highwire Press standard,
implemented by Nature, Science, IEEE, ACM, Springer, ...) — failing
that, it scans for the first ``href="...pdf"`` link (rescues PMC
articles). This rescues most of the failed-but-recoverable cases.
Cloudflare-protected hosts like ``www.biorxiv.org`` cannot be bypassed
without a real browser session.

Two collaborators share this module:

  1. :class:`PdfFetcher` — cache-aware ``text_for`` / ``prefetch``
     used by ``llm_runner`` for the ``full_text`` LLMFilter scope. Caches
     successes AND categorised failures in the ``paper_full_text`` table
     so a second pass never repeats a known-failing fetch.
  2. The ``fetch-pdfs`` CLI (``citeclaw.fetch_pdfs``) — saves PDF bytes
     and parsed text to ``<data_dir>/PDFs/<paper_id>.{pdf,txt}`` for
     downstream tools, while still warming the same cache.

Both routes share two free helpers in this module: :func:`download_pdf_bytes`
and :func:`parse_pdf_bytes`. The parser tries PyMuPDF (``fitz``) first
because it dramatically out-performs pypdf on two-column scientific PDFs
(correct reading order, faster, fewer mangled characters), and falls
back to ``pypdf`` if PyMuPDF isn't installed. PyMuPDF is AGPL — that's
fine for an internal research tool but worth knowing.

Other parser options worth considering for future work:
  - GROBID (Java server) — gold standard for structured scientific paper
    extraction (sections, refs, tables). Heavier but produces TEI XML.
  - Marker / Nougat / Docling — ML-based, output clean Markdown with
    formula and table preservation. GPU-friendly. Use when you need
    layout-aware extraction rather than plain reading-order text.

Failure modes are categorised, not raised:

  - ``"no_pdf"``         — paper.pdf_url is empty (closed-access)
  - ``"download_failed"`` — httpx error (timeout, 4xx/5xx, transport)
  - ``"parse_failed"``    — both parsers raised on the downloaded bytes
  - ``"too_large"``       — body exceeded ``max_size_mb``
  - ``"not_pdf"``         — server returned HTML / non-PDF bytes
"""

from __future__ import annotations

import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from urllib.parse import urljoin

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
# material end up in the 20-40 MB range. 20 MB used to be the cap and
# excluded too many legitimate preprints; 50 MB is well above honest
# preprint sizes and well below pathological scanned-image PDFs.
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_SIZE_MB = 50
# 80K chars ≈ 25K BPE tokens. Sized to fit a single paper into Gemma 4
# 31B's 32K context with room for system prompt + user wrapper +
# structured-output JSON. Bump if you serve a higher-context model.
DEFAULT_MAX_TEXT_CHARS = 80_000

# Default UA — bioRxiv, Wiley and several other preprint / publisher
# hosts return ``403 Forbidden`` for the obvious "Mozilla CiteClaw"
# string an academic crawler might use. They accept regular-looking
# browser UAs without complaint, and a real Chrome UA is the common
# convention for academic literature crawlers (e.g. zotero translators,
# scholarly APIs). Paired with a browser-style ``Accept`` header it
# matches what an unauthenticated browser would send when a user
# clicks a "PDF" link, which is exactly the access pattern we want.
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_DEFAULT_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def make_pdf_http_client(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> httpx.Client:
    """Build a long-lived ``httpx.Client`` configured for PDF downloads.

    Centralised so the ``PdfFetcher`` and the ``fetch-pdfs`` CLI use
    identical follow-redirect / UA / timeout settings.
    """
    return httpx.Client(
        timeout=timeout_s,
        follow_redirects=True,
        headers=_DEFAULT_HEADERS,
    )


_CITATION_PDF_RE = re.compile(
    rb"<meta\s+[^>]*name=[\"']citation_pdf_url[\"'][^>]*content=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_HREF_PDF_RE = re.compile(rb"href=[\"']([^\"'>]+\.pdf[^\"'>]*)[\"']", re.IGNORECASE)


def _looks_like_html(body: bytes) -> bool:
    head = body[:32].lstrip().lower()
    return (
        head.startswith(b"<!doc")
        or head.startswith(b"<html")
        or head.startswith(b"<?xml")
    )


def _validate_pdf_body(
    body: bytes, url: str, max_size_bytes: int,
) -> tuple[bytes | None, str | None]:
    """Sniff a downloaded body and return ``(body, None)`` if it's a PDF.

    Returns one of:
      - ``(bytes, None)`` — looks like a PDF (``%PDF`` magic) and within size cap.
      - ``(None, "too_large")`` — body exceeded the cap.
      - ``(None, "not_pdf")`` — body is HTML or has no PDF magic.
    """
    if len(body) > max_size_bytes:
        log.info("download_pdf_bytes: too large (%d bytes > cap)", len(body))
        return None, "too_large"
    if _looks_like_html(body):
        log.info("download_pdf_bytes: response is HTML, not PDF (%s)", url)
        return None, "not_pdf"
    if not body[:32].lstrip().startswith(b"%PDF"):
        log.info(
            "download_pdf_bytes: response is not a PDF (head=%r) (%s)",
            body[:16], url,
        )
        return None, "not_pdf"
    return body, None


def _extract_pdf_url_from_html(html: bytes, base_url: str) -> str | None:
    """Look for a real PDF URL inside an HTML landing page.

    Strategy:
      1. Highwire ``<meta name="citation_pdf_url" content="...">`` —
         standard implemented by Nature, Science, IEEE, ACM, Springer,
         Elsevier, JAMA, BMJ. Almost always an absolute URL.
      2. First ``href="...pdf"`` link — rescues PMC's
         ``pdf/<filename>.pdf`` link inside the article landing page.

    Landing-page URLs like ``.../articles/PMC10925082`` (no trailing
    slash) trip ``urljoin`` into stripping the last path segment when
    resolving relative hrefs. Append a trailing slash before joining
    so a relative ``pdf/foo.pdf`` resolves under the article path
    rather than its parent.
    """
    if not base_url.endswith("/"):
        base_for_join = base_url + "/"
    else:
        base_for_join = base_url
    m = _CITATION_PDF_RE.search(html)
    if m:
        return urljoin(base_for_join, m.group(1).decode("utf-8", errors="ignore"))
    m = _HREF_PDF_RE.search(html)
    if m:
        return urljoin(base_for_join, m.group(1).decode("utf-8", errors="ignore"))
    return None


def download_pdf_bytes(
    http: httpx.Client,
    url: str,
    *,
    max_size_bytes: int = DEFAULT_MAX_SIZE_MB * 1024 * 1024,
) -> tuple[bytes | None, str | None]:
    """Download a single PDF, applying the same checks PdfFetcher uses.

    If the initial response is HTML rather than a PDF, performs ONE
    follow-up request against ``<meta name="citation_pdf_url">`` (or
    the first ``href="...pdf"`` link found in the body). Rescues most
    DOI redirects to publisher landing pages (Nature, PMC, etc.).
    Hosts behind a Cloudflare bot challenge (e.g. www.biorxiv.org)
    can't be bypassed and still return ``download_failed``.

    Returns ``(body, error)``. Exactly one is non-None:

      - ``(bytes, None)`` — body is a valid PDF, within the size cap.
      - ``(None, "download_failed")`` — httpx error or non-2xx status.
      - ``(None, "too_large")`` — body exceeded ``max_size_bytes``.
      - ``(None, "not_pdf")`` — server returned HTML / non-PDF bytes
        even after the landing-page rescue attempt.
    """
    try:
        resp = http.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.info("download_pdf_bytes: failed (%s): %s", url, exc)
        return None, "download_failed"

    body = resp.content
    final_url = str(resp.url)

    # Fast path: response is already a PDF.
    validated, err = _validate_pdf_body(body, final_url, max_size_bytes)
    if err is None:
        return validated, None
    if err == "too_large":
        return None, err

    # Landing-page rescue: parse the HTML for a real PDF URL.
    if not _looks_like_html(body):
        return None, err
    rescue_url = _extract_pdf_url_from_html(body, final_url)
    if rescue_url is None or rescue_url == final_url:
        return None, "not_pdf"
    log.info(
        "download_pdf_bytes: HTML landing page, rescuing via %s", rescue_url,
    )
    try:
        resp2 = http.get(rescue_url)
        resp2.raise_for_status()
    except httpx.HTTPError as exc:
        log.info(
            "download_pdf_bytes: rescue request failed (%s): %s",
            rescue_url, exc,
        )
        return None, "download_failed"
    return _validate_pdf_body(resp2.content, rescue_url, max_size_bytes)


def parse_pdf_bytes(body: bytes, *, max_chars: int | None = None) -> str | None:
    """Extract reading-order text from a PDF byte string.

    Parser chain:

      1. **GROBID** — if ``$CITECLAW_GROBID_URL`` is set, POST the PDF
         to a GROBID server and return body + structured reference list.
         GROBID gives cleaner body text (no header/footer noise, no
         column-break artefacts) and delivers a properly-structured
         reference list instead of relying on the heuristic splitter
         downstream. See :mod:`citeclaw.clients.grobid`.
      2. **PyMuPDF** — falls back to ``pymupdf.get_text("text", sort=True)``
         for reading-order extraction on two-column scientific PDFs.
      3. **pypdf** — final fallback if neither of the above is available.

    Returns ``None`` only when ALL parsers fail or the extracted text
    is empty.

    ``max_chars`` truncates the result if non-None — useful for
    LLM-screening callers that need to fit a context window. The
    ``fetch-pdfs`` CLI passes ``None`` so the on-disk ``.txt`` sibling
    contains the full body.
    """
    text: str | None = _try_grobid(body)
    if text is None:
        text = _try_pymupdf(body)
    if text is None:
        text = _try_pypdf(body)
    if text is None:
        return None
    text = text.strip() or None
    if text and max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def _try_grobid(body: bytes) -> str | None:
    """GROBID path — only runs when ``$CITECLAW_GROBID_URL`` is set."""
    from citeclaw.clients.grobid import grobid_url, parse_pdf_with_grobid
    if grobid_url() is None:
        return None
    try:
        return parse_pdf_with_grobid(body)
    except Exception as exc:  # noqa: BLE001
        log.info("parse_pdf_bytes: grobid path failed: %s", exc)
        return None


def _try_pymupdf(body: bytes) -> str | None:
    """PyMuPDF extraction. Returns None on import failure or parse error."""
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        doc = pymupdf.open(stream=body, filetype="pdf")
    except Exception as exc:
        log.info("parse_pdf_bytes: pymupdf open failed: %s", exc)
        return None
    try:
        chunks: list[str] = []
        for page in doc:
            try:
                # ``sort=True`` reorders text blocks by reading order,
                # which is the entire reason we prefer pymupdf over
                # pypdf for two-column scientific PDFs.
                chunks.append(page.get_text("text", sort=True) or "")
            except Exception:
                continue
        return "\n".join(chunks)
    except Exception as exc:
        log.info("parse_pdf_bytes: pymupdf parse failed: %s", exc)
        return None
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _try_pypdf(body: bytes) -> str | None:
    """pypdf fallback. Returns None on import failure or parse error."""
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        reader = pypdf.PdfReader(io.BytesIO(body))
        chunks: list[str] = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks)
    except Exception as exc:
        log.info("parse_pdf_bytes: pypdf fallback failed: %s", exc)
        return None


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
        # the metadata URL. Built via the shared factory so the
        # ``fetch-pdfs`` CLI uses identical settings.
        self._http = make_pdf_http_client(timeout_s=timeout_s)

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

        body, dl_err = download_pdf_bytes(
            self._http, url, max_size_bytes=self._max_size_bytes,
        )
        if dl_err is not None:
            log.info(
                "PdfFetcher: download error %r for %s (%s)",
                dl_err, paper.paper_id, url,
            )
            return None, dl_err

        text = parse_pdf_bytes(body, max_chars=self._max_text_chars)
        if text is None:
            log.info("PdfFetcher: parse failed for %s (%s)", paper.paper_id, url)
            return None, "parse_failed"
        return text, None
