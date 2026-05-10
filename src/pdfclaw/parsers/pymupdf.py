"""PyMuPDF-backed :class:`Parser` — the lightweight default engine.

PyMuPDF (``import pymupdf``, formerly ``import fitz``) is a thin
binding around the MuPDF C library.  It parses PDFs directly from
bytes without needing any external service or model weights, and it's
already a top-level dependency of ``pdfclaw[browser]`` so this engine
is always available.

Quality is fine on single-column PDFs and on most modern two-column
papers (PyMuPDF's text extraction infers reading order from glyph
positions).  Pathological cases — scanned PDFs, papers with heavy
side-bars, multi-table layouts — produce noisier output; the
``"docling"`` engine in this package is the recommended upgrade.

This engine never populates :attr:`ParseResult.references` or
:attr:`ParseResult.tables`.  Body text contains everything the
underlying library extracts; downstream
:func:`citeclaw.steps._pdf_reference_extractor.split_references` is
the one that splits body from references heuristically.
"""

from __future__ import annotations

import logging
from typing import Any

from pdfclaw.parsers.base import ParseResult, ParserError

log = logging.getLogger("pdfclaw.parsers.pymupdf")


class PyMuPDFParser:
    """Parse PDFs with PyMuPDF.  Stateless beyond construction."""

    name = "pymupdf"

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        try:
            import pymupdf  # noqa: PLC0415
        except ImportError as exc:
            raise ParserError(
                "pymupdf is not installed. Run `pip install pymupdf` or "
                "`pip install citeclaw[pdf]` to add it."
            ) from exc

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 — wrapped intentionally
            raise ParserError(f"pymupdf failed to open PDF: {exc}") from exc

        try:
            pages: list[str] = []
            for page in doc:
                try:
                    pages.append(page.get_text("text") or "")
                except Exception as exc:  # noqa: BLE001
                    # A single bad page must not drop the whole document —
                    # PDFs with embedded images or unusual fonts can poison
                    # one page while the rest extract cleanly.  Logged at
                    # DEBUG so the audit trail exists without being noisy.
                    log.debug(
                        "pymupdf: per-page extract failed (page %s): %s",
                        getattr(page, "number", "?"), exc,
                    )
                    pages.append("")
            body = "\n\n".join(pages).strip()
            metadata = self._normalise_metadata(dict(doc.metadata or {}))
            n_pages = int(doc.page_count)
        finally:
            doc.close()

        return ParseResult(
            body_text=body,
            n_pages=n_pages,
            metadata=metadata,
            parser_used=self.name,
        )

    @staticmethod
    def _normalise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
        """Drop empty values and stringify what remains.

        PyMuPDF's metadata dict has predictable keys (``title``,
        ``author``, ``creator``, …) but values are sometimes
        ``None`` / ``""`` / ``b""``.  Filtering here keeps
        :class:`ParseResult.metadata` free of noise so downstream
        consumers can iterate without per-key None-checks.
        """
        return {
            str(k): str(v)
            for k, v in raw.items()
            if v not in (None, "", b"")
        }
