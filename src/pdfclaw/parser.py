"""Minimal PDF → structured dict parser using PyMuPDF.

Mirrors ``citeclaw.clients.pdf.parse_pdf_bytes`` in spirit (PyMuPDF first,
fall back to pypdf) but kept self-contained so pdfclaw doesn't import
from citeclaw. We deliberately don't try to do clever section / reference
extraction — that's GROBID / Marker / Nougat territory and well outside
scope for v0. The body text and a handful of metadata fields are enough
for downstream LLM ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("pdfclaw.parser")

# Cap to protect against pathological PDFs (huge appendices, scanned
# images). 80K chars ≈ 25K BPE tokens — fits in a 32K context window.
MAX_BODY_CHARS = 200_000


def parse_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    """Parse PDF bytes into a JSON-serialisable dict.

    Returns ``{"n_pages", "body_text", "n_chars", "meta"}`` on success.
    Raises ``RuntimeError`` if both PyMuPDF and pypdf fail.
    """
    # PyMuPDF first
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            pymupdf = None  # type: ignore[assignment]

    if pymupdf is not None:
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                pages: list[str] = []
                for page in doc:
                    pages.append(page.get_text("text"))
                body = "\n\n".join(pages)
                meta_raw = dict(doc.metadata or {})
                n_pages = doc.page_count
            finally:
                doc.close()
            return _finalise(body, meta_raw, n_pages)
        except Exception as exc:  # noqa: BLE001
            log.warning("PyMuPDF parse failed (%s); trying pypdf fallback", exc)

    # pypdf fallback
    try:
        import pypdf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Neither pymupdf nor pypdf is installed; cannot parse PDF"
        ) from exc

    import io  # noqa: PLC0415

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    body = "\n\n".join(pages)
    meta_raw = {}
    if reader.metadata is not None:
        for k, v in reader.metadata.items():
            key = k.lstrip("/") if isinstance(k, str) else str(k)
            meta_raw[key] = str(v) if v is not None else ""
    return _finalise(body, meta_raw, len(reader.pages))


def _finalise(body: str, meta_raw: dict, n_pages: int) -> dict[str, Any]:
    body = body.strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS]
    # Drop empty / None metadata, normalise types
    meta = {
        str(k): str(v)
        for k, v in meta_raw.items()
        if v not in (None, "", b"")
    }
    return {
        "n_pages": int(n_pages),
        "n_chars": len(body),
        "body_text": body,
        "meta": meta,
    }
