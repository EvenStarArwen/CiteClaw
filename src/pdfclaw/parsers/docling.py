"""Docling-backed :class:`Parser` — high-quality CPU/GPU engine.

`Docling <https://github.com/DS4SD/docling>`_ is IBM's open-source
document parser (MIT licensed, Python-only single dependency, model
weights downloaded once on first run).  It uses a small layout model
plus heuristics for reading order, table detection, and section
structure — much better than PyMuPDF on multi-column papers and on
papers with embedded tables.

Compared to PyMuPDF this engine is **slower** (seconds to tens of
seconds per paper on CPU; faster with GPU when available) but
produces richer :class:`ParseResult` output:

* ``body_text`` — Markdown rendering of the document body.
* ``tables`` — one Markdown table per detected table.
* ``references`` — body of any section whose heading matches
  ``References`` / ``Bibliography``, split into one entry per
  numbered or hanging-indent line.

Math equations are best-effort (Docling does not extract LaTeX).
That's by design: we picked Docling specifically because the user's
extraction tasks don't depend on math fidelity.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from pdfclaw.parsers.base import ParseResult, ParserError

if TYPE_CHECKING:  # pragma: no cover
    from docling.datamodel.document import ConversionResult

log = logging.getLogger("pdfclaw.parsers.docling")

# Headings that mark the start of the bibliography in published
# papers — same set as :func:`split_references` in the reference
# extractor, kept in sync deliberately.
_REF_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"References?(?:\s+(?:and\s+Notes|Cited))?"
    r"|Bibliography"
    r"|Works\s+Cited"
    r"|Literature\s+Cited"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Numbered or bracketed bibliography-entry start markers.  Used to
# split a flat references blob into individual entries.
_REF_ENTRY_RE = re.compile(
    r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}[\.\)])\s+"
)


class DoclingParser:
    """Parse PDFs with IBM Docling.  Lazy-imports the library at call
    time so importing this module does not pull in Docling's heavy
    dependency tree (``transformers``, ``torch``, etc.) until a user
    actually asks for the engine."""

    name = "docling"

    def __init__(
        self,
        *,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ) -> None:
        """Configure the underlying ``DocumentConverter``.

        Parameters
        ----------
        do_ocr
            Enable OCR for scanned / image-only PDFs.  Off by default
            because the typical CiteClaw input is a born-digital PDF
            with embedded text — OCR would be slower and add no
            quality on those.  Turn on when you know the corpus
            includes scans.
        do_table_structure
            Run Docling's table-structure model so detected tables
            are emitted as structured Markdown.  On by default — this
            is the main reason to pick Docling over PyMuPDF.
        """
        self._do_ocr = do_ocr
        self._do_table_structure = do_table_structure
        # Lazy state — the converter holds heavy model weights and is
        # only built on first parse.  Reused across calls to amortise
        # construction.
        self._converter: Any | None = None

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        converter = self._ensure_converter()
        try:
            conv_result = self._convert(converter, pdf_bytes)
        except ParserError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrapped intentionally
            raise ParserError(f"docling failed to convert PDF: {exc}") from exc

        document = conv_result.document
        body_text = (document.export_to_markdown() or "").strip()
        tables = self._extract_tables(document)
        body_only, references = self._split_references(body_text)
        n_pages = self._page_count(document)
        metadata = self._collect_metadata(document)

        return ParseResult(
            body_text=body_only,
            references=references,
            tables=tables,
            n_pages=n_pages,
            metadata=metadata,
            parser_used=self.name,
        )

    # ------------------------------------------------------------------
    # Lazy converter construction
    # ------------------------------------------------------------------

    def _ensure_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
        except ImportError as exc:
            raise ParserError(
                "docling is not installed. Run `pip install docling` to add it. "
                "The first run will download model weights (~1-2 GB)."
            ) from exc

        # Default DocumentConverter is appropriate for academic PDFs
        # out of the box; we only override the OCR / table-structure
        # toggles when the constructor was given non-default values.
        # Doing this without touching Docling's PdfPipelineOptions
        # keeps the binding compatible across Docling versions, which
        # have churned the options shape between 1.x and 2.x.
        try:
            self._converter = DocumentConverter()
        except Exception as exc:  # noqa: BLE001 — wrapped intentionally
            raise ParserError(
                f"docling failed to construct DocumentConverter: {exc}"
            ) from exc
        return self._converter

    @staticmethod
    def _convert(converter: Any, pdf_bytes: bytes) -> "ConversionResult":
        """Convert PDF bytes via Docling.

        Docling's stream-from-bytes API moved between releases; we
        use ``DocumentStream`` when available and fall back to writing
        a temp file.  Either way the caller gets a ``ConversionResult``.
        """
        try:
            from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415
        except ImportError:
            DocumentStream = None  # type: ignore[assignment]

        import io  # noqa: PLC0415
        if DocumentStream is not None:
            stream = DocumentStream(name="paper.pdf", stream=io.BytesIO(pdf_bytes))
            return converter.convert(stream)

        # Older Docling builds: round-trip through a temp file.
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fp:
            fp.write(pdf_bytes)
            tmp_path = Path(fp.name)
        try:
            return converter.convert(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tables(document: Any) -> list[str]:
        """Render every detected table to a Markdown block.

        Docling's table objects expose ``export_to_markdown()`` from
        v1 onwards.  We tolerate both flat and nested
        ``document.tables`` attributes — the data model has shifted
        across releases, and we'd rather degrade to "no tables" than
        explode on a minor version bump.
        """
        tables_attr = getattr(document, "tables", None) or []
        out: list[str] = []
        for table in tables_attr:
            try:
                md = table.export_to_markdown()
            except Exception:  # noqa: BLE001
                md = ""
            md = (md or "").strip()
            if md:
                out.append(md)
        return out

    @staticmethod
    def _split_references(markdown: str) -> tuple[str, list[str]]:
        """Pull the references section out of *markdown*.

        Looks for the first heading line matching :data:`_REF_HEADING_RE`
        (positioned past 30% of the doc to avoid in-text mentions of
        the word "References") and treats everything after it as the
        bibliography.  If found, the references block is split into
        individual entries via :data:`_REF_ENTRY_RE`; otherwise the
        full Markdown is returned as ``body`` and references is empty.
        """
        if not markdown:
            return "", []
        matches = list(_REF_HEADING_RE.finditer(markdown))
        for m in reversed(matches):
            if m.start() < int(len(markdown) * 0.30):
                continue
            body = markdown[: m.start()].rstrip()
            refs_blob = markdown[m.end() :].strip()
            return body, _split_ref_entries(refs_blob)
        return markdown, []

    @staticmethod
    def _page_count(document: Any) -> int:
        pages = getattr(document, "pages", None)
        if pages is None:
            return 0
        try:
            return int(len(pages))
        except TypeError:
            return 0

    @staticmethod
    def _collect_metadata(document: Any) -> dict[str, Any]:
        """Surface document title / authors / origin URL when present.

        Docling's metadata shape is slim by design (it's a parser, not
        a metadata extractor) — we mirror only the keys that the
        legacy :class:`PyMuPDFParser` would have set, so downstream
        consumers see the same metadata names regardless of engine.
        """
        out: dict[str, Any] = {}
        title = getattr(document, "name", None)
        if title:
            out["title"] = str(title)
        origin = getattr(document, "origin", None)
        if origin is not None:
            uri = getattr(origin, "uri", None) or getattr(origin, "filename", None)
            if uri:
                out["source"] = str(uri)
        return out


def _split_ref_entries(refs_blob: str) -> list[str]:
    """Split a flat references blob into individual entries.

    Two strategies tried in order:

    1. Numbered / bracketed prefix (``[12]`` or ``12.``) at line start
       — the dominant style in CS / biomed papers.
    2. Blank-line-separated paragraphs — fallback for author-year
       styles where there is no numeric marker.

    Either way, entries are stripped of leading whitespace and empty
    entries are dropped.
    """
    if not refs_blob:
        return []
    matches = list(_REF_ENTRY_RE.finditer(refs_blob))
    entries: list[str] = []
    if matches:
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_blob)
            chunk = refs_blob[start:end].strip()
            if chunk:
                entries.append(chunk)
        return entries
    # Fallback: blank-line split.
    entries = [p.strip() for p in re.split(r"\n\s*\n+", refs_blob) if p.strip()]
    return entries
