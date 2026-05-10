"""PDF parser registry — pick an engine by string identifier.

Three engines ship in this package:

* ``"pymupdf"`` — PyMuPDF.  Always available (already in
  ``pdfclaw[browser]`` extras).  Fast (~50 ms / paper), CPU only,
  no external services.  Quality is fine on single-column PDFs;
  multi-column reading order is approximate, tables come out as
  word-salad.  Pick this for tests, CI, and any time speed matters.

* ``"docling"`` — IBM Docling, a layout-aware vision-light parser.
  ``pip install docling`` adds a single optional dependency, runs on
  CPU (slow, ~tens of seconds / paper) or GPU (fast).  Strong on
  multi-column reading order and tables (rendered as Markdown);
  references are section-aware but not structured.  Pick this for
  production runs where extraction quality matters.

* ``"grobid"`` — GROBID HTTP client.  Requires a running GROBID
  server (the project's ``modal_grobid_server.py`` deploys one to
  Modal in one command).  Best-in-class for **structured** references
  (TEI XML with title / authors / year / DOI fields); body text is
  cleaner than PyMuPDF; tables are weak.  Pick this when reference
  resolution accuracy matters more than table parsing.

Selection precedence in :func:`parse` follows the
:func:`citeclaw.clients.llm.factory.build_llm_client` pattern:
explicit ``parser=`` argument → registry lookup → engine
construction → call.  No silent fallbacks: if an engine fails to
import or its server is unreachable, the call raises
:class:`ParserError` rather than quietly downgrading to a different
engine.  Callers that want a fallback chain compose one explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from pdfclaw.parsers.base import Parser, ParseResult, ParserError
from pdfclaw.parsers.docling import DoclingParser
from pdfclaw.parsers.grobid import GrobidParser
from pdfclaw.parsers.pymupdf import PyMuPDFParser

PdfInput = Union[bytes, str, Path]
"""Accepted input types for :func:`parse`.  ``bytes`` are taken
verbatim; ``str`` and ``Path`` are read from disk first."""

PARSER_REGISTRY: dict[str, type[Parser]] = {
    "pymupdf": PyMuPDFParser,
    "docling": DoclingParser,
    "grobid": GrobidParser,
}
"""Stable registry of parser names → classes.

Adding a new engine is a one-line dict entry plus a sibling module
implementing :class:`Parser`.  The dict is mutable on purpose: tests
and downstream consumers can register custom parsers at import time
the same way :mod:`citeclaw.filters.builder` lets users register new
filter blocks."""


__all__ = [
    "PARSER_REGISTRY",
    "Parser",
    "ParseResult",
    "ParserError",
    "DoclingParser",
    "GrobidParser",
    "PyMuPDFParser",
    "get_parser",
    "list_parsers",
    "parse",
]


def list_parsers() -> list[str]:
    """Return registry keys in insertion order — i.e., quality tier
    from lightweight to heavy."""
    return list(PARSER_REGISTRY.keys())


def get_parser(name: str = "pymupdf", **kwargs: Any) -> Parser:
    """Construct the parser registered under *name*.

    Engine-specific kwargs (``base_url`` for ``"grobid"``, ``do_ocr``
    for ``"docling"``, …) are passed through to the parser class's
    constructor.  Unknown kwargs raise ``TypeError`` from the
    constructor — consistent with the way
    :class:`~citeclaw.clients.llm.factory.build_llm_client` surfaces
    bad config rather than silently ignoring it.
    """
    try:
        cls = PARSER_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(
            f"Unknown parser {name!r}. Known parsers: {known}"
        ) from exc
    return cls(**kwargs)


def parse(
    pdf: PdfInput,
    *,
    parser: str = "pymupdf",
    max_chars: int | None = None,
    **parser_kwargs: Any,
) -> ParseResult:
    """One-shot parse: input → :class:`ParseResult`.

    Parameters
    ----------
    pdf
        Either raw PDF bytes, or a path (``str`` / :class:`pathlib.Path`)
        from which bytes will be read.
    parser
        Registry key; see module docstring for available engines.
    max_chars
        Optional cap on :attr:`ParseResult.body_text`.  Engines run to
        completion regardless; truncation happens after parsing so
        downstream callers don't need to plumb the cap through.
    **parser_kwargs
        Forwarded to the parser constructor (e.g. ``base_url=...``).

    Raises
    ------
    ParserError
        Re-raised from the engine when parsing fails.
    """
    if isinstance(pdf, (str, Path)):
        pdf_bytes = Path(pdf).expanduser().read_bytes()
    else:
        pdf_bytes = pdf

    engine = get_parser(parser, **parser_kwargs)
    result = engine.parse(pdf_bytes)
    if max_chars is not None and len(result.body_text) > max_chars:
        result.body_text = result.body_text[:max_chars]
    return result
