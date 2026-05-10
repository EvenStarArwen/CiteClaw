"""Parser Protocol and ``ParseResult`` value type.

Every PDF parsing engine — fast (PyMuPDF), high-quality CPU
(Docling), or HTTP-served (GROBID) — implements the same
:class:`Parser` Protocol so callers can swap engines via a string
identifier.  The Protocol mirrors :class:`citeclaw.clients.llm.LLMClient`
in spirit: a tiny surface (one method, one stable property) that
hides every provider-specific quirk behind a uniform return type.

The return type is :class:`ParseResult` — a dataclass carrying the
fields any extraction-facing consumer in CiteClaw cares about (body
text, references, tables, page count, metadata).  Engines that don't
support a given field leave it at its dataclass default (empty list /
empty dict / 0); callers that ask for "tables" from a parser that
can't extract them get ``[]`` rather than a mysterious ``KeyError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ParserError(Exception):
    """Raised when a parser cannot produce body text from a PDF.

    Concrete parsers raise this (or a subclass) when their underlying
    library refuses the input — corrupt PDF, encrypted document,
    network failure to a remote engine, etc.  Callers that want to
    cascade through multiple engines catch :class:`ParserError`
    explicitly.
    """


@dataclass
class ParseResult:
    """One PDF parsed into a structured representation.

    All fields default to "empty" so an engine that can't fill a slot
    simply leaves it alone.  The shape is intentionally additive — new
    parsers can populate richer fields later without breaking
    consumers that only read :attr:`body_text`.
    """

    body_text: str = ""
    """Plain text of the document body, with reading order respected
    where the engine can preserve it.  Always non-``None``; the empty
    string is the sentinel for "engine returned nothing usable"."""

    references: list[str] = field(default_factory=list)
    """One entry per bibliography item.  Engines that capture
    structured references (GROBID) emit pre-formatted lines; engines
    that don't (PyMuPDF) leave this empty and the references stay
    inlined in :attr:`body_text`.  Downstream
    :func:`citeclaw.steps._pdf_reference_extractor.split_references`
    still works either way."""

    tables: list[str] = field(default_factory=list)
    """Each item is one table rendered as Markdown.  Engines that
    detect tables (Docling) populate this; PyMuPDF / GROBID leave it
    empty and tables remain spread across :attr:`body_text` as best
    they can."""

    n_pages: int = 0
    """Page count of the source PDF.  ``0`` when the engine couldn't
    determine it (some HTTP engines like GROBID return only a flat
    text blob)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Document metadata as a flat string→string-ish dict.  Engines
    map their native shape into this — PyMuPDF surfaces ``title /
    author / subject / creator``; Docling adds page dimensions; etc."""

    parser_used: str = ""
    """The name (registry key) of the parser that produced this
    result.  Useful for downstream logging and for asserting against
    in tests."""

    @property
    def n_chars(self) -> int:
        """Convenience: ``len(body_text)``.  Kept as a property so it
        always reflects the current body, even if a caller mutates."""
        return len(self.body_text)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the flat shape used by ``parsed/<id>.json``.

        Kept structurally close to the legacy ``parse_pdf_bytes`` dict
        (``n_pages`` / ``n_chars`` / ``body_text`` / ``meta``) so any
        on-disk artefact previously written by ``Fetcher._save_parsed``
        can still be read by the same code, plus the new fields
        (``references`` / ``tables`` / ``parser_used``) get added on
        top.  Existing readers that ignore unknown keys keep working;
        new readers benefit.
        """
        return {
            "parser_used": self.parser_used,
            "n_pages": int(self.n_pages),
            "n_chars": self.n_chars,
            "body_text": self.body_text,
            "references": list(self.references),
            "tables": list(self.tables),
            "meta": dict(self.metadata),
        }


@runtime_checkable
class Parser(Protocol):
    """Provider-agnostic PDF → :class:`ParseResult` engine.

    Concrete implementations live in sibling modules
    (:mod:`pdfclaw.parsers.pymupdf`, :mod:`.docling`, :mod:`.grobid`)
    and the :func:`pdfclaw.parsers.get_parser` factory picks one based
    on the registry name.

    Implementations may take engine-specific kwargs at construction
    (``base_url`` for GROBID, ``do_ocr`` for Docling, …) but
    :meth:`parse` itself takes only PDF bytes — no per-call config —
    so the same parser instance is safe to reuse across many papers.
    """

    name: str
    """Stable registry key (e.g. ``"pymupdf"``).  Class attribute so
    callers can introspect the engine without instantiating."""

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        """Parse a single PDF.

        Raises
        ------
        ParserError
            When the engine cannot produce body text from the input.
            Network / library exceptions native to the engine are
            wrapped before being re-raised.
        """
        ...
