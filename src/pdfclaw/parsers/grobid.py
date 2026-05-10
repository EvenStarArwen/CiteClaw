"""GROBID-backed :class:`Parser` — best-in-class structured references.

`GROBID <https://github.com/kermitt2/grobid>`_ is a Java/ML tool
purpose-built for parsing scientific PDFs.  Unlike the in-process
engines (PyMuPDF / Docling) it runs as an HTTP service: this parser
is a thin client that POSTs a PDF and parses the returned TEI XML.

The flagship feature is **structured reference parsing**: GROBID
returns each bibliography entry as a TEI ``<biblStruct>`` with
typed fields (title, authors, year, DOI, raw text) — significantly
more reliable than the heuristic line-splitting the other engines
do.  Body text is also cleaner than PyMuPDF's because TEI extraction
strips headers / footers / page numbers automatically.

Tables are not GROBID's strong suit; if your downstream extraction
needs table parsing, prefer ``"docling"``.

The repository's :mod:`modal_grobid_server` deploys a GROBID instance
to Modal in a single command (see its module docstring).  Pass the
deployed URL via the ``base_url`` constructor argument; for ad-hoc
runs the ``PDFCLAW_GROBID_URL`` env var is consulted as a fallback so
the same instance can be picked up without plumbing the URL through
every CLI invocation.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from pdfclaw.parsers.base import ParseResult, ParserError

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger("pdfclaw.parsers.grobid")

# TEI namespace — every GROBID response uses this.
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}

# Default per-request timeout.  GROBID on CPU processes a 10-page
# paper in ~5s; 30+ page papers with hundreds of references can take
# 30s+ on a cold server.  120s leaves comfortable headroom.
_DEFAULT_TIMEOUT_S = 180.0

# Env var used as a fallback when ``base_url`` is not supplied.  The
# ``PDFCLAW_`` prefix matches the rest of the package's env-var
# namespace (``PDFCLAW_LLM_BASE_URL`` etc.).  We keep the legacy
# ``CITECLAW_GROBID_URL`` accepted too so existing deployments don't
# break, but new docs only mention the new name.
_ENV_PRIMARY = "PDFCLAW_GROBID_URL"
_ENV_LEGACY = "CITECLAW_GROBID_URL"


class GrobidParser:
    """Parse PDFs by POSTing to a GROBID server."""

    name = "grobid"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        include_raw_citations: bool = True,
    ) -> None:
        """Construct a GROBID client.

        Parameters
        ----------
        base_url
            GROBID server URL (no trailing slash needed).  When
            ``None``, ``PDFCLAW_GROBID_URL`` (or the legacy
            ``CITECLAW_GROBID_URL``) env var is consulted.  If neither
            is set, :meth:`parse` raises :class:`ParserError` on the
            first call rather than at construction so that test
            harnesses can build a parser instance without yet having
            a server to point it at.
        timeout_s
            Per-request HTTP timeout.
        include_raw_citations
            Ask GROBID to attach the original bibliography line as a
            ``<note type="raw_reference">`` on every ``<biblStruct>``.
            Preserves the paper's own numbering inside each entry —
            essential for the downstream reference extractor, which
            needs to map paper-body markers like ``[67]`` back to the
            correct bibliography entry.
        """
        self._base_url = (base_url or "").rstrip("/")
        self._timeout_s = timeout_s
        self._include_raw_citations = include_raw_citations

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        url = self._resolved_base_url()
        if not url:
            raise ParserError(
                "GrobidParser needs a server URL. Pass base_url=... or set "
                f"{_ENV_PRIMARY}=https://... before calling."
            )

        try:
            tei_xml = self._post(url, pdf_bytes)
        except ParserError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrapped intentionally
            raise ParserError(f"grobid request failed: {exc}") from exc

        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:
            raise ParserError(f"grobid returned malformed TEI XML: {exc}") from exc

        body_lines = _extract_body(root)
        ref_lines = _extract_references(root)
        body_text = "\n\n".join(body_lines).strip()

        return ParseResult(
            body_text=body_text,
            references=ref_lines,
            n_pages=0,  # TEI doesn't carry page count reliably
            metadata=_extract_metadata(root),
            parser_used=self.name,
        )

    # ------------------------------------------------------------------
    # HTTP I/O
    # ------------------------------------------------------------------

    def _resolved_base_url(self) -> str:
        """Return the configured URL, consulting env vars as fallback."""
        if self._base_url:
            return self._base_url
        for env in (_ENV_PRIMARY, _ENV_LEGACY):
            v = (os.environ.get(env) or "").strip().rstrip("/")
            if v:
                return v
        return ""

    def _post(self, base_url: str, pdf_bytes: bytes) -> str:
        """POST the PDF and return TEI XML on success.

        Imports ``httpx`` lazily so importing this module never pulls
        in the HTTP stack until a parser is actually used — keeps
        ``pdfclaw.parsers`` import time low.
        """
        import httpx  # noqa: PLC0415

        endpoint = f"{base_url}/api/processFulltextDocument"
        files = {"input": ("paper.pdf", pdf_bytes, "application/pdf")}
        data = {
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            "includeRawCitations": "1" if self._include_raw_citations else "0",
        }
        with httpx.Client(timeout=self._timeout_s) as http:
            resp = http.post(endpoint, files=files, data=data)
            resp.raise_for_status()
            return resp.text


# ---------------------------------------------------------------------
# TEI walkers — body / refs / metadata extraction.
#
# The walker functions are module-level (not methods) because they are
# stateless and easier to unit-test individually.  They mirror the
# original ``citeclaw.clients.grobid`` helpers verbatim aside from
# splitting body and refs into separate fields on :class:`ParseResult`
# rather than concatenating them with a "References" delimiter.
# ---------------------------------------------------------------------


def _extract_body(root: ET.Element) -> list[str]:
    body = root.find(".//tei:text/tei:body", _NS)
    if body is None:
        return []
    lines: list[str] = []
    for div in body.findall(".//tei:div", _NS):
        head = div.find("tei:head", _NS)
        if head is not None:
            head_text = _plain(head)
            if head_text:
                lines.append(head_text)
        for p in div.findall("tei:p", _NS):
            p_text = _plain(p)
            if p_text:
                lines.append(p_text)
    return lines


def _extract_references(root: ET.Element) -> list[str]:
    """One string per ``<biblStruct>``, preferring raw citation text.

    Raw text preserves the paper's own bibliography numbering inside
    each entry (e.g. ``[67] Suzek, ...``).  This matters because the
    paper body uses numeric citation markers that must map back to
    the correct bibliography entry — if we renumber the list
    sequentially, ``[67]`` in the body could map to a different paper.
    """
    refs: list[str] = []
    for idx, bibl in enumerate(
        root.findall(".//tei:listBibl/tei:biblStruct", _NS), start=1
    ):
        note = bibl.find(".//tei:note[@type='raw_reference']", _NS)
        if note is not None and note.text and note.text.strip():
            refs.append(" ".join(note.text.split()))
            continue

        # Structured-fallback path — reconstruct a citation line from
        # the parsed TEI fields when raw_reference is absent.
        parts: list[str] = []
        author_str = _format_authors(bibl)
        if author_str:
            parts.append(author_str)

        title = bibl.find(".//tei:analytic/tei:title", _NS)
        if title is None:
            title = bibl.find(".//tei:monogr/tei:title", _NS)
        if title is not None:
            title_text = _plain(title)
            if title_text:
                parts.append(title_text)

        venue = bibl.find(".//tei:monogr/tei:title", _NS)
        if venue is not None and title is not None and venue is not title:
            venue_text = _plain(venue)
            if venue_text:
                parts.append(venue_text)

        year = _extract_year(bibl)
        if year:
            parts.append(year)

        doi = _extract_doi(bibl)
        if doi:
            parts.append(doi)

        if parts:
            refs.append(f"[{idx}] {'. '.join(parts)}")
    return refs


def _extract_metadata(root: ET.Element) -> dict[str, str]:
    """Lift the title and author strings out of the TEI header."""
    out: dict[str, str] = {}
    title = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", _NS)
    if title is not None:
        title_text = _plain(title)
        if title_text:
            out["title"] = title_text
    authors = root.findall(
        ".//tei:teiHeader//tei:sourceDesc//tei:analytic/tei:author/tei:persName",
        _NS,
    )
    names = [n for n in (_format_persname(a) for a in authors) if n]
    if names:
        out["authors"] = "; ".join(names)
    return out


def _format_authors(bibl: ET.Element) -> str | None:
    """Format up to 3 authors as ``"F. Surname, F. Surname, F. Surname et al."``."""
    authors = bibl.findall(".//tei:analytic/tei:author/tei:persName", _NS)
    if not authors:
        authors = bibl.findall(".//tei:monogr/tei:author/tei:persName", _NS)
    if not authors:
        return None
    names = [n for n in (_format_persname(a) for a in authors[:3]) if n]
    if not names:
        return None
    out = ", ".join(names)
    if len(authors) > 3:
        out += " et al."
    return out


def _format_persname(persname: ET.Element) -> str:
    surname = persname.find("tei:surname", _NS)
    forename = persname.find("tei:forename", _NS)
    name = ""
    if forename is not None and forename.text:
        name += forename.text[0] + ". "
    if surname is not None and surname.text:
        name += surname.text
    return name.strip()


def _extract_year(bibl: ET.Element) -> str | None:
    date = bibl.find(".//tei:monogr/tei:imprint/tei:date", _NS)
    if date is None:
        return None
    raw = date.get("when") or _plain(date)
    if not raw:
        return None
    return str(raw)[:4]


def _extract_doi(bibl: ET.Element) -> str | None:
    doi = bibl.find(".//tei:idno[@type='DOI']", _NS)
    if doi is None or not doi.text:
        return None
    return f"doi:{doi.text.strip()}"


def _plain(el: ET.Element) -> str:
    """Return all text inside *el*, collapsed to a single space-joined string."""
    chunks: list[str] = []
    if el.text:
        chunks.append(el.text)
    for sub in el:
        if sub.text:
            chunks.append(sub.text)
        if sub.tail:
            chunks.append(sub.tail)
    return " ".join(" ".join(chunks).split())
