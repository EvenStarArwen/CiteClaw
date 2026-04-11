"""GROBID REST client — structure-aware PDF → body text + references.

GROBID is a Java/ML tool purpose-built for parsing scientific PDFs. It
delivers much cleaner body text than PyMuPDF (no header/footer noise,
no column-break artefacts, no mangled author affiliations) AND it
parses the reference list into structured entries with titles, authors,
years, and DOIs — the exact fields :class:`ExpandByPDF` needs to
resolve references against Semantic Scholar.

This module is intentionally thin: one function,
:func:`parse_pdf_with_grobid`, that POSTs a PDF to a GROBID server and
returns a single string in the shape :func:`citeclaw.clients.pdf.parse_pdf_bytes`
already emits — body text, followed by a ``References`` heading, followed
by the formatted reference list. This lets the downstream
:func:`~citeclaw.agents.pdf_reference_extractor.split_references` heuristic
pick up the cleanly-delimited references without any schema changes.

The client is used only when ``CITECLAW_GROBID_URL`` is set in the
environment. If unset or the GROBID call fails, the PyMuPDF fallback
in :mod:`citeclaw.clients.pdf` runs unchanged.

GROBID can be hosted on Modal via ``modal_grobid_server.py`` at the
project root (CPU-only, ``lfoppiano/grobid:0.8.2-crf``).
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx

log = logging.getLogger("citeclaw.grobid")

# TEI namespace — GROBID always uses this.
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

# Default timeout — GROBID on CPU processes a 10-page paper in ~5s but
# a 30-page paper with many references can take 30s+ on first-request
# warmup.
DEFAULT_TIMEOUT_S = 180.0

# Environment variable that enables GROBID. When set, the parser in
# ``citeclaw.clients.pdf`` tries GROBID first and falls back to PyMuPDF
# on any failure.
ENV_GROBID_URL = "CITECLAW_GROBID_URL"


def grobid_url() -> str | None:
    """Return the configured GROBID base URL, or ``None`` if unset."""
    url = os.environ.get(ENV_GROBID_URL, "").strip().rstrip("/")
    return url or None


def parse_pdf_with_grobid(
    pdf_bytes: bytes,
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    include_raw_citations: bool = True,
) -> str | None:
    """Parse a PDF via GROBID and return a plain-text body + refs string.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF contents.
    base_url:
        GROBID server base URL. Defaults to ``$CITECLAW_GROBID_URL``.
    timeout_s:
        Per-request timeout.
    include_raw_citations:
        If True, requests ``consolidateCitations=0`` + ``includeRawCitations=1``
        so GROBID returns the original bibliography entry text as well
        as the structured fields. This gives the downstream reference
        extractor two chances to match a title.

    Returns
    -------
    str | None
        A string shaped like the current :func:`parse_pdf_bytes` output —
        body paragraphs joined with blank lines, followed by a
        ``References`` heading, followed by one reference per line.
        Returns ``None`` on any network / parse error so callers can
        fall through to the PyMuPDF path without special-casing.
    """
    url = (base_url or grobid_url())
    if not url:
        return None

    endpoint = f"{url}/api/processFulltextDocument"
    files = {"input": ("paper.pdf", pdf_bytes, "application/pdf")}
    data = {
        # Structured reference entries (we need the structured form
        # to pull a clean title out for S2 search_match).
        "consolidateHeader": "0",
        "consolidateCitations": "0",
        "includeRawCitations": "1" if include_raw_citations else "0",
    }
    try:
        with httpx.Client(timeout=timeout_s) as http:
            resp = http.post(endpoint, files=files, data=data)
            resp.raise_for_status()
            tei_xml = resp.text
    except httpx.HTTPError as exc:
        log.warning("GROBID request failed (%s): %s", endpoint, exc)
        return None

    try:
        return _tei_to_text(tei_xml)
    except ET.ParseError as exc:
        log.warning("GROBID returned malformed TEI XML: %s", exc)
        return None


def _tei_to_text(tei_xml: str) -> str | None:
    """Convert a GROBID TEI document into a plain-text body + refs string."""
    root = ET.fromstring(tei_xml)
    body_lines = _extract_body(root)
    ref_lines = _extract_references(root)

    parts: list[str] = []
    if body_lines:
        parts.append("\n\n".join(body_lines))
    if ref_lines:
        parts.append("References")
        parts.append("\n".join(ref_lines))
    text = "\n\n".join(parts).strip()
    return text or None


def _extract_body(root: ET.Element) -> list[str]:
    """Flatten all <div> text in <body> into paragraph strings."""
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []
    lines: list[str] = []
    for div in body.findall(".//tei:div", NS):
        # Section heading
        head = div.find("tei:head", NS)
        if head is not None:
            head_text = _plain(head)
            if head_text:
                lines.append(head_text)
        # Paragraphs
        for p in div.findall("tei:p", NS):
            p_text = _plain(p)
            if p_text:
                lines.append(p_text)
    return lines


def _extract_references(root: ET.Element) -> list[str]:
    """Format every ``<biblStruct>`` in the back matter into a ref line.

    Prefers the raw bibliography text when GROBID captured it via
    ``<note type="raw_reference">`` (set ``includeRawCitations=1`` on
    the request). Raw text preserves the original in-paper numbering
    — which is critical because the paper body uses numbered citation
    markers (``[67]``, ``Uniref90 67``, …) that must map back to the
    correct bibliography entry. If we renumber the list ourselves, the
    LLM sees "[67]" in the body and looks up our sequential [67] in
    the refs list, which may point to an entirely different paper.

    Falls back to structured field assembly (authors / title / venue
    / year / DOI) when ``raw_reference`` is missing.
    """
    refs: list[str] = []
    for idx, bibl in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", NS), start=1):
        # Prefer the raw citation text — preserves the paper's own
        # numbering inside each entry (e.g. ``[67] Suzek, ...``).
        note = bibl.find(".//tei:note[@type='raw_reference']", NS)
        if note is not None and note.text and note.text.strip():
            raw = " ".join(note.text.split())
            refs.append(raw)
            continue

        # Structured fallback — reconstruct a plausible citation line.
        parts: list[str] = []

        # Authors — up to 3, then "et al."
        authors = bibl.findall(".//tei:analytic/tei:author/tei:persName", NS)
        if not authors:
            authors = bibl.findall(".//tei:monogr/tei:author/tei:persName", NS)
        if authors:
            names: list[str] = []
            for a in authors[:3]:
                surname = a.find("tei:surname", NS)
                forename = a.find("tei:forename", NS)
                name = ""
                if forename is not None and forename.text:
                    name += forename.text[0] + ". "
                if surname is not None and surname.text:
                    name += surname.text
                if name:
                    names.append(name.strip())
            if names:
                author_str = ", ".join(names)
                if len(authors) > 3:
                    author_str += " et al."
                parts.append(author_str)

        # Article title (analytic) or book/journal title (monogr)
        title = bibl.find(".//tei:analytic/tei:title", NS)
        if title is None:
            title = bibl.find(".//tei:monogr/tei:title", NS)
        if title is not None:
            title_text = _plain(title)
            if title_text:
                parts.append(title_text)

        # Venue — monogr title when analytic title was found
        venue = bibl.find(".//tei:monogr/tei:title", NS)
        if venue is not None and title is not None and venue is not title:
            venue_text = _plain(venue)
            if venue_text:
                parts.append(venue_text)

        # Year
        date = bibl.find(".//tei:monogr/tei:imprint/tei:date", NS)
        year = None
        if date is not None:
            year = date.get("when") or _plain(date)
        if year:
            parts.append(str(year)[:4])

        # DOI
        doi = bibl.find(".//tei:idno[@type='DOI']", NS)
        if doi is not None and doi.text:
            parts.append(f"doi:{doi.text.strip()}")

        if not parts:
            continue

        # Sequential [N] prefix is a best-effort fallback only — may not
        # match the paper's own numbering. Prefer raw_reference path above.
        refs.append(f"[{idx}] {'. '.join(parts)}")
    return refs


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
