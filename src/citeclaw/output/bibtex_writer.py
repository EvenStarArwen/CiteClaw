"""BibTeX writer for ``literature_collection.bib``.

Produces one ``@article`` entry per accepted paper, keyed by the
sanitised S2 ``paper_id`` (alphanumeric only, BibTeX rejects
punctuation in citation keys). Title and venue values have ``{`` /
``}`` characters escaped so user-facing braces don't break BibTeX
parsing downstream. The ``note`` field carries the original
``paper_id`` and citation count so users can cross-reference the
exported BibTeX entries against the JSON output.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.output.bibtex")


def _sanitize_key(paper_id: str) -> str:
    """Strip every non-alphanumeric character from ``paper_id``.

    BibTeX citation keys can contain only ``[a-zA-Z0-9]``; an S2 paper
    id like ``5b8b8a13...`` (sha) survives unchanged but a DOI-style
    id like ``DOI:10.1038/...`` would otherwise fail to parse.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", paper_id)


def _escape_braces(s: str) -> str:
    """Backslash-escape ``{`` / ``}`` so they don't terminate the BibTeX value.

    Title / venue strings are wrapped in ``{...}`` in the output, so an
    unescaped brace inside the value (common in chemistry / math
    titles) would close the field early.
    """
    return s.replace("{", "\\{").replace("}", "\\}")


def write_bibtex(papers: list[PaperRecord], path: Path) -> None:
    """Write a BibTeX file with one ``@article`` per paper to ``path``.

    Creates the parent directory if missing. Entries are separated by
    a blank line; the file ends with a single trailing newline. Per
    paper: ``title`` (brace-escaped), ``year``, ``journal`` (= venue,
    brace-escaped), ``note`` (S2 id + citation count for traceability).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for p in papers:
        key = _sanitize_key(p.paper_id)
        title = _escape_braces(p.title or "")
        venue = _escape_braces(p.venue or "")
        entries.append(
            f"@article{{{key},\n"
            f"  title     = {{{title}}},\n"
            f"  year      = {{{p.year or ''}}},\n"
            f"  journal   = {{{venue}}},\n"
            f"  note      = {{S2: {p.paper_id}, citations: {p.citation_count or 0}}},\n"
            f"}}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
        f.write("\n")
    log.info("Wrote BibTeX: %s (%d entries)", path, len(entries))
