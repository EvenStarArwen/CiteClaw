"""BibTeX writer."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.output.bibtex")


def _sanitize_key(paper_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", paper_id)


def write_bibtex(papers: list[PaperRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for p in papers:
        key = _sanitize_key(p.paper_id)
        title = (p.title or "").replace("{", "\\{").replace("}", "\\}")
        venue = (p.venue or "").replace("{", "\\{").replace("}", "\\}")
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
