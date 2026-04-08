"""Convert raw S2 paper / edge dicts to PaperRecord."""

from __future__ import annotations

from typing import Any

from citeclaw.models import PaperRecord


def paper_to_record(data: dict[str, Any]) -> PaperRecord | None:
    pid = data.get("paperId")
    if not pid:
        return None
    pdf_blob = data.get("openAccessPdf") or {}
    pdf_url = pdf_blob.get("url") if isinstance(pdf_blob, dict) else None
    raw_authors = data.get("authors") or []
    authors: list[dict] = []
    if isinstance(raw_authors, list):
        for a in raw_authors:
            if not isinstance(a, dict):
                continue
            authors.append({
                "authorId": a.get("authorId"),
                "name": a.get("name") or "",
            })
    # ``externalIds`` maps source → id (e.g. ``{"DOI": "10.1/abc", "ArXiv": "..."}``).
    # Normalise values to strings; S2 occasionally returns ints for PubMed ids.
    raw_ext = data.get("externalIds") or {}
    external_ids: dict[str, str] = {}
    if isinstance(raw_ext, dict):
        for k, v in raw_ext.items():
            if v is None:
                continue
            external_ids[str(k)] = str(v)
    # ``fieldsOfStudy`` (legacy flat list of strings) merged with
    # ``s2FieldsOfStudy`` (newer list of ``{"category", "source"}`` dicts).
    # Order is preserved (legacy first, then S2 categories) and duplicates
    # are dropped via a seen-set.
    fields_of_study: list[str] = []
    seen_fos: set[str] = set()
    fos_legacy = data.get("fieldsOfStudy") or []
    if isinstance(fos_legacy, list):
        for f in fos_legacy:
            if isinstance(f, str) and f and f not in seen_fos:
                seen_fos.add(f)
                fields_of_study.append(f)
    fos_s2 = data.get("s2FieldsOfStudy") or []
    if isinstance(fos_s2, list):
        for entry in fos_s2:
            if isinstance(entry, dict):
                cat = entry.get("category")
                if isinstance(cat, str) and cat and cat not in seen_fos:
                    seen_fos.add(cat)
                    fields_of_study.append(cat)
    # ``publicationTypes`` is a flat list of S2's enum strings.
    pub_types_raw = data.get("publicationTypes") or []
    publication_types: list[str] = []
    if isinstance(pub_types_raw, list):
        publication_types = [t for t in pub_types_raw if isinstance(t, str) and t]
    return PaperRecord(
        paper_id=pid,
        title=data.get("title") or "",
        abstract=data.get("abstract"),
        year=data.get("year"),
        venue=data.get("venue") or None,
        citation_count=data.get("citationCount"),
        influential_citation_count=data.get("influentialCitationCount"),
        references=[
            r["citedPaper"]["paperId"]
            for r in (data.get("references") or [])
            if r.get("citedPaper") and r["citedPaper"].get("paperId")
        ],
        pdf_url=pdf_url or None,
        authors=authors,
        external_ids=external_ids,
        fields_of_study=fields_of_study,
        publication_types=publication_types,
    )


def edge_to_record(edge: dict[str, Any], key: str) -> PaperRecord | None:
    """Convert an S2 ``references`` or ``citations`` edge to a PaperRecord.

    ``key`` is ``citedPaper`` (for references) or ``citingPaper`` (citations).
    """
    inner = edge.get(key)
    if not inner or not inner.get("paperId"):
        return None
    return PaperRecord(
        paper_id=inner["paperId"],
        title=inner.get("title") or "",
        abstract=inner.get("abstract"),
        year=inner.get("year"),
        venue=inner.get("venue") or None,
        citation_count=inner.get("citationCount"),
    )
