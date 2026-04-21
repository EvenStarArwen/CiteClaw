"""Convert raw Semantic Scholar JSON dicts into :class:`PaperRecord` values.

Two entry points:

* :func:`paper_to_record` — full ``GET /paper/{id}`` response (also used
  for ``/paper/batch``, ``/paper/search`` items, etc.).
* :func:`edge_to_record` — one element of a ``references`` /
  ``citations`` array, where the actual paper sits behind a
  ``citedPaper`` / ``citingPaper`` indirection.

S2's API has been observed to return ``null`` for omitted list fields,
ints for PubMed ``externalIds``, and the occasional non-dict element in
otherwise-typed arrays. Every helper here defends against those shapes
so the upstream pipeline never sees a partially-malformed record.
"""

from __future__ import annotations

from typing import Any

from citeclaw.models import PaperRecord


def paper_to_record(data: dict[str, Any]) -> PaperRecord | None:
    """Convert one S2 paper-shaped dict into a :class:`PaperRecord`.

    Returns ``None`` when ``data`` lacks a ``paperId`` (the only field S2
    treats as load-bearing). Every other field is best-effort: missing
    keys map to the corresponding :class:`PaperRecord` default; malformed
    sub-shapes are filtered (see helpers below) rather than raising.
    """
    pid = data.get("paperId")
    if not pid:
        return None
    return PaperRecord(
        paper_id=pid,
        title=data.get("title") or "",
        abstract=data.get("abstract"),
        year=data.get("year"),
        venue=data.get("venue") or None,
        citation_count=data.get("citationCount"),
        influential_citation_count=data.get("influentialCitationCount"),
        references=_extract_reference_ids(data.get("references")),
        pdf_url=_extract_pdf_url(data.get("openAccessPdf")),
        authors=_normalize_authors(data.get("authors")),
        external_ids=_normalize_external_ids(data.get("externalIds")),
        fields_of_study=_merge_fields_of_study(
            data.get("fieldsOfStudy"), data.get("s2FieldsOfStudy")
        ),
        publication_types=_normalize_publication_types(data.get("publicationTypes")),
    )


def edge_to_record(edge: dict[str, Any], key: str) -> PaperRecord | None:
    """Convert one S2 ``references`` or ``citations`` edge to a :class:`PaperRecord`.

    ``key`` is ``"citedPaper"`` (for the ``references`` endpoint) or
    ``"citingPaper"`` (for ``citations``). Returns ``None`` when the
    inner paper is missing its ``paperId``. Edge endpoints return a
    smaller field set than the full paper endpoint — only the fields
    they actually populate are mapped.
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


# ---- helpers ---------------------------------------------------------------


def _extract_pdf_url(blob: Any) -> str | None:
    """Pull ``url`` from S2's ``openAccessPdf`` blob; ``None`` when absent."""
    if not isinstance(blob, dict):
        return None
    return blob.get("url") or None


def _extract_reference_ids(refs: Any) -> list[str]:
    """Flatten S2's ``references`` array to a list of cited ``paperId`` strings."""
    if not isinstance(refs, list):
        return []
    return [
        r["citedPaper"]["paperId"]
        for r in refs
        if isinstance(r, dict)
        and isinstance(r.get("citedPaper"), dict)
        and r["citedPaper"].get("paperId")
    ]


def _normalize_authors(raw: Any) -> list[dict]:
    """Normalize S2's ``authors`` list to ``[{"authorId", "name"}, ...]``.

    Drops non-dict entries; missing ``name`` defaults to empty string.
    ``authorId`` is passed through unchanged (may be ``None`` for
    authors S2 hasn't disambiguated yet).
    """
    if not isinstance(raw, list):
        return []
    return [
        {"authorId": a.get("authorId"), "name": a.get("name") or ""}
        for a in raw
        if isinstance(a, dict)
    ]


def _normalize_external_ids(raw: Any) -> dict[str, str]:
    """Coerce S2's ``externalIds`` mapping to ``str: str``.

    S2 occasionally returns ints for PubMed ids (e.g. ``"PubMed": 12345``);
    we coerce both keys and values to strings so downstream callers
    don't have to type-check. ``None`` values are dropped.
    """
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def _merge_fields_of_study(legacy: Any, s2: Any) -> list[str]:
    """Merge the legacy flat list and the newer ``{category, source}`` list.

    Order is preserved (legacy first, then S2 categories) and duplicates
    are dropped via a seen-set. Both inputs may be ``None`` (S2 returns
    ``null`` for omitted list fields), non-list (defended), or contain
    malformed entries (filtered).
    """
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(legacy, list):
        for f in legacy:
            if isinstance(f, str) and f and f not in seen:
                seen.add(f)
                out.append(f)
    if isinstance(s2, list):
        for entry in s2:
            if not isinstance(entry, dict):
                continue
            cat = entry.get("category")
            if isinstance(cat, str) and cat and cat not in seen:
                seen.add(cat)
                out.append(cat)
    return out


def _normalize_publication_types(raw: Any) -> list[str]:
    """Filter S2's ``publicationTypes`` to the non-empty string entries."""
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str) and t]
