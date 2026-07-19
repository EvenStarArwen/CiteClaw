"""Render S2AG dataset rows into graph-API-shaped records.

The dumps use lowercase field names (``citationcount``) while the API
speaks camelCase (``citationCount``); the mapper renders once at ingest
so the serving path never transforms — it just field-filters a stored
dict.
"""

from __future__ import annotations

from typing import Any

from s2mirror.schema import canonical_sha

# dataset externalids key -> API externalIds key (identity for all
# current keys, but kept explicit so a dump rename breaks loudly here).
_XID_KEYS = ("DOI", "ArXiv", "MAG", "ACL", "PubMed", "PubMedCentral", "DBLP", "CorpusId")


def render_paper(row: dict[str, Any]) -> tuple[int, str, dict[str, Any]] | None:
    """papers-dataset row -> (corpusid, canonical_sha, api_record).

    The record is the *full* superset CiteClaw's ``PAPER_FIELDS`` can
    request (abstract is merged in later by the reducer). Returns None
    when the row has no corpusid or no extractable canonical sha.
    """
    corpusid = row.get("corpusid")
    sha = canonical_sha(row.get("url"))
    if not isinstance(corpusid, int) or not sha:
        return None

    xids_raw = row.get("externalids") or {}
    xids = {k: xids_raw[k] for k in _XID_KEYS if xids_raw.get(k) is not None}
    xids.setdefault("CorpusId", str(corpusid))

    authors = [
        {"authorId": a.get("authorId"), "name": a.get("name") or ""}
        for a in (row.get("authors") or ())
        if isinstance(a, dict)
    ]
    s2fos = [
        {"category": f.get("category"), "source": f.get("source")}
        for f in (row.get("s2fieldsofstudy") or ())
        if isinstance(f, dict) and f.get("category")
    ]

    rec: dict[str, Any] = {
        "paperId": sha,
        "corpusId": corpusid,
        "externalIds": xids,
        "url": row.get("url"),
        "title": row.get("title") or "",
        "abstract": None,
        "venue": row.get("venue") or "",
        "year": row.get("year"),
        "referenceCount": row.get("referencecount"),
        "citationCount": row.get("citationcount"),
        "influentialCitationCount": row.get("influentialcitationcount"),
        "isOpenAccess": bool(row.get("isopenaccess")),
        # The dumps carry no OA pdf URL — served as null; pdfclaw's own
        # resolution chain (unpaywall/openalex/...) is unaffected.
        "openAccessPdf": None,
        "fieldsOfStudy": None,
        "s2FieldsOfStudy": s2fos or None,
        "publicationTypes": row.get("publicationtypes"),
        "publicationDate": row.get("publicationdate"),
        "journal": row.get("journal"),
        "authors": authors,
    }
    return corpusid, sha, rec


def render_author(row: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    """authors-dataset row -> (authorid, api_record) or None."""
    aid = row.get("authorid")
    try:
        aid_int = int(aid)
    except (TypeError, ValueError):
        return None
    rec = {
        "authorId": str(aid),
        "name": row.get("name") or "",
        "aliases": row.get("aliases"),
        "affiliations": row.get("affiliations") or [],
        "homepage": row.get("homepage"),
        "paperCount": row.get("papercount"),
        "citationCount": row.get("citationcount"),
        "hIndex": row.get("hindex"),
        "externalIds": row.get("externalids"),
        "url": row.get("url"),
    }
    return aid_int, rec
