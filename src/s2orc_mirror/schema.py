"""Shared constants + pure helpers the mapper, reducer, store, and server
must agree on, so a mismatch is impossible by construction.

One dataset (``s2orc``), keyed by integer ``corpusid``. Each dump record
looks like::

    {"corpusid": int,
     "externalids": {"doi":.., "arxiv":.., "pubmed":.., "pubmedcentral":..,
                     "mag":.., "acl":.., "dblp":..},          # lowercase keys
     "content": {"source": {"oainfo": {"license":.., "status":..,
                                        "openaccessurl":..}},
                 "text": "<full body>",
                 "annotations": {"paragraph": "<json-string span list>", ...}}}

``content.annotations`` values are JSON-encoded STRINGS of ``[{start,end,
attributes?}]`` char spans into ``content.text`` — we store them verbatim.
"""

from __future__ import annotations

import hashlib
import re

# ---- shard layout ----------------------------------------------------------

TEXT_SHARDS = 64   # fulltext_XX.db, partitioned by corpusid % TEXT_SHARDS
ID_SHARDS = 32     # ids_XX.db, partitioned by a stable hash of the key text


def text_shard(corpusid: int) -> int:
    return corpusid % TEXT_SHARDS


def id_shard(key: str) -> int:
    """Stable text-hash shard for idmap keys (sha1 -> int, NOT Python hash())."""
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") % ID_SHARDS


# ---- external id indexing --------------------------------------------------

# S2ORC externalids keys vary by record schema: the modern `content` files
# use lowercase (doi/arxiv/pubmed/pubmedcentral); the legacy `body` files nest
# them under openaccessinfo with TitleCase (DOI/ArXiv/Medline/PubMedCentral).
# Match case-insensitively. We index only the kinds CiteClaw resolves papers
# by; mag / acl / dblp / medrxiv are deliberately not indexed.
_XID_PREFIX = {
    "doi": "doi",
    "arxiv": "arxiv",
    "pubmed": "pmid",
    "medline": "pmid",
    "pubmedcentral": "pmcid",
}


def idmap_keys(externalids: dict | None) -> list[str]:
    """The ``<prefix>:<value>`` idmap keys a record contributes."""
    out: list[str] = []
    for key, val in (externalids or {}).items():
        if not val:
            continue
        prefix = _XID_PREFIX.get(str(key).strip().lower())
        if prefix:
            out.append(f"{prefix}:{str(val).strip().lower()}")
    return out


# ---- paper id parsing (API-style id -> local lookup) -----------------------

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LOOKUP_PREFIX = {
    "doi": "doi",
    "arxiv": "arxiv",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "pubmed": "pmid",
    "pubmedcentral": "pmcid",
}


def parse_paper_id(raw: str) -> tuple[str, str | int] | None:
    """Normalize an API-style paper id into a local lookup.

    Returns ``("corpus", int)`` for ``CorpusId:123``; ``("key", "<text>")``
    for ``DOI:`` / ``ARXIV:`` / ``PMID:`` / ``PMCID:`` (and bare 40-hex sha,
    which S2ORC has no index for -> resolves to a miss); ``None`` otherwise.
    """
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if _SHA_RE.match(low):
        return ("key", low)  # no sha idmap in S2ORC; deliberately a miss
    prefix, sep, rest = s.partition(":")
    if not sep or not rest:
        return None
    p = prefix.strip().lower()
    rest = rest.strip()
    if p == "corpusid":
        try:
            return ("corpus", int(rest))
        except ValueError:
            return None
    mapped = _LOOKUP_PREFIX.get(p)
    if mapped:
        key = f"{mapped}:{rest.lower()}"
        # DataCite arXiv DOIs (10.48550/arXiv.<id>) aren't in the S2ORC
        # externalids DOI column — normalize them to the arXiv id, which is.
        if mapped == "doi" and key.startswith("doi:10.48550/arxiv."):
            key = "arxiv:" + key[len("doi:10.48550/arxiv."):]
        return ("key", key)
    return None


# ---- record extraction -----------------------------------------------------

def slim_record(row: dict) -> dict | None:
    """Turn one raw S2ORC dump record into the slim dict we store.

    Handles BOTH record schemas the ``s2orc`` dataset ships across its files:

    * modern ``content`` files -- full text at ``content.text``, OA info at
      ``content.source.oainfo`` (``openaccessurl``), externalids top-level
      and lowercase;
    * legacy ``body`` files -- full text at ``body.text``, OA info at
      top-level ``openaccessinfo`` (``url`` + a nested TitleCase
      ``externalids``), and no ``content`` key at all.

    Returns ``None`` when the record has no ``corpusid`` or no body text.
    """
    cid = row.get("corpusid")
    if not isinstance(cid, int):
        return None

    content = row.get("content")
    if isinstance(content, dict):  # modern schema
        text = content.get("text")
        if not text or not isinstance(text, str):
            return None
        oa = (content.get("source") or {}).get("oainfo") or {}
        return {
            "corpusid": cid,
            "text": text,
            "annotations": content.get("annotations") or {},
            "license": oa.get("license"),
            "status": oa.get("status"),
            "oaurl": oa.get("openaccessurl"),
            "externalids": row.get("externalids") or {},
        }

    body = row.get("body")  # legacy schema
    text = body.get("text") if isinstance(body, dict) else body
    if not text or not isinstance(text, str):
        return None
    oai = row.get("openaccessinfo") or {}
    return {
        "corpusid": cid,
        "text": text,
        "annotations": (body.get("annotations") if isinstance(body, dict) else None) or {},
        "license": oai.get("license"),
        "status": oai.get("status"),
        "oaurl": oai.get("url"),
        "externalids": oai.get("externalids") or {},
    }
