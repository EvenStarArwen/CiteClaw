"""Shared constants + pure helpers: shard layout, id parsing, edge flags.

Everything the mapper, reducer, store, and server must agree on lives
here so a mismatch is impossible by construction.
"""

from __future__ import annotations

import hashlib
import re

# ---- shard layout ----------------------------------------------------------

PAPER_SHARDS = 64    # papers_XX.db + graph_XX.db, partitioned by corpusid
ID_SHARDS = 32       # ids_XX.db, partitioned by fnv-style hash of the key text
AUTHOR_SHARDS = 16   # authors_XX.db, partitioned by int(authorid)

# Binary struct of one partitioned edge row emitted by the mapper:
#   <int64 key> <int64 other> <uint8 flags>   (little-endian, packed)
EDGE_ROW = "<qqB"
EDGE_ROW_SIZE = 17
# Struct of one adjacency entry inside a graph blob:
#   <int64 other> <uint8 flags>
ADJ_ROW = "<qB"
ADJ_ROW_SIZE = 9
# Author->paper pair emitted by the mapper: <int64 authorid> <int64 corpusid>
PAIR_ROW = "<qq"
PAIR_ROW_SIZE = 16

# ---- edge flag bitmask -----------------------------------------------------

FLAG_INFLUENTIAL = 0x01
_INTENT_BITS = {
    "methodology": 0x02,
    "background": 0x04,
    "result": 0x08,
}
_BITS_INTENT = {v: k for k, v in _INTENT_BITS.items()}


def pack_flags(is_influential: bool, intents: list | None) -> int:
    flags = FLAG_INFLUENTIAL if is_influential else 0
    for intent in intents or ():
        flags |= _INTENT_BITS.get(str(intent).lower(), 0)
    return flags


def unpack_flags(flags: int) -> tuple[bool, list[str]]:
    intents = [name for bit, name in _BITS_INTENT.items() if flags & bit]
    return bool(flags & FLAG_INFLUENTIAL), intents


# ---- shard routing ---------------------------------------------------------

def paper_shard(corpusid: int) -> int:
    return corpusid % PAPER_SHARDS


def author_shard(authorid: int) -> int:
    return authorid % AUTHOR_SHARDS


def id_shard(key: str) -> int:
    """Stable text-hash shard for idmap keys (sha1 → int, NOT Python hash())."""
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") % ID_SHARDS


# ---- paper id parsing ------------------------------------------------------

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_URL_SHA_RE = re.compile(r"/paper/([0-9a-f]{40})")

# external-id kinds we index locally (dataset key -> idmap prefix).
# MAG / DBLP / URL are deliberately NOT indexed (CiteClaw never looks
# papers up by them; ~150M MAG rows saved) — those lookups fall through
# to the upstream proxy.
XID_PREFIX = {
    "DOI": "doi",
    "ArXiv": "arxiv",
    "PubMed": "pmid",
    "PubMedCentral": "pmcid",
    "ACL": "acl",
}
_LOOKUP_PREFIX = {
    "doi": "doi",
    "arxiv": "arxiv",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "acl": "acl",
}


def canonical_sha(url: str | None) -> str | None:
    """Extract the canonical 40-hex paperId from a papers-dataset ``url``."""
    if not url:
        return None
    m = _URL_SHA_RE.search(url)
    return m.group(1) if m else None


def parse_paper_id(raw: str) -> tuple[str, str | int] | None:
    """Normalize an API-style paper id into a local lookup.

    Returns:
      ("corpus", int)   for ``CorpusId:123``
      ("key", "<text>") for sha hex / DOI: / ARXIV: / PMID: / PMCID: / ACL:
      None              for forms we don't index (MAG:, DBLP:, URL:, junk)
    """
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if _SHA_RE.match(low):
        return ("key", low)
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
        # DataCite arXiv DOIs (10.48550/arXiv.<id>) aren't in the S2AG
        # externalids DOI column — normalize them to the arXiv id, which is.
        if mapped == "doi" and key.startswith("doi:10.48550/arxiv."):
            key = "arxiv:" + key[len("doi:10.48550/arxiv."):]
        return ("key", key)
    return None


# ---- fields param ----------------------------------------------------------

def parse_fields(fields: str | None, default: str = "paperId,title") -> dict[str, set[str] | None]:
    """Parse an S2 ``fields`` param into {top: None | {subfields}}.

    ``None`` as the value means "the whole field"; a set means "only
    these sub-keys" (e.g. ``authors.authorId,authors.name``).
    ``paperId`` is always implied, matching S2 behaviour.
    """
    out: dict[str, set[str] | None] = {}
    for part in (fields or default).split(","):
        part = part.strip()
        if not part:
            continue
        top, sep, sub = part.partition(".")
        if not sep:
            out[top] = None
        else:
            cur = out.get(top)
            if cur is None and top in out:
                continue  # whole field already requested
            if cur is None:
                out[top] = {sub}
            else:
                cur.add(sub)
    out.setdefault("paperId", None)
    return out


def project(record: dict, wants: dict[str, set[str] | None]) -> dict:
    """Field-filter a stored full record per a parsed ``fields`` spec."""
    out = {}
    for top, subs in wants.items():
        if top not in record:
            continue
        val = record[top]
        if subs and isinstance(val, list):
            val = [
                {k: item.get(k) for k in subs}
                for item in val
                if isinstance(item, dict)
            ]
        elif subs and isinstance(val, dict):
            val = {k: val.get(k) for k in subs}
        out[top] = val
    return out
