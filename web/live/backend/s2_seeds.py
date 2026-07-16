"""Semantic Scholar seed search for the design's Seeds panel.

A thin direct call to the S2 paper-search endpoint, returning rows in the
exact ``SEED_PAPERS`` shape the front end expects
({id, title, authors, year, venue, cites}). Works keyless at a low rate;
uses ``S2_API_KEY`` when present for higher throughput.
"""

from __future__ import annotations

import os

import httpx

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,year,venue,authors,citationCount,externalIds"


def _authors_str(authors: list[dict] | None) -> str:
    names = [a.get("name", "") for a in (authors or []) if a.get("name")]
    if not names:
        return ""
    return " & ".join(names) if len(names) <= 2 else f"{names[0]} et al."


def search_seeds(query: str, limit: int = 20) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    key = (os.environ.get("S2_API_KEY")
           or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
           or os.environ.get("CITECLAW_S2_API_KEY") or "")
    headers = {"x-api-key": key} if key else {}
    params = {"query": query, "limit": max(1, min(40, int(limit))), "fields": _FIELDS}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(_S2_SEARCH_URL, params=params, headers=headers)
        r.raise_for_status()
        data = r.json().get("data") or []
    out = []
    for p in data:
        out.append({
            "id": p.get("paperId", ""),
            "title": p.get("title") or "",
            "authors": _authors_str(p.get("authors")),
            "year": p.get("year") or 0,
            "venue": p.get("venue") or "",
            "cites": p.get("citationCount") or 0,
            "externalIds": p.get("externalIds") or {},
        })
    return out
