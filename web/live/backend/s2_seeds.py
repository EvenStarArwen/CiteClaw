"""Semantic Scholar seed search for the design's Seeds panel.

A thin direct call to the S2 paper-search endpoint, returning rows in the
exact ``SEED_PAPERS`` shape the front end expects
({id, title, authors, year, venue, cites}). Works keyless at a low rate;
uses ``S2_API_KEY`` when present for higher throughput.
"""

from __future__ import annotations

import os
import time

import httpx

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,year,venue,authors,citationCount,externalIds"


class S2SearchError(RuntimeError):
    """Search failed with a user-facing, actionable message."""


def _authors_str(authors: list[dict] | None) -> str:
    names = [a.get("name", "") for a in (authors or []) if a.get("name")]
    if not names:
        return ""
    return " & ".join(names) if len(names) <= 2 else f"{names[0]} et al."


def _api_key() -> str:
    return (os.environ.get("S2_API_KEY")
            or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            or os.environ.get("CITECLAW_S2_API_KEY") or "").strip()


def search_seeds(query: str, limit: int = 20) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    key = _api_key()
    headers = {"x-api-key": key} if key else {}
    params = {"query": query, "limit": max(1, min(40, int(limit))), "fields": _FIELDS}

    with httpx.Client(timeout=20.0) as client:
        r = None
        # S2 throttles hard (especially keyless / a brand-new key); one short
        # backoff-retry clears most transient 429s.
        for attempt in range(2):
            r = client.get(_S2_SEARCH_URL, params=params, headers=headers)
            if r.status_code != 429 or attempt == 1:
                break
            time.sleep(1.5)

        if r.status_code == 429:
            if key:
                raise S2SearchError(
                    "Semantic Scholar is rate-limiting this key. Wait a few "
                    "seconds and search again — a fresh key can take a minute "
                    "to warm up.")
            raise S2SearchError(
                "Semantic Scholar rate-limited the search (no API key). Add a "
                "Semantic Scholar key in Settings (gear, top-right) for reliable "
                "searches — it's free.")
        if r.status_code in (401, 403):
            raise S2SearchError(
                "Semantic Scholar rejected the API key. Check the key in "
                "Settings (gear, top-right).")
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
