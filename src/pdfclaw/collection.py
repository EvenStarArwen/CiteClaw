"""Load CiteClaw ``literature_collection.json`` + harvest DOIs from ``cache.db``.

The accepted-papers JSON that CiteClaw writes (``literature_collection.json``
or ``literature_collection.exp2.json``) records ``paper_id``, ``title``,
``year``, ``venue`` and a few other fields per paper, but **does not store
the DOI**. The DOIs live in the sibling ``cache.db`` file, in the
``paper_metadata`` table — except they're often inside the ``openAccessPdf``
disclaimer string rather than a clean ``externalIds.DOI`` field, because
S2 elides metadata for closed-access papers.

This module reconciles the two:

  1. Read the collection JSON for the canonical list of accepted papers
     and their human-readable titles.
  2. Look each paper up in ``cache.db`` and, for every entry, try BOTH:
       (a) ``data['externalIds']['DOI']`` — the clean path
       (b) regex extraction from ``data['openAccessPdf']['disclaimer']`` —
           the fallback that recovers ~80% of closed-access entries

We deliberately do NOT hit the network here. If a paper has no DOI in the
local cache, it's reported as such and skipped — the user can re-run
CiteClaw to refresh the cache, or hit the S2 batch endpoint manually.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Pulls a DOI out of strings like "Paper or abstract available at
# https://api.unpaywall.org/v2/10.1038/s41576-024-00786-y?email=...
# or https://doi.org/10.1038/s41576-024-00786-y, which is subject ..."
DOI_RE = re.compile(r"(?:doi\.org|/v2)/(10\.\d{4,9}/[^\s\"<>?]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Paper:
    """One accepted paper, ready to be dispatched to a publisher recipe."""

    paper_id: str       # S2 corpus ID hash, e.g. "dc32a984..."
    title: str
    year: int | None
    venue: str | None
    doi: str | None     # None if we couldn't recover it
    doi_source: str     # "external_ids" | "disclaimer" | "missing"
    arxiv_id: str | None = None   # bare arXiv ID e.g. "2007.06225"
    pmcid: str | None = None      # e.g. "PMC8285895"


def load_collection(json_path: Path) -> list[dict]:
    """Read CiteClaw's literature_collection JSON, return raw paper dicts."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "papers" in data:
        return list(data["papers"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(
        f"{json_path}: expected list of papers or dict with 'papers' key, "
        f"got {type(data).__name__}"
    )


def harvest_dois(paper_ids: list[str], cache_db_path: Path) -> dict[str, tuple[str, str]]:
    """For every paper id, return (doi, source) if we can find one.

    ``source`` is ``"external_ids"`` for the clean S2 path or
    ``"disclaimer"`` for the regex-extracted fallback. Papers with no
    cache entry or no extractable DOI are simply absent from the dict.
    """
    if not paper_ids:
        return {}

    out: dict[str, tuple[str, str]] = {}
    conn = sqlite3.connect(str(cache_db_path))
    try:
        cur = conn.cursor()
        # SQLite has a limit on the number of variables per statement
        # (default 999); chunk the IN clause defensively.
        chunk_size = 500
        for start in range(0, len(paper_ids), chunk_size):
            chunk = paper_ids[start : start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cur.execute(
                f"SELECT paper_id, data FROM paper_metadata WHERE paper_id IN ({placeholders})",
                chunk,
            )
            for pid, raw in cur.fetchall():
                try:
                    blob = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                ext = blob.get("externalIds") or {}
                doi = ext.get("DOI") or ext.get("doi")
                if doi:
                    out[pid] = (doi.strip(), "external_ids")
                    continue
                disclaimer = (blob.get("openAccessPdf") or {}).get("disclaimer") or ""
                m = DOI_RE.search(disclaimer)
                if m:
                    # Strip any trailing punctuation the regex picked up
                    doi = m.group(1).rstrip(".,;)")
                    out[pid] = (doi, "disclaimer")
    finally:
        conn.close()
    return out


def reconcile(collection_papers: list[dict], doi_map: dict[str, tuple[str, str]]) -> list[Paper]:
    """Merge the collection JSON with the harvested DOIs into Paper objects."""
    out: list[Paper] = []
    for p in collection_papers:
        pid = p.get("paper_id") or p.get("paperId")
        if not pid:
            continue
        doi, src = (None, "missing")
        if pid in doi_map:
            doi, src = doi_map[pid]
        out.append(
            Paper(
                paper_id=pid,
                title=p.get("title") or "",
                year=p.get("year"),
                venue=p.get("venue"),
                doi=doi,
                doi_source=src,
            )
        )
    return out


def load_papers(checkpoint_dir: Path, *, json_name: str | None = None) -> list[Paper]:
    """High-level loader: find the collection JSON + cache.db in a dir."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No such directory: {checkpoint_dir}")

    # Find the JSON file. Prefer .expN if present (latest expansion).
    if json_name:
        json_path = checkpoint_dir / json_name
    else:
        candidates = sorted(checkpoint_dir.glob("literature_collection*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No literature_collection*.json in {checkpoint_dir}"
            )
        # Sort: bare name first, then exp1, exp2... — pick the latest
        def sort_key(p: Path) -> tuple[int, str]:
            stem = p.stem  # e.g. "literature_collection.exp2"
            m = re.search(r"\.exp(\d+)$", stem)
            return (int(m.group(1)) if m else 0, p.name)
        json_path = sorted(candidates, key=sort_key)[-1]

    cache_db = checkpoint_dir / "cache.db"
    if not cache_db.exists():
        raise FileNotFoundError(f"No cache.db in {checkpoint_dir}")

    raw_papers = load_collection(json_path)
    paper_ids = [p.get("paper_id") or p.get("paperId") for p in raw_papers]
    paper_ids = [pid for pid in paper_ids if pid]
    doi_map = harvest_dois(paper_ids, cache_db)
    return reconcile(raw_papers, doi_map)
