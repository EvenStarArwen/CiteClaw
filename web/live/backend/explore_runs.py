"""Discover finished runs on disk and shape them for the Exploration page.

A "run" here is any directory under ``runs/`` (one or two levels deep, so
both CLI runs like ``runs/data_bio/`` and WebUI runs like
``runs/webui/<hex>/`` are found) that contains a ``literature_collection.json``
written by Finalize. The exploration payload mirrors the live-store shapes:

  * paper -> {id, title, authors, year, venue, cites, seed, depth, source,
              score, abstract}
  * edge  -> {source, target}   (paper-id pairs, references ∩ collection)

Edges are derived from each paper's ``references`` + ``supporting_papers``
intersected with the collection — the same rule ``snapshots.build_graph``
uses for the live graph, so live and on-disk networks look alike.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .snapshots import _authors_str, _score

RUNS_ROOT = Path("runs")

_MAX_PAPERS = 1500  # keep FA2 + the list responsive on big runs

# path-str -> (mtime, meta dict); avoids re-parsing unchanged JSONs on list
_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _collection_files() -> list[Path]:
    if not RUNS_ROOT.is_dir():
        return []
    files = list(RUNS_ROOT.glob("*/literature_collection.json"))
    files += RUNS_ROOT.glob("*/*/literature_collection.json")
    return files


def _run_meta(jf: Path) -> dict[str, Any] | None:
    try:
        mtime = jf.stat().st_mtime
    except OSError:
        return None
    key = str(jf)
    cached = _meta_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary") or {}
        src_dist = summary.get("source_distribution") or {}
        meta = {
            "path": jf.parent.relative_to(RUNS_ROOT).as_posix(),
            "label": jf.parent.relative_to(RUNS_ROOT).as_posix(),
            "papers": int(summary.get("total_accepted") or len(data.get("papers") or [])),
            "seeds": int(src_dist.get("seed") or 0),
            "mtime": mtime,
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
        }
    except (OSError, ValueError):
        return None
    _meta_cache[key] = (mtime, meta)
    return meta


def list_explore_runs() -> list[dict[str, Any]]:
    """All explorable runs, newest first."""
    metas = [m for jf in _collection_files() if (m := _run_meta(jf)) is not None]
    metas.sort(key=lambda m: m["mtime"], reverse=True)
    return metas


def _resolve_run_dir(rel_path: str) -> Path:
    """Map a client-supplied relative path back to a real run dir, refusing
    anything that escapes ``runs/``."""
    root = RUNS_ROOT.resolve()
    candidate = (RUNS_ROOT / rel_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("invalid run path")
    return candidate


def load_explore_run(rel_path: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(rel_path)
    jf = run_dir / "literature_collection.json"
    if not jf.is_file():
        raise FileNotFoundError(rel_path)
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)

    raw = list(data.get("papers") or [])
    # prefer seeds + most-cited when capping (same rule as the live graph)
    raw.sort(key=lambda p: (p.get("source") == "seed",
                            int(p.get("citation_count") or 0)), reverse=True)
    dropped = max(0, len(raw) - _MAX_PAPERS)
    raw = raw[:_MAX_PAPERS]

    papers = []
    kept: set[str] = set()
    for p in raw:
        pid = p.get("paper_id")
        if not pid or pid in kept:
            continue
        kept.add(pid)
        cites = int(p.get("citation_count") or 0)
        papers.append({
            "id": pid,
            "title": p.get("title") or pid,
            "authors": _authors_str(p.get("authors")),
            "year": int(p.get("year") or 0),
            "venue": p.get("venue") or "",
            "cites": cites,
            "seed": p.get("source") == "seed",
            "depth": int(p.get("depth") or 0),
            "source": p.get("source") or "",
            "score": _score(cites),
            "abstract": p.get("abstract") or "",
        })

    seen: set[tuple[str, str]] = set()
    edges = []
    for p in raw:
        a = p.get("paper_id")
        neighbors = list(p.get("references") or []) + list(p.get("supporting_papers") or [])
        for b in neighbors:
            if b not in kept or b == a:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": a, "target": b})

    meta = _run_meta(jf) or {}
    return {
        "papers": papers,
        "edges": edges,
        "meta": {**meta, "dropped": dropped},
    }
