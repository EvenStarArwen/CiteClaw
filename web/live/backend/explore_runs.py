"""Discover finished runs on disk and shape them for the Exploration page.

A "run" here is any directory under ``runs/`` (one or two levels deep, so
both CLI runs like ``runs/data_bio/`` and WebUI runs like
``runs/webui/<hex>/`` are found) that contains either:

  * ``literature_collection.json`` written by Finalize, or
  * ``citation_network.gexf`` — a Gephi export (e.g. from CitNet); drop one
    into ``runs/<name>/`` and it becomes explorable as a demo dataset.

The exploration payload mirrors the live-store shapes:

  * paper -> {id, title, authors, year, venue, cites, seed, depth, source,
              score, abstract}
  * edge  -> {source, target}   (paper-id pairs)

For JSON runs, edges are derived from each paper's ``references`` +
``supporting_papers`` intersected with the collection — the same rule
``snapshots.build_graph`` uses for the live graph. GEXF files carry their
own edge list, which is used directly.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .snapshots import _authors_str, _score

RUNS_ROOT = Path("runs")
GEXF_NAME = "citation_network.gexf"

_MAX_PAPERS = 1500  # keep FA2 + the list responsive on big runs

# path-str -> (mtime, meta dict); avoids re-parsing unchanged files on list
_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# path-str -> (mtime, payload) for parsed GEXFs (parsing is the pricey part)
_gexf_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _collection_files() -> list[Path]:
    """One data file per run dir; literature_collection.json wins over GEXF."""
    if not RUNS_ROOT.is_dir():
        return []
    by_dir: dict[Path, Path] = {}
    for jf in list(RUNS_ROOT.glob("*/literature_collection.json")) \
            + list(RUNS_ROOT.glob("*/*/literature_collection.json")):
        by_dir[jf.parent] = jf
    for gx in list(RUNS_ROOT.glob(f"*/{GEXF_NAME}")) \
            + list(RUNS_ROOT.glob(f"*/*/{GEXF_NAME}")):
        by_dir.setdefault(gx.parent, gx)
    return list(by_dir.values())


# ---------------------------------------------------------------- GEXF ----

def _tag(el) -> str:
    """Local tag name, namespace-agnostic (gexf 1.2/1.3 vary)."""
    return el.tag.rsplit("}", 1)[-1]


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _parse_gexf(path: Path) -> dict[str, Any]:
    """Parse a GEXF file into the exploration payload (uncapped)."""
    root = ET.parse(path).getroot()

    # attribute-id -> declared title, for <attvalue for="..."> resolution
    attr_titles: dict[str, str] = {}
    for attrs in root.iter():
        if _tag(attrs) != "attributes" or attrs.get("class") != "node":
            continue
        for a in attrs:
            if _tag(a) == "attribute" and a.get("id") is not None:
                attr_titles[a.get("id")] = (a.get("title") or "").strip().lower()

    papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in root.iter():
        if _tag(node) != "node":
            continue
        vals: dict[str, str] = {}
        for child in node:
            if _tag(child) != "attvalues":
                continue
            for av in child:
                if _tag(av) != "attvalue":
                    continue
                key = attr_titles.get(av.get("for") or "", av.get("for") or "")
                vals[key] = av.get("value") or ""
        pid = vals.get("paper_id") or node.get("id") or ""
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        cites = _int(vals.get("citation_count") or vals.get("cites"))
        papers.append({
            "id": pid,
            "title": vals.get("title") or node.get("label") or pid,
            "authors": vals.get("authors") or "",
            "year": _int(vals.get("year")),
            "venue": vals.get("venue") or "",
            "cites": cites,
            "seed": _truthy(vals.get("seed") or vals.get("is_seed") or ""),
            "depth": _int(vals.get("depth")),
            "source": vals.get("source") or "",
            "score": _score(cites),
            "abstract": vals.get("abstract") or "",
        })

    # GEXF node ids may differ from paper_id; map both spellings to paper id
    node_to_pid: dict[str, str] = {}
    for node in root.iter():
        if _tag(node) != "node":
            continue
        nid = node.get("id") or ""
        pid = nid
        for child in node:
            if _tag(child) == "attvalues":
                for av in child:
                    if _tag(av) == "attvalue" \
                            and attr_titles.get(av.get("for") or "") == "paper_id" \
                            and av.get("value"):
                        pid = av.get("value")
        if nid:
            node_to_pid[nid] = pid

    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    for edge in root.iter():
        if _tag(edge) != "edge":
            continue
        a = node_to_pid.get(edge.get("source") or "", edge.get("source") or "")
        b = node_to_pid.get(edge.get("target") or "", edge.get("target") or "")
        if not a or not b or a == b or a not in seen_ids or b not in seen_ids:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        try:
            w = float(edge.get("weight") or 0.0)
        except ValueError:
            w = 0.0
        edges.append({"source": a, "target": b, "weight": w})

    return {"papers": papers, "edges": edges}


def _gexf_payload(path: Path) -> dict[str, Any]:
    mtime = path.stat().st_mtime
    key = str(path)
    cached = _gexf_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    payload = _parse_gexf(path)
    _gexf_cache[key] = (mtime, payload)
    return payload


def _run_meta(jf: Path) -> dict[str, Any] | None:
    try:
        mtime = jf.stat().st_mtime
    except OSError:
        return None
    key = str(jf)
    cached = _meta_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    # jf arrives relative from the list scan but absolute from load_explore_run
    rel = jf.parent.resolve().relative_to(RUNS_ROOT.resolve()).as_posix()
    try:
        if jf.name == GEXF_NAME:
            payload = _gexf_payload(jf)
            papers, seeds = len(payload["papers"]), sum(1 for p in payload["papers"] if p["seed"])
        else:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary") or {}
            src_dist = summary.get("source_distribution") or {}
            papers = int(summary.get("total_accepted") or len(data.get("papers") or []))
            seeds = int(src_dist.get("seed") or 0)
        meta = {
            "path": rel,
            "label": rel,
            "papers": papers,
            "seeds": seeds,
            "mtime": mtime,
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
        }
    except (OSError, ValueError, ET.ParseError):
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


def _load_gexf_run(gx: Path) -> dict[str, Any]:
    payload = _gexf_payload(gx)
    raw = sorted(payload["papers"],
                 key=lambda p: (p["seed"], p["cites"]), reverse=True)
    dropped = max(0, len(raw) - _MAX_PAPERS)
    papers = raw[:_MAX_PAPERS]
    kept = {p["id"] for p in papers}
    edges = [e for e in payload["edges"] if e["source"] in kept and e["target"] in kept]
    meta = _run_meta(gx) or {}
    return {"papers": papers, "edges": edges, "meta": {**meta, "dropped": dropped}}


def load_explore_run(rel_path: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(rel_path)
    jf = run_dir / "literature_collection.json"
    if not jf.is_file():
        gx = run_dir / GEXF_NAME
        if gx.is_file():
            return _load_gexf_run(gx)
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
