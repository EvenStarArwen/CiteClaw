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
COLLAB_NAME = "collaboration_network.graphml"

_MAX_PAPERS = 1500  # keep FA2 + the list responsive on big runs
_MAX_AUTHORS = 1500
_MAX_AUTHORS_PER_PAPER = 25  # consortium papers would explode the pair count

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
    return _parse_gexf_root(ET.parse(path).getroot())


def _parse_gexf_root(root) -> dict[str, Any]:
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


# ------------------------------------------------- collaboration network ----
#
# The Explore page can switch to the author co-authorship view. Preferred
# source is Finalize's ``collaboration_network.graphml`` (igraph export with
# h_index / affiliation / edge strength); runs that only have a
# ``literature_collection.json`` get the same graph derived from the papers'
# full author lists (strength = Σ 1/N per shared paper — the Finalize rule).
# GEXF drop-ins carry no author lists, so they have no collaboration view.
#
# Author nodes reuse the paper payload shape (id/title/venue/cites/...) so the
# front-end graph, list and detail panels work unchanged; author-specific
# extras ride along as ``hIndex`` / ``nPapers`` / ``authorId`` / ``citesAll``.
#
# Semantics chosen to match the CLI's author_graph.py exports:
#   * ``cites``    — intra_network_citation: total citations of the author's
#                    papers INSIDE the collection (drives node size)
#   * ``year``     — year_entered: the year the author's first paper appears
#                    in the collection (drives node colour)
#   * ``citesAll`` — the author's S2-wide citation count (detail panel only)

def _author_row(*, aid: str, name: str, affiliation: str, cites: int,
                h_index: int | None, n_papers: int, author_id: str,
                first_year: int = 0, cites_all: int | None = None) -> dict[str, Any]:
    return {
        "id": aid,
        "title": name,
        "authors": "",
        "year": first_year,
        "venue": affiliation,
        "cites": cites,
        "seed": False,
        "depth": 0,
        "source": "",
        "score": 0.0,
        "abstract": "",
        "hIndex": h_index,
        "nPapers": n_papers,
        "authorId": author_id,
        "citesAll": cites_all,
    }


def _parse_graphml_collab_root(root) -> dict[str, Any]:
    key_names: dict[str, str] = {}  # key id -> attr.name
    for el in root.iter():
        if _tag(el) == "key" and el.get("id"):
            key_names[el.get("id")] = (el.get("attr.name") or "").strip().lower()

    authors: list[dict[str, Any]] = []
    node_to_aid: dict[str, str] = {}
    seen: set[str] = set()
    for el in root.iter():
        if _tag(el) != "node":
            continue
        nid = el.get("id") or ""
        vals: dict[str, str] = {}
        for d in el:
            if _tag(d) == "data":
                vals[key_names.get(d.get("key") or "", d.get("key") or "")] = d.text or ""
        name = (vals.get("name") or "").strip() or nid
        aid = (vals.get("author_id") or "").strip() or name
        if not aid or aid in seen:
            continue
        seen.add(aid)
        node_to_aid[nid] = aid
        # in-network citations when the export has them (author_graph.py always
        # writes the column); pre-intra exports fall back to the S2-wide count
        cites = (_int(vals.get("intra_network_citation"))
                 if "intra_network_citation" in vals
                 else _int(vals.get("total_citation")))
        authors.append(_author_row(
            aid=aid, name=name,
            affiliation=(vals.get("affiliation") or "").strip(),
            cites=cites,
            h_index=_int(vals.get("h_index")),
            n_papers=_int(vals.get("paper_count_in_community"))
                     or _int(vals.get("paper_count_s2")),
            author_id=(vals.get("author_id") or "").strip(),
            first_year=_int(vals.get("year_entered")),
            cites_all=_int(vals.get("total_citation"))
                      if "total_citation" in vals else None,
        ))

    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    for el in root.iter():
        if _tag(el) != "edge":
            continue
        a = node_to_aid.get(el.get("source") or "")
        b = node_to_aid.get(el.get("target") or "")
        if not a or not b or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        w = 0.0
        for d in el:
            if _tag(d) == "data" and key_names.get(d.get("key") or "") in ("strength", "weight"):
                try:
                    w = float(d.text or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                break
        edges.append({"source": a, "target": b, "weight": w})
    return {"papers": authors, "edges": edges}


def _parse_graphml_collab(path: Path) -> dict[str, Any]:
    return _parse_graphml_collab_root(ET.parse(path).getroot())


def _derive_collab(raw_papers: list[dict[str, Any]]) -> dict[str, Any]:
    n_papers: dict[str, int] = {}
    cites: dict[str, int] = {}
    first_year: dict[str, int] = {}
    strength: dict[tuple[str, str], float] = {}
    for p in raw_papers:
        names = [a.get("name", "").strip() for a in (p.get("authors") or [])
                 if isinstance(a, dict) and a.get("name", "").strip()]
        names = list(dict.fromkeys(names))[:_MAX_AUTHORS_PER_PAPER]
        if not names:
            continue
        n = len(names)
        c = int(p.get("citation_count") or 0)
        yr = int(p.get("year") or 0)
        for nm in names:
            n_papers[nm] = n_papers.get(nm, 0) + 1
            cites[nm] = cites.get(nm, 0) + c
            if yr and (nm not in first_year or yr < first_year[nm]):
                first_year[nm] = yr
        for i in range(n):
            for j in range(i + 1, n):
                key = (names[i], names[j]) if names[i] < names[j] else (names[j], names[i])
                strength[key] = strength.get(key, 0.0) + 1.0 / n
    authors = [
        _author_row(aid=nm, name=nm, affiliation="", cites=cites.get(nm, 0),
                    h_index=None, n_papers=k, author_id="",
                    first_year=first_year.get(nm, 0))
        for nm, k in n_papers.items()
    ]
    edges = [{"source": a, "target": b, "weight": round(w, 4)}
             for (a, b), w in strength.items()]
    return {"papers": authors, "edges": edges}


def load_explore_collab(rel_path: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(rel_path)
    gm = run_dir / COLLAB_NAME
    if gm.is_file():
        payload = _parse_graphml_collab(gm)
        derived = False
    else:
        jf = run_dir / "literature_collection.json"
        if not jf.is_file():
            if (run_dir / GEXF_NAME).is_file():
                raise LookupError(
                    "This dataset has no author data — GEXF exports carry no author lists.")
            raise FileNotFoundError(rel_path)
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        payload = _derive_collab(list(data.get("papers") or []))
        derived = True

    raw = sorted(payload["papers"],
                 key=lambda a: (a["nPapers"], a["cites"]), reverse=True)
    dropped = max(0, len(raw) - _MAX_AUTHORS)
    authors = raw[:_MAX_AUTHORS]
    kept = {a["id"] for a in authors}
    edges = [e for e in payload["edges"] if e["source"] in kept and e["target"] in kept]
    return {
        "papers": authors,
        "edges": edges,
        "meta": {"path": rel_path, "kind": "author", "derived": derived,
                 "dropped": dropped},
    }


# ----------------------------------------------------- uploaded graph files ----
#
# The Explore page can open any local GraphML/GEXF via the browser file
# picker. The bytes are parsed in memory (nothing is written to disk) after a
# compatibility sniff: the XML root decides the format, the declared node
# attributes decide whether it's a CiteClaw citation network or a
# collaboration network. Anything else raises a ValueError whose message
# explains exactly what was found and what was expected.

# attribute names that identify each family (case-insensitive)
_COLLAB_KEYS = {"author_id", "h_index", "paper_count_in_community",
                "intra_network_citation", "strength", "year_entered"}
_CITE_KEYS = {"paper_id", "citation_count", "title", "abstract",
              "ref_similarity", "cit_similarity", "semantic_similarity"}


def _parse_graphml_cite(root) -> dict[str, Any]:
    """Citation-network GraphML (Finalize's citation_network.graphml shape)."""
    key_names: dict[str, str] = {}
    for el in root.iter():
        if _tag(el) == "key" and el.get("id"):
            key_names[el.get("id")] = (el.get("attr.name") or "").strip().lower()

    papers: list[dict[str, Any]] = []
    node_to_pid: dict[str, str] = {}
    seen: set[str] = set()
    for el in root.iter():
        if _tag(el) != "node":
            continue
        nid = el.get("id") or ""
        vals: dict[str, str] = {}
        for d in el:
            if _tag(d) == "data":
                vals[key_names.get(d.get("key") or "", d.get("key") or "")] = d.text or ""
        pid = (vals.get("paper_id") or "").strip() or nid
        if not pid or pid in seen:
            continue
        seen.add(pid)
        node_to_pid[nid] = pid
        cites = _int(vals.get("citation_count") or vals.get("cites"))
        papers.append({
            "id": pid,
            "title": (vals.get("title") or "").strip() or pid,
            "authors": vals.get("authors") or "",
            "year": _int(vals.get("year")),
            "venue": vals.get("venue") or "",
            "cites": cites,
            "seed": _truthy(vals.get("seed") or "")
                    or (vals.get("source") or "").strip().lower() == "seed",
            "depth": _int(vals.get("depth")),
            "source": vals.get("source") or "",
            "score": _score(cites),
            "abstract": vals.get("abstract") or "",
        })

    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    for el in root.iter():
        if _tag(el) != "edge":
            continue
        a = node_to_pid.get(el.get("source") or "")
        b = node_to_pid.get(el.get("target") or "")
        if not a or not b or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        w = 0.0
        for d in el:
            if _tag(d) == "data" and key_names.get(d.get("key") or "") == "weight":
                try:
                    w = float(d.text or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                break
        edges.append({"source": a, "target": b, "weight": w})
    return {"papers": papers, "edges": edges}


def _declared_attr_names(root) -> set[str]:
    """GraphML attribute names: <key attr.name> declarations, falling back to
    raw data-key ids sampled from the first few nodes (some exporters skip
    the declarations)."""
    names = {(el.get("attr.name") or "").strip().lower()
             for el in root.iter() if _tag(el) == "key"}
    names.discard("")
    if not names:
        sampled = 0
        for el in root.iter():
            if _tag(el) != "node":
                continue
            for d in el:
                if _tag(d) == "data" and d.get("key"):
                    names.add(d.get("key").strip().lower())
            sampled += 1
            if sampled >= 5:
                break
    return names


def load_explore_upload(name: str, data: bytes) -> dict[str, Any]:
    """Parse an uploaded graph file into an exploration payload.

    Returns the run-shaped payload with ``meta.kind`` = "paper" | "author".
    Raises ValueError with a human-readable reason when the file is not a
    CiteClaw-compatible citation/collaboration network.
    """
    label = name or "uploaded file"
    if not data or not data.strip():
        raise ValueError(f"'{label}' is empty.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(
            f"'{label}' is not well-formed XML ({e}). CiteClaw networks are "
            "GraphML (.graphml) or GEXF (.gexf) files — e.g. "
            "citation_network.graphml or collaboration_network.graphml from a run directory.")

    fmt = _tag(root).lower()
    if fmt == "graphml":
        attrs = _declared_attr_names(root)
        if attrs & _COLLAB_KEYS:
            payload, kind = _parse_graphml_collab_root(root), "author"
        elif attrs & _CITE_KEYS:
            payload, kind = _parse_graphml_cite(root), "paper"
        else:
            found = ", ".join(sorted(attrs)) or "none"
            raise ValueError(
                f"'{label}' is GraphML, but its node attributes ({found}) match neither "
                "a CiteClaw citation network (expects e.g. paper_id / title / "
                "citation_count) nor a collaboration network (expects e.g. author_id / "
                "h_index / strength). Was this exported by CiteClaw?")
    elif fmt == "gexf":
        payload, kind = _parse_gexf_root(root), "paper"
    else:
        raise ValueError(
            f"'{label}' is XML but its root element is <{_tag(root)}>, not <graphml> or "
            "<gexf> — CiteClaw can only open GraphML / GEXF network files.")

    if not payload["papers"]:
        raise ValueError(
            f"'{label}' parsed as {fmt.upper()} but contains no usable nodes "
            "(no <node> elements, or none with an id).")

    if kind == "author":
        raw = sorted(payload["papers"],
                     key=lambda a: (a["nPapers"], a["cites"]), reverse=True)
        cap = _MAX_AUTHORS
    else:
        raw = sorted(payload["papers"],
                     key=lambda p: (p["seed"], p["cites"]), reverse=True)
        cap = _MAX_PAPERS
    dropped = max(0, len(raw) - cap)
    rows = raw[:cap]
    kept = {r["id"] for r in rows}
    edges = [e for e in payload["edges"] if e["source"] in kept and e["target"] in kept]
    return {
        "papers": rows,
        "edges": edges,
        "meta": {"kind": kind, "name": label, "format": fmt,
                 "papers": len(rows), "dropped": dropped, "source": "upload"},
    }
