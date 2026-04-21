"""Build and export an author collaboration graph from a CiteClaw collection.

The graph is **undirected**. Each node is an S2 author (keyed by
``authorId``, falling back to ``name:<author-name>`` when the ID is
missing). Each edge represents a co-authorship between two authors
in some paper of the collection.

Node attributes (set on every vertex):

* ``name`` — display name (from S2 author dict, falling back to the
  ``author_details`` lookup)
* ``author_id`` — the keying string (``"<S2 authorId>"`` or
  ``"name:<name>"``)
* ``year_entered`` — earliest paper year for this author in the
  collection (``0`` when no paper has a year)
* ``total_citation`` — S2 citationCount from author_details (0 when
  unknown)
* ``h_index`` — S2 hIndex from author_details (0 when unknown)
* ``paper_count_s2`` — S2 paperCount from author_details (0 when unknown)
* ``paper_count_in_community`` — number of this author's papers
  inside the input collection
* ``intra_network_citation`` — sum of citationCount across this
  author's collection papers
* ``affiliation`` — ``" / "``-joined affiliation strings from
  author_details

Edge attributes:

* ``strength`` — sum of ``1/N`` per shared paper, where ``N`` is the
  paper's author count (so co-authoring a 2-author paper contributes
  more than co-authoring a 50-author paper)
* ``first_year`` / ``last_year`` — earliest / latest paper year
  shared by the pair (``0`` when no shared paper has a year)
* ``duration`` — ``last_year - first_year`` (0 when years unknown)
* ``n_collaborations`` — raw count of papers shared by the pair
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.author_graph")

_FALLBACK_KEY_WARNED = False


def _author_key(a: dict) -> str | None:
    """Return a stable key for an author dict, preferring authorId.

    Falls back to ``name`` (with a one-time warning) when the authorId is
    missing. Returns ``None`` if neither is available.
    """
    global _FALLBACK_KEY_WARNED
    if not isinstance(a, dict):
        return None
    aid = a.get("authorId")
    if aid:
        return str(aid)
    name = a.get("name")
    if name:
        if not _FALLBACK_KEY_WARNED:
            log.warning("author dict missing authorId; falling back to name as key (further warnings suppressed)")
            _FALLBACK_KEY_WARNED = True
        return f"name:{name}"
    return None


def _set_vertex_attributes(
    g,
    node_keys: list[str],
    author_papers: dict[str, list[PaperRecord]],
    author_name: dict[str, str],
    author_details: dict[str, dict[str, Any]],
) -> None:
    """Populate the 9 per-author vertex attributes on ``g``.

    Walks ``node_keys`` once accumulating parallel lists, then assigns
    each as a `g.vs[...]` column. Defaults (``0`` for numeric, empty
    string for textual) are intentional — igraph's GraphML writer
    needs every column to be the same length, so we can't just skip
    authors with missing data.
    """
    names: list[str] = []
    years_entered: list[int] = []
    total_citation: list[int] = []
    h_index: list[int] = []
    paper_count_s2: list[int] = []
    paper_count_in_community: list[int] = []
    intra_network_citation: list[int] = []
    affiliation: list[str] = []

    for key in node_keys:
        papers_for_a = author_papers[key]
        # name
        name = author_name.get(key) or ""
        if not name:
            details = author_details.get(key) or {}
            name = details.get("name") or ""
        names.append(name)
        # year_entered
        years = [p.year for p in papers_for_a if p.year is not None]
        years_entered.append(min(years) if years else 0)
        # author details
        details = author_details.get(key) or {}
        total_citation.append(int(details.get("citationCount") or 0))
        h_index.append(int(details.get("hIndex") or 0))
        paper_count_s2.append(int(details.get("paperCount") or 0))
        paper_count_in_community.append(len(papers_for_a))
        intra_network_citation.append(
            sum((p.citation_count or 0) for p in papers_for_a)
        )
        affs = details.get("affiliations") or []
        if isinstance(affs, list) and affs:
            affiliation.append(" / ".join(str(a) for a in affs if a))
        else:
            affiliation.append("")

    g.vs["name"] = names
    g.vs["author_id"] = node_keys
    g.vs["year_entered"] = years_entered
    g.vs["total_citation"] = total_citation
    g.vs["h_index"] = h_index
    g.vs["paper_count_s2"] = paper_count_s2
    g.vs["paper_count_in_community"] = paper_count_in_community
    g.vs["intra_network_citation"] = intra_network_citation
    g.vs["affiliation"] = affiliation


def build_author_graph(
    collection: dict[str, PaperRecord],
    author_details: dict[str, dict[str, Any]] | None = None,
):
    """Build an undirected igraph collaboration graph.

    See module docstring for the node/edge attribute conventions. Returns
    an ``igraph.Graph`` (directed=False).
    """
    import igraph as ig

    author_details = author_details or {}

    # First pass — collect per-author papers / names / years and per-pair edge stats.
    author_papers: dict[str, list[PaperRecord]] = {}
    author_name: dict[str, str] = {}
    edge_stats: dict[frozenset[str], dict[str, Any]] = {}

    for paper in collection.values():
        authors = paper.authors or []
        if not authors:
            continue
        keyed: list[tuple[str, str]] = []
        for a in authors:
            key = _author_key(a)
            if not key:
                continue
            name = (a.get("name") if isinstance(a, dict) else "") or ""
            keyed.append((key, name))
        if not keyed:
            continue

        n_authors = len(keyed)
        for key, name in keyed:
            author_papers.setdefault(key, []).append(paper)
            if key not in author_name and name:
                author_name[key] = name

        # All unordered pairs
        for i in range(n_authors):
            for j in range(i + 1, n_authors):
                a_key, _ = keyed[i]
                b_key, _ = keyed[j]
                if a_key == b_key:
                    continue
                pair = frozenset({a_key, b_key})
                stats = edge_stats.get(pair)
                if stats is None:
                    stats = {
                        "strength": 0.0,
                        "first_year": None,
                        "last_year": None,
                        "n_collaborations": 0,
                    }
                    edge_stats[pair] = stats
                stats["strength"] += 1.0 / n_authors
                stats["n_collaborations"] += 1
                yr = paper.year
                if yr is not None:
                    if stats["first_year"] is None or yr < stats["first_year"]:
                        stats["first_year"] = yr
                    if stats["last_year"] is None or yr > stats["last_year"]:
                        stats["last_year"] = yr

    # Build vertex list
    node_keys = sorted(author_papers.keys())
    key_to_idx = {k: i for i, k in enumerate(node_keys)}

    g = ig.Graph(n=len(node_keys), directed=False)

    if not node_keys:
        log.info("Author collaboration graph: 0 authors, 0 edges (collection has no author data)")
        return g

    _set_vertex_attributes(g, node_keys, author_papers, author_name, author_details)

    if edge_stats:
        edge_pairs: list[tuple[int, int]] = []
        strengths: list[float] = []
        first_years: list[int] = []
        last_years: list[int] = []
        durations: list[int] = []
        n_collabs: list[int] = []
        for pair, stats in edge_stats.items():
            ks = list(pair)
            if len(ks) != 2:
                continue
            a, b = ks[0], ks[1]
            if a not in key_to_idx or b not in key_to_idx:
                continue
            edge_pairs.append((key_to_idx[a], key_to_idx[b]))
            strengths.append(float(stats["strength"]))
            fy = stats["first_year"] or 0
            ly = stats["last_year"] or 0
            first_years.append(fy)
            last_years.append(ly)
            if fy and ly:
                durations.append(ly - fy)
            else:
                durations.append(0)
            n_collabs.append(int(stats["n_collaborations"]))
        if edge_pairs:
            g.add_edges(edge_pairs)
            g.es["strength"] = strengths
            g.es["first_year"] = first_years
            g.es["last_year"] = last_years
            g.es["duration"] = durations
            g.es["n_collaborations"] = n_collabs

    log.info(
        "Author collaboration graph: %d authors, %d edges",
        g.vcount(), g.ecount(),
    )
    return g


def export_author_graphml(
    collection: dict[str, PaperRecord],
    author_details: dict[str, dict[str, Any]] | None,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Build the author collaboration graph and write it as GraphML."""
    g = build_author_graph(collection, author_details)
    if metadata:
        for k, v in metadata.items():
            g[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(path))
    log.info(
        "Wrote collaboration GraphML: %s (%d authors, %d edges)",
        path, g.vcount(), g.ecount(),
    )
