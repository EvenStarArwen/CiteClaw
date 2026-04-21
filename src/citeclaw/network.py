"""Citation network graph — igraph-based, with PageRank + saturation metrics.

Public API:

* :func:`build_citation_graph` — build a directed igraph from a paper
  collection. Edge convention: ``A → B`` when A is in B's reference
  list (i.e. A is *cited by* B).
* :func:`compute_pagerank` — personalised PageRank over the graph;
  returns ``[(paper_id, score)]`` sorted descending. Optional
  ``seed_ids`` set drives the personalisation vector.
* :func:`compute_saturation` — pure 2-arg ratio helper; returns
  ``NaN`` for the 0/0 case so callers can distinguish "no data"
  from "0% saturation".
* :func:`per_paper_saturation` — single-paper saturation derived
  from a reference list and the current collection.
* :func:`saturation_for_paper` — convenience wrapper used by every
  ExpandBy* step that wants the dashboard metric; defensive against
  fake S2 clients in tests and never triggers new S2 calls.

The graph is directed (so PageRank can flow correctly along the
"who cites whom" direction). Vertex attributes are populated from
:class:`PaperRecord` fields and consumed by both the PageRank metric
(via ``compute_metric("pagerank", ...)``) and the cluster step (via
``WalktrapClusterer`` / ``LouvainClusterer``).
"""

from __future__ import annotations

import logging
from typing import Any

import igraph as ig

from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.network")


def build_citation_graph(collection: dict[str, PaperRecord]) -> ig.Graph:
    """Build a directed citation graph from the paper collection.

    Nodes: all papers in collection. Vertex attributes (``paper_id``,
    ``title``, ``year``, ``citation_count``, ``depth``, ``source``)
    are populated from the matching :class:`PaperRecord` fields.

    Edges:

    * ``A → B`` when A is in B's ``references`` list (A is *cited by*
      B). Only edges between papers in the collection are included —
      external references are dropped because we have no node for
      them.
    * ``supporting_papers`` adds an extra edge whose direction
      depends on the citing paper's ``source``: a forward-expansion
      paper inherits an edge **from** its supporting source (the
      source cited the new paper); a backward-expansion paper
      inherits an edge **to** its supporting source (the new paper
      cites the source). The ``edge_set`` dedup handles overlap with
      the references-pass.

    Self-loops (``ref_id == pid``) are dropped — they reflect data
    quirks in S2 references rather than real self-citations.
    """
    node_ids = sorted(collection.keys())
    id_to_idx = {pid: i for i, pid in enumerate(node_ids)}

    g = ig.Graph(n=len(node_ids), directed=True)
    g.vs["paper_id"] = node_ids
    g.vs["title"] = [collection[pid].title for pid in node_ids]
    g.vs["year"] = [collection[pid].year for pid in node_ids]
    g.vs["citation_count"] = [collection[pid].citation_count or 0 for pid in node_ids]
    g.vs["depth"] = [collection[pid].depth for pid in node_ids]
    g.vs["source"] = [collection[pid].source for pid in node_ids]

    node_set = set(node_ids)
    edge_set: set[tuple[int, int]] = set()
    for pid in node_ids:
        paper = collection[pid]
        for ref_id in paper.references:
            if ref_id in node_set and ref_id != pid:
                edge_set.add((id_to_idx[ref_id], id_to_idx[pid]))
        for sp_id in paper.supporting_papers:
            if sp_id in node_set and sp_id != pid:
                if paper.source == "forward":
                    edge_set.add((id_to_idx[sp_id], id_to_idx[pid]))
                else:
                    edge_set.add((id_to_idx[pid], id_to_idx[sp_id]))

    if edge_set:
        g.add_edges(list(edge_set))

    log.info("Citation graph: %d nodes, %d edges", g.vcount(), g.ecount())
    return g


def compute_pagerank(
    g: ig.Graph,
    seed_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Compute personalized PageRank. Returns [(paper_id, score)] sorted descending."""
    if g.vcount() == 0:
        return []

    reset = None
    if seed_ids:
        reset = [1.0 if v["paper_id"] in seed_ids else 0.0 for v in g.vs]

    scores = g.personalized_pagerank(reset=reset, directed=True, damping=0.85)

    ranked = [
        (g.vs[i]["paper_id"], scores[i])
        for i in range(g.vcount())
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def compute_saturation(
    refs_already_in_collection: int,
    newly_accepted_refs: int,
) -> float:
    """
    Per-paper saturation: proportion of valid references already in collection.
    Returns NaN if no valid references were found (0/0 is undefined).
    """
    total_valid = refs_already_in_collection + newly_accepted_refs
    if total_valid == 0:
        return float("nan")
    return refs_already_in_collection / total_valid


def per_paper_saturation(
    paper_id: str,
    refs: list[str] | None,
    collection: dict[str, PaperRecord],
) -> float | None:
    """Compute the per-paper saturation as a single number.

    Saturation = (paper refs that are already in ``collection``) / (paper refs).

    A value close to 1.0 means the paper mostly cites work already in the
    corpus (the corpus is "saturated" w.r.t. this paper). A value close to
    0.0 means the paper opens up new territory.

    The paper itself is excluded from the comparison so a self-citation
    (rare) doesn't inflate the score. Returns ``None`` if the refs list is
    missing or empty (e.g. cache miss) — the caller should treat ``None``
    as "no signal yet".

    Callers typically pass ``refs`` from ``ctx.s2.cached_reference_ids(...)``
    so the saturation metric never triggers new S2 calls.
    """
    if not refs:
        return None
    valid = [r for r in refs if r and r != paper_id]
    if not valid:
        return None
    in_coll = sum(1 for r in valid if r in collection)
    return in_coll / len(valid)


def saturation_for_paper(paper, ctx) -> float | None:
    """Convenience wrapper used by steps: compute per-paper saturation
    using the S2 cache, defensive against fake clients in tests.

    Returns ``None`` if the S2 client doesn't expose ``cached_reference_ids``
    or the references aren't cached. Never triggers new S2 calls.
    """
    s2 = getattr(ctx, "s2", None)
    if s2 is None:
        return None
    fn = getattr(s2, "cached_reference_ids", None)
    if not callable(fn):
        return None
    refs = fn(paper.paper_id)
    return per_paper_saturation(paper.paper_id, refs, ctx.collection)
