"""Cycle handling — collapse or transform strongly connected components
so :mod:`citeclaw.mainpath.weights` can assume a DAG.

Real citation graphs are usually "almost" DAGs: snowball corpora pick
up forward-citing arcs from preprints, same-special-issue cross-cites,
or occasional S2 metadata errors where a cited paper is dated later
than the citer. Main path analysis only makes sense on a DAG, so one
of the two policies below must run before weight computation.

* :func:`shrink_family` (default) — collapse every non-trivial SCC
  into a single representative vertex, per Liu, Lu & Ho (2019,
  §"Loop and the solution"). Justified there on the grounds that
  papers in a cycle are usually tightly related in topic and deserve
  to appear together on the main path as one "family" node. Chosen as
  the CiteClaw default because it's the simpler story for users and
  produces a less cluttered output subgraph.

* :func:`preprint_transform` — Batagelj (2003, §5): for each SCC,
  duplicate every node u into a "preprint" u′, rewire non-SCC
  out-arcs of u to originate from u′, delete all SCC-internal arcs,
  and add an arc ``u → v′`` for every ordered pair (u, v) of nodes
  in the same SCC (including ``u = v``). Preserves SCC members as
  individual vertices at the cost of 2× node count inside SCCs and
  the fact that preprint vertices get synthetic paper ids.

Both functions return ``(dag, trace)`` where ``trace`` is a
:class:`citeclaw.mainpath.base.CyclePolicyTrace` recording which SCCs
were found and what was done. :mod:`.weights` will raise on non-DAG
input — always route through one of these before calling it.

Representative selection for shrink-family: oldest ``year`` (with
ties broken by higher ``citation_count``, then lower vertex index).
This matches the "earliest canonical reference" intuition and keeps
the collapsed node's metadata as close as possible to the foundational
paper in the cluster.
"""

from __future__ import annotations

import logging
from typing import Callable

import igraph as ig

from citeclaw.mainpath.base import CyclePolicyTrace

log = logging.getLogger("citeclaw.mainpath.cycles")


def _vertex_attrs(g: ig.Graph) -> list[str]:
    """Return the list of vertex attribute names currently on ``g``."""
    return list(g.vs.attributes())


def _representative(g: ig.Graph, members: list[int]) -> int:
    """Pick the SCC representative: oldest year, highest cite, lowest idx.

    Missing ``year`` is treated as "very large" (end-of-time) so it
    loses against any real year; missing ``citation_count`` defaults
    to 0. Always returns a valid vertex index from ``members``.
    """
    def key(u: int) -> tuple[int, int, int]:
        attrs = g.vs[u].attributes()
        year = attrs.get("year")
        year_key = 99999 if year is None else int(year)
        cc = attrs.get("citation_count")
        cc_key = -int(cc) if cc is not None else 0
        return (year_key, cc_key, u)
    return min(members, key=key)


def shrink_family(g: ig.Graph) -> tuple[ig.Graph, CyclePolicyTrace]:
    """Collapse each non-trivial SCC into a single representative node.

    Returns a fresh ``igraph.Graph`` carrying the representatives'
    vertex attributes unchanged. Inter-SCC edges are re-anchored to
    the representatives and deduplicated; SCC-internal edges are
    dropped (they would become self-loops). Edge attributes are not
    preserved — the weight step computes fresh values anyway, and
    merging edge attributes across a contraction has no canonical
    semantics.

    If the input is already a DAG this returns a copy and a
    trace with empty ``scc_sizes``.
    """
    comps = g.connected_components(mode="strong")
    membership = list(comps.membership)
    all_sizes = [len(c) for c in comps]
    n_trivial_sids = [
        sid for sid, sz in enumerate(all_sizes) if sz >= 2
    ]

    if not n_trivial_sids:
        copy = g.copy()
        return copy, CyclePolicyTrace(
            policy="shrink",
            n_nodes_before=g.vcount(),
            n_nodes_after=g.vcount(),
            n_edges_before=g.ecount(),
            n_edges_after=g.ecount(),
        )

    rep_of_scc: dict[int, int] = {}
    supernode_members: dict[str, list[str]] = {}
    for sid in range(len(comps)):
        members = list(comps[sid])
        rep = _representative(g, members) if len(members) > 1 else members[0]
        rep_of_scc[sid] = rep
        if len(members) > 1:
            rep_pid = g.vs[rep]["paper_id"]
            supernode_members[rep_pid] = [g.vs[u]["paper_id"] for u in members]

    rep_for_v = [rep_of_scc[membership[v]] for v in range(g.vcount())]
    unique_reps = sorted(set(rep_for_v))
    rep_to_new_idx = {rep: i for i, rep in enumerate(unique_reps)}

    new_g = ig.Graph(n=len(unique_reps), directed=True)
    for attr in _vertex_attrs(g):
        new_g.vs[attr] = [g.vs[rep][attr] for rep in unique_reps]

    edge_set: set[tuple[int, int]] = set()
    for e in g.es:
        new_src = rep_to_new_idx[rep_for_v[e.source]]
        new_tgt = rep_to_new_idx[rep_for_v[e.target]]
        if new_src != new_tgt:
            edge_set.add((new_src, new_tgt))
    if edge_set:
        new_g.add_edges(list(edge_set))

    log.info(
        "shrink_family: %d non-trivial SCC(s) collapsed; "
        "nodes %d -> %d, edges %d -> %d",
        len(n_trivial_sids),
        g.vcount(), new_g.vcount(),
        g.ecount(), new_g.ecount(),
    )
    return new_g, CyclePolicyTrace(
        policy="shrink",
        scc_sizes=[all_sizes[sid] for sid in n_trivial_sids],
        supernode_members=supernode_members,
        n_nodes_before=g.vcount(),
        n_nodes_after=new_g.vcount(),
        n_edges_before=g.ecount(),
        n_edges_after=new_g.ecount(),
    )


_PREPRINT_SUFFIX = "__preprint"


def preprint_transform(g: ig.Graph) -> tuple[ig.Graph, CyclePolicyTrace]:
    """Preprint-transform every non-trivial SCC per Batagelj (2003, §5).

    For each non-trivial SCC:

    1. Add a preprint copy u′ for every member u (inherits u's
       attributes except ``paper_id`` which gets a ``__preprint``
       suffix so downstream code can tell real papers from preprints).
    2. Delete all SCC-internal arcs.
    3. Rewire every non-SCC-internal out-arc (u, w) from u ∈ SCC to
       (u′, w).
    4. Add an arc ``u → v′`` for every ordered pair (u, v) where both
       are in the same SCC, including ``u == v`` — i.e. ``k²`` new
       arcs per SCC of size ``k``.

    Non-SCC incoming arcs (w, u) where w ∉ SCC stay on u unchanged.
    Arcs between two different non-trivial SCCs are treated as
    "non-SCC-internal" at their source SCC: they're rewired out of
    the source preprint and kept pointing at the target's original
    (non-preprint) vertex.

    Returns ``(dag, trace)``. ``dag`` has ``g.vcount() + Σ|SCC|``
    vertices; the trace's ``supernode_members`` is empty since
    nothing is collapsed. If the input is already a DAG this returns
    a copy and an empty trace.
    """
    comps = g.connected_components(mode="strong")
    all_sizes = [len(c) for c in comps]
    non_trivial_sids = [sid for sid, sz in enumerate(all_sizes) if sz >= 2]

    if not non_trivial_sids:
        copy = g.copy()
        return copy, CyclePolicyTrace(
            policy="preprint",
            n_nodes_before=g.vcount(),
            n_nodes_after=g.vcount(),
            n_edges_before=g.ecount(),
            n_edges_after=g.ecount(),
        )

    scc_of = list(comps.membership)
    non_trivial_set = set(non_trivial_sids)
    scc_nodes = [v for v in range(g.vcount()) if scc_of[v] in non_trivial_set]
    n_original = g.vcount()
    preprint_of = {v: n_original + i for i, v in enumerate(scc_nodes)}

    new_n = n_original + len(scc_nodes)
    new_g = ig.Graph(n=new_n, directed=True)

    for attr in _vertex_attrs(g):
        values = list(g.vs[attr]) + [g.vs[v][attr] for v in scc_nodes]
        new_g.vs[attr] = values

    if "paper_id" in g.vs.attributes():
        for v in scc_nodes:
            new_g.vs[preprint_of[v]]["paper_id"] = (
                g.vs[v]["paper_id"] + _PREPRINT_SUFFIX
            )

    edges: set[tuple[int, int]] = set()
    for e in g.es:
        src, tgt = e.source, e.target
        same_scc_nontrivial = (
            scc_of[src] == scc_of[tgt] and scc_of[src] in non_trivial_set
        )
        if same_scc_nontrivial:
            continue
        if scc_of[src] in non_trivial_set:
            edges.add((preprint_of[src], tgt))
        else:
            edges.add((src, tgt))

    for sid in non_trivial_sids:
        members = list(comps[sid])
        for u in members:
            for v in members:
                edges.add((u, preprint_of[v]))

    if edges:
        new_g.add_edges(list(edges))

    log.info(
        "preprint_transform: %d non-trivial SCC(s); "
        "nodes %d -> %d, edges %d -> %d",
        len(non_trivial_sids),
        g.vcount(), new_g.vcount(),
        g.ecount(), new_g.ecount(),
    )
    return new_g, CyclePolicyTrace(
        policy="preprint",
        scc_sizes=[all_sizes[sid] for sid in non_trivial_sids],
        n_nodes_before=g.vcount(),
        n_nodes_after=new_g.vcount(),
        n_edges_before=g.ecount(),
        n_edges_after=new_g.ecount(),
    )


CYCLE_REGISTRY: dict[str, Callable[[ig.Graph], tuple[ig.Graph, CyclePolicyTrace]]] = {
    "shrink":   shrink_family,
    "preprint": preprint_transform,
}
