"""Main path extraction — five search variants over a weighted DAG.

Every function here takes ``(g, weights)`` and returns a ``set[int]``
of edge indices that lie on the main path(s). The runner builds the
output subgraph by selecting these edges via
``g.subgraph_edges(..., delete_vertices=True)``.

The variants (all standard in the literature):

* :func:`local_forward` — priority-first search from every
  in-degree-0 vertex (Hummon & Doreian 1989, §"Network connectivity
  and search paths"; Pajek's default behaviour extended to all
  sources). At each frontier vertex pick every outgoing arc with the
  maximum weight; ties become parallel branches. Union across all
  source starts.

* :func:`local_backward` — mirror of local_forward: priority-first
  from every out-degree-0 vertex, following *incoming* max-weight
  arcs (Liu & Lu 2012, §"Backward Versus Forward"). Complements the
  forward view by tracing "roots of influence" rather than
  "offspring".

* :func:`global_cpm` — the critical path (Batagelj 2003, §6; Liu &
  Lu 2012, §"Global Versus Local"). Topological DP to find the
  path from some source to some sink with maximum *sum* of arc
  weights. Returns the arcs on that path. On ties returns just one
  representative.

* :func:`key_route` (default) — Liu & Lu (2012, §"Key-Route Search"):
  take the top-``k`` arcs by weight as key routes; from each, run
  forward priority-first from the target and backward priority-first
  from the source, union the three (key arc, forward trace, backward
  trace). Guarantees the single most-weighted arc in the DAG lands
  on the output.

* :func:`multi_local` — relaxed local_forward (Liu & Lu 2012,
  §"Multiple Versus Single"): at each frontier vertex include every
  outgoing arc whose weight is at least ``(1 - tolerance)`` of the
  per-vertex max, not just the exact max. ``tolerance = 0`` is the
  strict local_forward.

Each function is a pure ``Graph + weights → edge set`` transformation
with no side effects. Tolerance for ties is the source of the
"network of main paths" behaviour — a greedy unit-weight choice would
collapse to one path, but real weights frequently tie (especially for
SPC on small networks with few paths), and tracking ties is how
Hummon-Doreian's original DNA analysis surfaced the rich main-path
network from a 40-node graph.

Edge weights are expected to be non-negative. Zero-weight arcs can
arise under SPC when the arc is unreachable from any source or
sink — they're dropped by the ``>0`` guard in the greedy loops so
disconnected fragments don't appear in the output.
"""

from __future__ import annotations

import logging
from typing import Callable

import igraph as ig

log = logging.getLogger("citeclaw.mainpath.search")


def _greedy_traverse(
    g: ig.Graph,
    weights: list[int],
    start: int,
    mode: str,
) -> set[int]:
    """Priority-first traversal from ``start``.

    ``mode="out"`` walks along out-arcs (forward in time);
    ``mode="in"`` walks along in-arcs (backward in time). At each
    frontier vertex, pick every adjacent arc with weight equal to the
    per-vertex max. The frontier is dedup'd so a vertex reached along
    two paths is processed only once — this keeps the subgraph a
    tree-like DAG and the runtime linear in the output size.
    """
    on_path: set[int] = set()
    frontier: set[int] = {start}
    visited: set[int] = {start}
    while frontier:
        next_frontier: set[int] = set()
        for v in frontier:
            adj = g.incident(v, mode=mode)
            if not adj:
                continue
            max_w = max(weights[e] for e in adj)
            if max_w <= 0:
                continue
            for e in adj:
                if weights[e] != max_w:
                    continue
                on_path.add(e)
                nxt = g.es[e].target if mode == "out" else g.es[e].source
                if nxt not in visited:
                    visited.add(nxt)
                    next_frontier.add(nxt)
        frontier = next_frontier
    return on_path


def local_forward(g: ig.Graph, weights: list[int], **_: object) -> set[int]:
    """Priority-first from every in-degree-0 vertex, unioned."""
    on_path: set[int] = set()
    for v in range(g.vcount()):
        if g.indegree(v) == 0:
            on_path |= _greedy_traverse(g, weights, v, mode="out")
    return on_path


def local_backward(g: ig.Graph, weights: list[int], **_: object) -> set[int]:
    """Priority-first from every out-degree-0 vertex, walking backwards."""
    on_path: set[int] = set()
    for v in range(g.vcount()):
        if g.outdegree(v) == 0:
            on_path |= _greedy_traverse(g, weights, v, mode="in")
    return on_path


def global_cpm(g: ig.Graph, weights: list[int], **_: object) -> set[int]:
    """Critical path: maximum-summed-weight path from any source to any sink.

    One topological forward pass fills ``best[v]`` = max accumulated
    weight of any path ending at ``v`` from some source, and
    ``pred_edge[v]`` = the arc on the best path entering ``v``.
    Backtrack from the sink with the highest ``best`` value.

    Ties for the best sink are resolved by vertex index (deterministic
    but arbitrary). Ties within the backtrack are handled by the
    first-written-wins order in the forward pass.
    """
    n = g.vcount()
    if n == 0:
        return set()
    order = g.topological_sorting(mode="out")

    NEG_INF = float("-inf")
    best: list[float] = [NEG_INF] * n
    pred_edge: list[int] = [-1] * n
    for v in range(n):
        if g.indegree(v) == 0:
            best[v] = 0.0

    for v in order:
        if best[v] == NEG_INF:
            continue
        for e in g.incident(v, mode="out"):
            tgt = g.es[e].target
            new_best = best[v] + weights[e]
            if new_best > best[tgt]:
                best[tgt] = new_best
                pred_edge[tgt] = e

    sinks = [v for v in range(n) if g.outdegree(v) == 0]
    if not sinks:
        return set()
    best_sink = max(sinks, key=lambda v: best[v])
    if best[best_sink] == NEG_INF:
        return set()

    on_path: set[int] = set()
    v = best_sink
    while pred_edge[v] != -1:
        e = pred_edge[v]
        on_path.add(e)
        v = g.es[e].source
    return on_path


def key_route(
    g: ig.Graph,
    weights: list[int],
    *,
    k: int = 1,
    **_: object,
) -> set[int]:
    """Top-``k`` key-route search (Liu & Lu 2012).

    For each of the ``k`` highest-weighted arcs: include the arc;
    walk forward priority-first from its target; walk backward
    priority-first from its source. Union across all ``k`` seeds.

    Zero-weight arcs are skipped (they're unreachable from any
    source/sink path). ``k`` larger than the number of positive-weight
    arcs silently truncates.
    """
    if k <= 0:
        return set()
    positive_edges = [e for e in range(g.ecount()) if weights[e] > 0]
    positive_edges.sort(key=lambda e: weights[e], reverse=True)
    seeds = positive_edges[:k]

    on_path: set[int] = set(seeds)
    for e in seeds:
        src, tgt = g.es[e].source, g.es[e].target
        on_path |= _greedy_traverse(g, weights, tgt, mode="out")
        on_path |= _greedy_traverse(g, weights, src, mode="in")
    return on_path


def multi_local(
    g: ig.Graph,
    weights: list[int],
    *,
    tolerance: float = 0.0,
    **_: object,
) -> set[int]:
    """Per-vertex tolerance-relaxed forward priority-first.

    At each frontier vertex include every outgoing arc ``e`` with
    ``weights[e] >= (1 - tolerance) * max_at_vertex``. ``tolerance``
    is clamped to ``[0, 1]`` — ``0.0`` matches :func:`local_forward`
    exactly; ``1.0`` includes every outgoing arc.
    """
    tol = max(0.0, min(1.0, tolerance))
    on_path: set[int] = set()
    sources = [v for v in range(g.vcount()) if g.indegree(v) == 0]
    frontier = set(sources)
    visited = set(sources)
    while frontier:
        next_frontier: set[int] = set()
        for v in frontier:
            out_edges = g.incident(v, mode="out")
            if not out_edges:
                continue
            max_w = max(weights[e] for e in out_edges)
            if max_w <= 0:
                continue
            threshold = max_w * (1.0 - tol)
            for e in out_edges:
                if weights[e] < threshold:
                    continue
                on_path.add(e)
                nxt = g.es[e].target
                if nxt not in visited:
                    visited.add(nxt)
                    next_frontier.add(nxt)
        frontier = next_frontier
    return on_path


SEARCH_REGISTRY: dict[str, Callable[..., set[int]]] = {
    "local-forward":  local_forward,
    "local-backward": local_backward,
    "global":         global_cpm,
    "key-route":      key_route,
    "multi-local":    multi_local,
}
