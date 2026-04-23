"""SPC, SPLC, SPNP — Batagelj (2003) linear-time traversal weights.

All three weights reduce to "Search Path Count on an extended network
with virtual source s and sink t" (Batagelj 2003, §3.3). The three
variants differ only in how s and t are connected to ``U``:

* :func:`compute_spc` — Search Path Count. s connects only to
  in-degree-0 vertices (``Min R``); t connects only from out-degree-0
  vertices (``Max R``). Weight of arc (u, v) = number of distinct
  paths from a source vertex to a sink vertex that pass through
  (u, v). Obeys Kirchhoff's node law (inflow = outflow through every
  intermediate vertex), which is why Batagelj and Price & Evans
  recommend it as the principled default.

* :func:`compute_splc` — Search Path Link Count. s connects to
  *every* u ∈ U (every vertex can originate a path). Models the
  intuition that every publication also emits knowledge, not just the
  foundational ones. SPLC(u, v) ≥ SPC(u, v) pointwise (§4.1).

* :func:`compute_spnp` — Search Path Node Pair. s connects to every
  u and t connects from every u (every vertex is both a possible
  origin and a possible destination). Middle-of-network arcs receive
  disproportionately high weight. SPNP ≥ SPLC ≥ SPC.

Input must be a DAG — run :mod:`citeclaw.mainpath.cycles` first on
cyclic citation graphs. Arc direction follows the convention in
:mod:`citeclaw.network`: ``A → B`` iff A is cited by B, so A is the
older paper and B the newer one. Topological order is therefore
chronological (oldest first).

Implementation: one forward DP pass in topological order to compute
``N_minus`` (path-count from virtual source to each vertex) and one
reverse pass to compute ``N_plus``. The weight of an arc (u, v) ∈ R
is then ``N_minus[u] · N_plus[v]`` — one multiplication per arc,
O(n + m) total.

The shared helper :func:`_compute_path_counts` parameterises the
source/sink extension by ``source_mode`` / ``sink_mode``, so the three
public functions are three-liners dispatching on the right mode.

All returned weight lists are in **edge-index order** (same ordering
as ``g.es``). Values are Python ``int`` — arbitrary precision, so
dense DAGs don't overflow. For very large networks, the caller may
want to take logarithms before visualising (Batagelj 2003, §3.6).
"""

from __future__ import annotations

import logging
from typing import Callable

import igraph as ig

log = logging.getLogger("citeclaw.mainpath.weights")


def _compute_path_counts(
    g: ig.Graph,
    *,
    source_mode: str,
    sink_mode: str,
) -> tuple[list[int], list[int]]:
    """Compute N_minus / N_plus in one topological pass each.

    ``source_mode``:
        ``"minimal"`` — only in-degree-0 vertices get the +1 from the
        virtual source (SPC).
        ``"every"`` — every vertex gets +1 (SPLC, SPNP).

    ``sink_mode``:
        ``"maximal"`` — only out-degree-0 vertices get the +1 from the
        virtual sink (SPC, SPLC).
        ``"every"`` — every vertex gets +1 (SPNP).

    The +1 reflects the virtual s → u (resp. u → t) arc that is
    present in Batagelj's standard form; everything else is the sum
    over in-DAG predecessors / successors.
    """
    n = g.vcount()
    order = g.topological_sorting(mode="out")
    in_deg = g.indegree()
    out_deg = g.outdegree()

    n_minus = [0] * n
    for u in order:
        base = 1 if (source_mode == "every" or in_deg[u] == 0) else 0
        n_minus[u] = base + sum(n_minus[p] for p in g.predecessors(u))

    n_plus = [0] * n
    for u in reversed(order):
        base = 1 if (sink_mode == "every" or out_deg[u] == 0) else 0
        n_plus[u] = base + sum(n_plus[s] for s in g.successors(u))

    return n_minus, n_plus


def _arc_weights(
    g: ig.Graph,
    n_minus: list[int],
    n_plus: list[int],
) -> list[int]:
    """Weight of arc (u, v) is ``N_minus[u] * N_plus[v]`` — one pass."""
    weights = [0] * g.ecount()
    for e in g.es:
        weights[e.index] = n_minus[e.source] * n_plus[e.target]
    return weights


def compute_spc(g: ig.Graph) -> list[int]:
    """SPC weights per Batagelj (2003, §3.2).

    Paths run from in-degree-0 vertices to out-degree-0 vertices.
    Raises ``ValueError`` if ``g`` is not a DAG.
    """
    n_minus, n_plus = _compute_path_counts(
        g, source_mode="minimal", sink_mode="maximal",
    )
    return _arc_weights(g, n_minus, n_plus)


def compute_splc(g: ig.Graph) -> list[int]:
    """SPLC weights — every vertex is a possible path origin (§3.3)."""
    n_minus, n_plus = _compute_path_counts(
        g, source_mode="every", sink_mode="maximal",
    )
    return _arc_weights(g, n_minus, n_plus)


def compute_spnp(g: ig.Graph) -> list[int]:
    """SPNP weights — every vertex is both origin and destination (§3.3)."""
    n_minus, n_plus = _compute_path_counts(
        g, source_mode="every", sink_mode="every",
    )
    return _arc_weights(g, n_minus, n_plus)


WEIGHT_REGISTRY: dict[str, Callable[[ig.Graph], list[int]]] = {
    "spc":  compute_spc,
    "splc": compute_splc,
    "spnp": compute_spnp,
}
