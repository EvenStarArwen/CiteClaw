"""LouvainClusterer — graph-based clustering via igraph's community_multilevel.

Operates on the citation graph built from ``ctx.collection`` by
:func:`citeclaw.network.build_citation_graph`. Louvain (igraph's
``community_multilevel``) auto-determines the community count by
maximising modularity, in contrast to Walktrap
(:mod:`citeclaw.cluster.walktrap`), which targets a fixed
``n_communities``. The ``n_communities`` constructor knob here is
**advisory** — it's only used as the divisor for the round-robin
fallback when the igraph algorithm raises (e.g. on disconnected
graphs igraph's C bindings occasionally trip on).

Same projection contract as Walktrap: the partition runs over the
full collection so cluster IDs reflect the true neighbourhood, then
the returned ``membership`` map is restricted to the signal-paper
subset.
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.network import build_citation_graph

log = logging.getLogger("citeclaw.cluster.louvain")


class LouvainClusterer:
    """Cluster the citation graph via igraph's ``community_multilevel`` Louvain."""

    name = "louvain"

    def __init__(self, *, n_communities: int | None = None) -> None:
        # Louvain auto-determines community count; n_communities is
        # advisory and only consumed by the round-robin fallback below.
        self.n_communities = n_communities

    def cluster(self, signal, ctx) -> ClusterResult:
        """Run Louvain over ``ctx.collection`` and project membership for ``signal``.

        Returns an empty :class:`ClusterResult` when the citation graph
        has no vertices. The partition is computed over the full
        collection so neighbourhood structure beyond the signal is
        honoured; the returned ``membership`` map is then restricted
        to the signal-paper subset.

        When ``community_multilevel`` raises (typically on degenerate
        disconnected graphs), falls back to round-robin assignment over
        ``n_communities or 3`` buckets and logs the silent rotation at
        WARNING (audit "no silent failure" rule).
        """
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        if g.vcount() == 0:
            return ClusterResult(membership={}, algorithm=self.name)
        gu = g.as_undirected(mode="collapse")
        try:
            membership_list = gu.community_multilevel().membership  # igraph's Louvain
        except Exception as exc:  # noqa: BLE001
            n = self.n_communities or 3
            log.warning(
                "louvain: community_multilevel raised on graph of %d vertices "
                "(fallback n=%d); falling back to round-robin assignment: %s",
                gu.vcount(), n, exc,
            )
            membership_list = [i % n for i in range(gu.vcount())]
        membership = {
            v["paper_id"]: int(membership_list[v.index])
            for v in g.vs
            if v["paper_id"] in sig_ids
        }
        sizes: dict[int, int] = {}
        for cid in membership.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        metadata = {cid: ClusterMetadata(size=n) for cid, n in sizes.items()}
        log.info(
            "louvain: %d signal papers, %d communities",
            len(membership),
            len(sizes),
        )
        return ClusterResult(
            membership=membership,
            metadata=metadata,
            algorithm=self.name,
        )
