"""WalktrapClusterer — graph-based clustering via igraph's community_walktrap.

Operates on the citation graph built from ``ctx.collection`` by
:func:`citeclaw.network.build_citation_graph`. Walktrap is a
random-walk-based community detection algorithm that targets a fixed
number of communities (``n_communities``) — different from Louvain
(:mod:`citeclaw.cluster.louvain`), which auto-determines the count
by maximising modularity.

The algorithm runs over the *whole* collection so cluster IDs reflect
the full citation neighbourhood, but the returned ``membership`` dict
is restricted to the signal papers — that's what callers (Rerank
diversity, Cluster step's GraphML attribute write-out) actually
consume.
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.network import build_citation_graph

log = logging.getLogger("citeclaw.cluster.walktrap")


class WalktrapClusterer:
    """Cluster the citation graph into ``n_communities`` Walktrap communities."""

    name = "walktrap"

    def __init__(self, *, n_communities: int = 3) -> None:
        self.n_communities = n_communities

    def cluster(self, signal, ctx) -> ClusterResult:
        """Run Walktrap over ``ctx.collection`` and project membership for ``signal``.

        Returns an empty :class:`ClusterResult` when the citation graph
        has no vertices. Otherwise the partition is computed over the
        full collection (so neighbourhood structure beyond the signal
        is honoured) and the returned ``membership`` map is then
        restricted to the signal-paper subset.
        """
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        if g.vcount() == 0:
            return ClusterResult(membership={}, algorithm=self.name)
        membership_list = self._partition(g)
        membership = {
            v["paper_id"]: int(membership_list[v.index])
            for v in g.vs
            if v["paper_id"] in sig_ids
        }
        # Per-cluster sizes (only for clusters that contain at least one
        # signal paper). The naming pipeline reads these to skip empty
        # clusters and to populate ClusterMetadata.size.
        sizes: dict[int, int] = {}
        for cid in membership.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        metadata = {cid: ClusterMetadata(size=n) for cid, n in sizes.items()}
        log.info(
            "walktrap: %d signal papers, %d communities target=%d",
            len(membership),
            len(sizes),
            self.n_communities,
        )
        return ClusterResult(
            membership=membership,
            metadata=metadata,
            algorithm=self.name,
        )

    def _partition(self, g):
        """Compute a Walktrap partition over the undirected collapse of ``g``.

        Returns the per-vertex ``membership`` list. Falls back to
        round-robin assignment ``[i % n for i in range(g.vcount())]``
        when (a) ``n_communities <= 0`` (degenerate config) or (b) the
        igraph algorithm raises — typically on disconnected graphs
        smaller than the requested ``n``. The fallback path now logs
        at WARNING so the silent rotation is visible in postmortem
        diagnostics (audit "no silent failure" rule).
        """
        gu = g.as_undirected(mode="collapse")
        n = min(self.n_communities, gu.vcount())
        if n <= 0:
            return [0] * g.vcount()
        try:
            return gu.community_walktrap().as_clustering(n=n).membership
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "walktrap: community_walktrap raised on graph of %d vertices "
                "(n=%d); falling back to round-robin assignment: %s",
                gu.vcount(), n, exc,
            )
            return [i % n for i in range(g.vcount())]
