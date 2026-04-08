"""WalktrapClusterer — graph-based clustering via igraph's community_walktrap.

Operates on the citation graph built from ``ctx.collection``. Walktrap is a
random-walk-based community detection algorithm that targets a fixed number
of communities (``n_communities``).
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.network import build_citation_graph

log = logging.getLogger("citeclaw.cluster.walktrap")


class WalktrapClusterer:
    name = "walktrap"

    def __init__(self, *, n_communities: int = 3) -> None:
        self.n_communities = n_communities

    def cluster(self, signal, ctx) -> ClusterResult:
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
        gu = g.as_undirected(mode="collapse")
        n = min(self.n_communities, gu.vcount())
        if n <= 0:
            return [0] * g.vcount()
        try:
            return gu.community_walktrap().as_clustering(n=n).membership
        except Exception:
            return [i % n for i in range(g.vcount())]  # fallback
