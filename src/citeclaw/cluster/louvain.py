"""LouvainClusterer — graph-based clustering via igraph's community_multilevel.

Louvain auto-determines the community count by maximising modularity, so
``n_communities`` is advisory and used only as a fallback when igraph fails
to run the algorithm at all.
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.network import build_citation_graph

log = logging.getLogger("citeclaw.cluster.louvain")


class LouvainClusterer:
    name = "louvain"

    def __init__(self, *, n_communities: int | None = None) -> None:
        # Louvain auto-determines community count; n_communities is advisory.
        self.n_communities = n_communities

    def cluster(self, signal, ctx) -> ClusterResult:
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        if g.vcount() == 0:
            return ClusterResult(membership={}, algorithm=self.name)
        gu = g.as_undirected(mode="collapse")
        try:
            membership_list = gu.community_multilevel().membership  # igraph's Louvain
        except Exception:
            n = self.n_communities or 3
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
