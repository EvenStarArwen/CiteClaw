"""Clustering algorithms — graph-based community detection + embedding-based topic modeling.

Every clusterer implements the :class:`~citeclaw.cluster.base.Clusterer` Protocol,
returning a :class:`~citeclaw.cluster.base.ClusterResult` whose ``membership``
dict maps each paper id to an integer cluster id (``-1`` = noise/unassigned).

The registry is a flat string ↔ class mapping consumed by
:func:`build_clusterer`, which accepts either a bare algorithm name
(``"walktrap"``) or a dict ``{"type": "walktrap", "n_communities": 5}``.
"""

from __future__ import annotations

from citeclaw.cluster.base import Clusterer, ClusterMetadata, ClusterResult
from citeclaw.cluster.louvain import LouvainClusterer
from citeclaw.cluster.topic_model import TopicModelClusterer
from citeclaw.cluster.walktrap import WalktrapClusterer

CLUSTERER_REGISTRY: dict[str, type] = {
    "walktrap":    WalktrapClusterer,
    "louvain":     LouvainClusterer,
    "topic_model": TopicModelClusterer,
}


def build_clusterer(spec: str | dict) -> Clusterer:
    """Construct a Clusterer from a name or a ``{"type": ..., **kwargs}`` dict."""
    if isinstance(spec, str):
        cls = CLUSTERER_REGISTRY.get(spec)
        if cls is None:
            raise ValueError(
                f"Unknown clusterer {spec!r}. Available: {sorted(CLUSTERER_REGISTRY)}"
            )
        return cls()
    if isinstance(spec, dict):
        algo = spec.get("type")
        if algo is None:
            raise ValueError(
                f"Clusterer spec dict missing required 'type' key: {spec!r}"
            )
        cls = CLUSTERER_REGISTRY.get(algo)
        if cls is None:
            raise ValueError(
                f"Unknown clusterer {algo!r}. Available: {sorted(CLUSTERER_REGISTRY)}"
            )
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return cls(**kwargs)
    raise ValueError(f"Bad clusterer spec: {spec!r}")


__all__ = [
    "Clusterer",
    "ClusterMetadata",
    "ClusterResult",
    "WalktrapClusterer",
    "LouvainClusterer",
    "TopicModelClusterer",
    "CLUSTERER_REGISTRY",
    "build_clusterer",
]
