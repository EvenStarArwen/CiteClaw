"""TopicModelClusterer — UMAP + HDBSCAN over S2 SPECTER2 embeddings.

A BERTopic-inspired pipeline that uses Semantic Scholar's precomputed
SPECTER2 embeddings (already cached in ``cache.db``) as input. Algorithms
are fixed (UMAP for dimensionality reduction, HDBSCAN for density-based
clustering) but every important hyperparameter is exposed.

This clusterer requires the optional ``topic_model`` extras::

    pip install 'citeclaw[topic_model]'

which pulls in ``umap-learn``, ``hdbscan``, ``scikit-learn``, and ``numpy``.
The imports are deferred to :meth:`cluster` so the package keeps loading
on installs that don't have them — only constructing and *running* the
clusterer fails.

Topic *naming* (label, summary, top keywords) is intentionally not part of
this clusterer. The :class:`~citeclaw.steps.cluster.Cluster` step runs an
algorithm-agnostic naming pipeline (c-TF-IDF + optional LLM call) over any
``ClusterResult`` after this clusterer returns.
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult

log = logging.getLogger("citeclaw.cluster.topic_model")

_EXTRAS_HINT = (
    "TopicModelClusterer requires the optional 'topic_model' extras: "
    "pip install 'citeclaw[topic_model]'"
)


class TopicModelClusterer:
    name = "topic_model"

    def __init__(
        self,
        *,
        # UMAP — defaults match BERTopic.
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        umap_metric: str = "cosine",
        random_state: int = 42,
        # HDBSCAN — defaults match BERTopic.
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        hdbscan_metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.umap_metric = umap_metric
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.hdbscan_metric = hdbscan_metric
        self.cluster_selection_method = cluster_selection_method

    def cluster(self, signal, ctx) -> ClusterResult:
        try:
            import numpy as np
            import umap
            import hdbscan
        except ImportError as exc:
            raise RuntimeError(_EXTRAS_HINT) from exc

        if not signal:
            return ClusterResult(membership={}, algorithm=self.name)

        # 1. Fetch embeddings (cache-warm; only newly-seen IDs hit the network).
        ids = [p.paper_id for p in signal]
        embeddings = ctx.s2.fetch_embeddings_batch(ids)

        # 2. Split into vectors-we-have vs vectors-we-don't. Papers with no
        #    embedding can't be clustered, so they all go to cluster -1.
        kept_ids: list[str] = []
        kept_vectors: list[list[float]] = []
        missing: list[str] = []
        for pid in ids:
            v = embeddings.get(pid)
            if v:
                kept_ids.append(pid)
                kept_vectors.append(v)
            else:
                missing.append(pid)
        if missing:
            log.warning(
                "topic_model: %d/%d papers have no SPECTER2 embedding; "
                "they'll be assigned to cluster -1 (noise)",
                len(missing), len(ids),
            )

        membership: dict[str, int] = {pid: -1 for pid in missing}

        # 3. Need at least min_cluster_size papers with embeddings; otherwise
        #    HDBSCAN can't form a single cluster. Treat that as "all noise".
        if len(kept_ids) < max(2, self.min_cluster_size):
            log.warning(
                "topic_model: only %d papers with embeddings (need >= %d); "
                "skipping clustering, all papers assigned to -1",
                len(kept_ids), max(2, self.min_cluster_size),
            )
            for pid in kept_ids:
                membership[pid] = -1
            return ClusterResult(
                membership=membership,
                metadata={},
                algorithm=self.name,
            )

        X = np.asarray(kept_vectors, dtype=np.float32)

        # 4. UMAP — reduce to n_components dims, preserving local structure.
        # n_neighbors must be < n_samples; clamp it.
        n_neighbors = min(self.n_neighbors, max(2, len(kept_ids) - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=min(self.n_components, max(2, len(kept_ids) - 1)),
            min_dist=self.min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
        )
        X_reduced = reducer.fit_transform(X)

        # 5. HDBSCAN — density-based clustering with -1 for noise.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, self.min_cluster_size),
            min_samples=self.min_samples,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(X_reduced)

        # 6. Build membership dict; -1 stays -1.
        for pid, label in zip(kept_ids, labels):
            membership[pid] = int(label)

        # 7. Per-cluster sizes (excluding -1).
        sizes: dict[int, int] = {}
        for cid in membership.values():
            if cid == -1:
                continue
            sizes[cid] = sizes.get(cid, 0) + 1
        metadata = {cid: ClusterMetadata(size=n) for cid, n in sizes.items()}

        log.info(
            "topic_model: %d papers, %d clusters, %d noise",
            len(kept_ids),
            len(sizes),
            sum(1 for cid in membership.values() if cid == -1),
        )
        return ClusterResult(
            membership=membership,
            metadata=metadata,
            algorithm=self.name,
        )
