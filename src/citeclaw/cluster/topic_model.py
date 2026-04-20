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


def _compute_cluster_params(n: int, *, size_factor: float = 1.0) -> dict[str, int]:
    """Adaptive UMAP + HDBSCAN parameters tuned to corpus size ``n``.

    Three coupled knobs scale together:

    * ``min_cluster_size`` — HDBSCAN's minimum-cluster threshold. A
      static value of 10 collapses every topic into noise on corpora
      below ~50 papers. Scales as ``n // 30`` clamped to ``[5, 100]``:
      5 on small corpora, 100 on large ones so micro-clusters don't
      flood at scale.
    * ``min_samples`` — HDBSCAN density strictness. Defaults to
      ``min_cluster_size`` when unset (overly strict); we cap it at 20%
      of ``min_cluster_size`` so ~30% more papers escape the ``-1`` noise
      bucket on small corpora.
    * ``n_neighbors`` — UMAP local-vs-global balance. Scales as ``n // 50``
      clamped to ``[5, 30]`` so small corpora preserve local structure
      and large corpora use the global-structure default.

    Worked examples (size_factor=1.0)::

        n=50    → mcs=5,  min_samples=1,  n_neighbors=5
        n=300   → mcs=10, min_samples=2,  n_neighbors=6
        n=1000  → mcs=33, min_samples=7,  n_neighbors=20
        n=3000  → mcs=100,min_samples=20, n_neighbors=30
        n=10000 → mcs=100 (cap), min_samples=20 (cap), n_neighbors=30 (cap)

    ``size_factor`` shrinks ``min_cluster_size`` (and, via the 0.2× ratio,
    ``min_samples`` — the density / noise threshold) so shared-anchor
    corpora (e.g. every paper in the set already passes the topic's core
    terms) still surface meaningfully-sized sub-clusters. ExpandBySearch
    passes ``size_factor=0.5``; the pipeline-wide ``Cluster`` step keeps
    the default 1.0.
    """
    mcs_base = max(5, min(100, n // 30))
    mcs = max(2, int(round(mcs_base * size_factor)))
    return {
        "min_cluster_size": mcs,
        "min_samples": max(1, int(0.2 * mcs)),
        "n_neighbors": max(5, min(30, n // 50)),
    }


def cluster_embeddings(
    ids: list[str],
    embeddings_by_id: dict[str, list[float] | None],
    *,
    size_factor: float = 1.0,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    n_neighbors: int | None = None,
    n_components: int = 5,
    random_state: int = 42,
    umap_metric: str = "cosine",
    hdbscan_metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> tuple[dict[str, int], dict[str, int]]:
    """Pure UMAP → HDBSCAN over pre-fetched embeddings.

    Returns ``(membership, effective_params)``. ``membership`` maps each
    input id to a cluster id (or -1 for noise / missing embedding).
    ``effective_params`` lists the actual ``min_cluster_size`` /
    ``min_samples`` / ``n_neighbors`` that were applied (useful for
    downstream display / logging).
    """
    try:
        import numpy as np
        import umap
        import hdbscan
    except ImportError as exc:
        raise RuntimeError(_EXTRAS_HINT) from exc

    if not ids:
        return {}, {"min_cluster_size": 0, "min_samples": 0, "n_neighbors": 0}

    kept_ids: list[str] = []
    kept_vectors: list[list[float]] = []
    missing: list[str] = []
    for pid in ids:
        v = embeddings_by_id.get(pid)
        if v:
            kept_ids.append(pid)
            kept_vectors.append(v)
        else:
            missing.append(pid)

    membership: dict[str, int] = {pid: -1 for pid in missing}

    adaptive = _compute_cluster_params(len(kept_ids), size_factor=size_factor)
    effective_mcs = min_cluster_size if min_cluster_size is not None else adaptive["min_cluster_size"]
    effective_min_samples = min_samples if min_samples is not None else adaptive["min_samples"]
    effective_n_neighbors = n_neighbors if n_neighbors is not None else adaptive["n_neighbors"]

    if len(kept_ids) < max(2, effective_mcs):
        for pid in kept_ids:
            membership[pid] = -1
        return membership, {
            "min_cluster_size": effective_mcs,
            "min_samples": effective_min_samples,
            "n_neighbors": effective_n_neighbors,
        }

    X = np.asarray(kept_vectors, dtype=np.float32)
    nn = min(effective_n_neighbors, max(2, len(kept_ids) - 1))
    try:
        reducer = umap.UMAP(
            n_neighbors=nn,
            n_components=min(n_components, max(2, len(kept_ids) - 1)),
            min_dist=0.0,
            metric=umap_metric,
            random_state=random_state,
        )
        X_reduced = reducer.fit_transform(X)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, effective_mcs),
            min_samples=effective_min_samples,
            metric=hdbscan_metric,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(X_reduced)
    except Exception as exc:  # noqa: BLE001
        # UMAP's spectral-init can fall over with scipy-eigh errors on
        # tiny / degenerate corpora (<~30 papers). Rather than crash the
        # whole worker, treat everything as noise and let the caller
        # decide what to do.
        log.warning(
            "topic_model: UMAP/HDBSCAN failed (%s); returning all-noise membership",
            exc,
        )
        for pid in kept_ids:
            membership[pid] = -1
        return membership, {
            "min_cluster_size": effective_mcs,
            "min_samples": effective_min_samples,
            "n_neighbors": effective_n_neighbors,
        }
    for pid, label in zip(kept_ids, labels):
        membership[pid] = int(label)
    return membership, {
        "min_cluster_size": effective_mcs,
        "min_samples": effective_min_samples,
        "n_neighbors": effective_n_neighbors,
    }


class TopicModelClusterer:
    name = "topic_model"

    def __init__(
        self,
        *,
        # UMAP. ``n_neighbors=None`` → adaptive via :func:`_compute_cluster_params`.
        n_neighbors: int | None = None,
        n_components: int = 5,
        min_dist: float = 0.0,
        umap_metric: str = "cosine",
        random_state: int = 42,
        # HDBSCAN. ``min_cluster_size=None`` / ``min_samples=None`` → adaptive.
        # Hand-set values still win; the adaptive path only fires when
        # the caller didn't pin a number.
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        hdbscan_metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        # size_factor < 1 shrinks adaptive min_cluster_size + min_samples;
        # ExpandBySearch passes 0.5. Ignored when min_cluster_size is pinned.
        size_factor: float = 1.0,
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
        self.size_factor = size_factor

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

        # Resolve adaptive parameters from the number of papers we can
        # actually cluster. Caller-set values (non-None attributes) always
        # win over the adaptive defaults.
        adaptive = _compute_cluster_params(len(kept_ids), size_factor=self.size_factor)
        effective_mcs = (
            self.min_cluster_size if self.min_cluster_size is not None
            else adaptive["min_cluster_size"]
        )
        effective_min_samples = (
            self.min_samples if self.min_samples is not None
            else adaptive["min_samples"]
        )
        effective_n_neighbors = (
            self.n_neighbors if self.n_neighbors is not None
            else adaptive["n_neighbors"]
        )

        # 3. Need at least min_cluster_size papers with embeddings; otherwise
        #    HDBSCAN can't form a single cluster. Treat that as "all noise".
        if len(kept_ids) < max(2, effective_mcs):
            log.warning(
                "topic_model: only %d papers with embeddings (need >= %d); "
                "skipping clustering, all papers assigned to -1",
                len(kept_ids), max(2, effective_mcs),
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
        n_neighbors = min(effective_n_neighbors, max(2, len(kept_ids) - 1))
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
            min_cluster_size=max(2, effective_mcs),
            min_samples=effective_min_samples,
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
