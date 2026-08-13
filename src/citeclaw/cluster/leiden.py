"""LeidenClusterer — real Leiden community detection via ``leidenalg``.

Leiden (Traag, Waltman & van Eck 2019) is Louvain's successor: it adds a
refinement phase that guarantees every community is internally connected,
which is exactly the pathology Louvain is known for (badly connected —
sometimes *disconnected* — communities that still score well on
modularity). CiteClaw's UI groups papers by community and lets the user
drill into one, so a disconnected "community" is a visible product bug,
not just a metric footnote.

Why ``leidenalg`` and not the alternatives
------------------------------------------
* **``leidenalg``** (Traag's own reference implementation; the package the
  Leiden paper ships) — chosen. It is the only candidate that reproduces
  the design hand-off fixture ``web/design-reference/uploads/run37``
  *exactly*: over that run's 354-node / 1239-edge citation graph,
  ``find_partition(g, ModularityVertexPartition, seed=42, n_iterations=2)``
  returns the fixture's ``community_leiden`` column label-for-label
  (12 communities, sizes ``[57, 54, 52, 49, 37, 31, 30, 27, …]``,
  modularity 0.5558). That makes the fixture a golden test rather than a
  vague shape reference — see ``tests/test_cluster_leiden.py``.
* **igraph's built-in ``Graph.community_leiden``** — rejected as the
  primary. It is a genuine Leiden implementation and needs no new
  dependency, but on the same fixture graph it lands on a materially
  different partition (ARI 0.53 / NMI 0.74 against the fixture, and a
  much lumpier size profile: largest community 79 vs 57). It also exposes
  no RNG seed, so runs are not reproducible across igraph builds. Kept as
  the **fallback** when ``leidenalg`` is not installed.
* **``graspologic``** — rejected. Its Leiden is a Rust port behind a heavy
  transitive stack (numpy/scipy/scikit-learn/networkx/gensim/umap …) that
  would drag the optional ``topic_model`` extras into the core install,
  and it is not the reference implementation.

Licensing note: ``leidenalg`` is GPL-3.0-or-later. CiteClaw already
depends on ``python-igraph`` (GPL-2.0-or-later) in its *core* dependency
set, so this introduces no new licence class for the project.

Contract
--------
Same projection contract as :mod:`citeclaw.cluster.walktrap` /
:mod:`.louvain`: the partition runs over the whole ``ctx.collection`` so
community ids reflect the true citation neighbourhood, then the returned
``membership`` is restricted to the signal-paper subset.

One deliberate divergence from the other two graph clusterers: Leiden
ids are **renumbered by descending community size** (0 = largest) via
:func:`~citeclaw.cluster.base.relabel_by_size`, because the UI renders
them as ``C00``/``C01``/… ranked cards. Walktrap and Louvain keep their
raw igraph ids so their existing on-disk output is unchanged.
"""

from __future__ import annotations

import logging

from citeclaw.cluster.base import ClusterMetadata, ClusterResult, relabel_by_size
from citeclaw.network import build_citation_graph

log = logging.getLogger("citeclaw.cluster.leiden")

#: Default RNG seed. Pinned (rather than left to ``leidenalg``'s global
#: RNG) so two runs over the same corpus produce the same ``C##`` codes —
#: users compare community cards across pipeline iterations.
DEFAULT_SEED = 42

_OBJECTIVES = ("modularity", "cpm", "rber", "rb")


def leiden_partition(
    graph,
    *,
    objective: str = "modularity",
    resolution: float = 1.0,
    n_iterations: int = 2,
    seed: int = DEFAULT_SEED,
) -> tuple[list[int], str]:
    """Partition an igraph graph with Leiden; return ``(membership, backend)``.

    ``graph`` may be directed — it is collapsed to an undirected simple
    graph first (Leiden's objective functions are undirected). ``backend``
    is ``"leidenalg"`` on the happy path and ``"igraph"`` when
    ``leidenalg`` is missing and we fell back to
    ``Graph.community_leiden``; callers record it so the artifact's
    "how this was computed" panel never claims a provenance it doesn't
    have.

    Raises ``ValueError`` on an unknown ``objective`` — a config typo
    should fail loudly rather than silently pick modularity.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"Unknown Leiden objective {objective!r}. Available: {list(_OBJECTIVES)}"
        )
    gu = graph.as_undirected(mode="collapse") if graph.is_directed() else graph
    if gu.vcount() == 0:
        return [], "leidenalg"

    try:
        import leidenalg
    except ImportError:
        log.warning(
            "leiden: 'leidenalg' is not installed; falling back to igraph's "
            "community_leiden, which yields a different (still valid) partition "
            "and honours no RNG seed. Install it with: pip install leidenalg"
        )
        return _igraph_fallback(gu, objective=objective, resolution=resolution,
                                n_iterations=n_iterations), "igraph"

    partition_type = {
        "modularity": leidenalg.ModularityVertexPartition,
        "cpm":        leidenalg.CPMVertexPartition,
        "rber":       leidenalg.RBERVertexPartition,
        "rb":         leidenalg.RBConfigurationVertexPartition,
    }[objective]
    kwargs: dict = {"n_iterations": n_iterations, "seed": seed}
    # ModularityVertexPartition has no resolution knob (it *is* resolution 1).
    if objective != "modularity":
        kwargs["resolution_parameter"] = resolution
    part = leidenalg.find_partition(gu, partition_type, **kwargs)
    return [int(c) for c in part.membership], "leidenalg"


def _igraph_fallback(gu, *, objective: str, resolution: float, n_iterations: int) -> list[int]:
    """``Graph.community_leiden`` stand-in used when ``leidenalg`` is absent.

    igraph names the objective ``"modularity"`` or ``"CPM"`` only; the two
    RB variants map onto CPM-with-resolution, which is the closest honest
    approximation. ``n_iterations`` is passed through unchanged.
    """
    obj = "modularity" if objective == "modularity" else "CPM"
    kwargs: dict = {"objective_function": obj, "n_iterations": n_iterations}
    if obj == "CPM":
        kwargs["resolution"] = resolution
    return [int(c) for c in gu.community_leiden(**kwargs).membership]


class LeidenClusterer:
    """Cluster the citation graph with the Leiden algorithm.

    Constructor knobs mirror ``leidenalg.find_partition``:

    ``objective``
        ``"modularity"`` (default; matches the design fixture),
        ``"cpm"``, ``"rber"`` or ``"rb"``.
    ``resolution``
        Resolution parameter for every objective *except* ``modularity``
        (which is resolution-1 by construction). Higher → more, smaller
        communities.
    ``n_iterations``
        Leiden refinement passes. ``2`` is ``leidenalg``'s default and
        the fixture's setting; ``-1`` iterates to convergence.
    ``seed``
        RNG seed, defaulting to :data:`DEFAULT_SEED` so runs are
        reproducible.
    ``n_communities``
        **Advisory only**, exactly as on :class:`LouvainClusterer` —
        Leiden determines the count itself. It is consumed solely as the
        divisor of the round-robin fallback when the partition raises.
    """

    name = "leiden"

    def __init__(
        self,
        *,
        objective: str = "modularity",
        resolution: float = 1.0,
        n_iterations: int = 2,
        seed: int = DEFAULT_SEED,
        n_communities: int | None = None,
    ) -> None:
        if objective not in _OBJECTIVES:
            raise ValueError(
                f"Unknown Leiden objective {objective!r}. Available: {list(_OBJECTIVES)}"
            )
        self.objective = objective
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.seed = seed
        self.n_communities = n_communities

    def cluster(self, signal, ctx) -> ClusterResult:
        """Run Leiden over ``ctx.collection`` and project membership for ``signal``.

        Returns an empty :class:`ClusterResult` when the citation graph
        has no vertices. On success the result carries
        ``quality["modularity"]`` (computed over the *full* partition,
        before the signal restriction — that is the number the UI's
        "How this was computed" panel quotes) plus the backend and
        seed actually used, so a fallback can never masquerade as the
        reference implementation.

        When the partition raises, falls back to round-robin assignment
        over ``n_communities or 3`` buckets and logs at WARNING (audit
        "no silent failure" rule) — same shape as Walktrap / Louvain.
        """
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        if g.vcount() == 0:
            return ClusterResult(membership={}, algorithm=self.name)

        gu = g.as_undirected(mode="collapse")
        backend = "leidenalg"
        try:
            membership_list, backend = leiden_partition(
                gu,
                objective=self.objective,
                resolution=self.resolution,
                n_iterations=self.n_iterations,
                seed=self.seed,
            )
        except Exception as exc:  # noqa: BLE001
            n = self.n_communities or 3
            log.warning(
                "leiden: partition raised on graph of %d vertices (fallback n=%d); "
                "falling back to round-robin assignment: %s",
                gu.vcount(), n, exc,
            )
            membership_list = [i % n for i in range(gu.vcount())]
            backend = "round_robin"

        try:
            modularity = float(gu.modularity(membership_list))
        except Exception as exc:  # noqa: BLE001
            log.debug("leiden: modularity computation failed: %s", exc)
            modularity = float("nan")

        # Rank ids over the *full* collection partition, then restrict —
        # so C00 means "largest community in this corpus", not "largest
        # among whatever happened to be in the signal".
        full = {v["paper_id"]: int(membership_list[v.index]) for v in g.vs}
        ranked = relabel_by_size(full)
        membership = {pid: cid for pid, cid in ranked.items() if pid in sig_ids}

        sizes: dict[int, int] = {}
        for cid in membership.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        metadata = {cid: ClusterMetadata(size=n) for cid, n in sizes.items()}
        log.info(
            "leiden: %d signal papers, %d communities (backend=%s, objective=%s, "
            "seed=%s, modularity=%.4f)",
            len(membership), len(sizes), backend, self.objective, self.seed, modularity,
        )
        return ClusterResult(
            membership=membership,
            metadata=metadata,
            algorithm=self.name,
            quality={
                "modularity": modularity,
                "n_communities": float(len(sizes)),
                "seed": float(self.seed),
                "resolution": float(self.resolution),
                "backend_is_leidenalg": 1.0 if backend == "leidenalg" else 0.0,
            },
        )
