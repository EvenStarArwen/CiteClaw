"""Clusterer Protocol + ClusterResult / ClusterMetadata dataclasses.

The ``Clusterer`` Protocol is the unified contract that every clustering
algorithm in CiteClaw implements — graph-structure-based community detection
(``walktrap``, ``louvain``) and embedding-based topic modeling
(``topic_model``) all return the same shape so any downstream consumer
(Rerank diversity, GraphML export, future cluster-aware filters) can read
them uniformly.

A clusterer's job is *only* to assign papers to integer cluster ids. The
human-readable description of each cluster (label, top keywords, summary,
representative documents) is filled in by an algorithm-agnostic naming
pipeline in :mod:`citeclaw.cluster.representation` — see :class:`ClusterMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.context import Context


@dataclass
class ClusterMetadata:
    """Human-readable description of one cluster.

    Mutated post-construction by the naming pipeline in
    :mod:`citeclaw.cluster.representation`: ``size`` is set by the
    clusterer, ``keywords`` by c-TF-IDF, ``label`` and ``summary`` by an
    optional LLM call, and ``representative_papers`` by centroid distance.
    All fields default to empty so a clusterer that only cares about
    membership can ship ``metadata={cid: ClusterMetadata(size=N)}``.
    """

    label: str = ""
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    size: int = 0
    representative_papers: list[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    """Output of one clusterer over a list of papers.

    ``membership`` maps every input paper id to its integer cluster id.
    Cluster id ``-1`` is reserved for "noise" / "unassigned" (HDBSCAN
    convention; reused by every clusterer with an unclustered bucket) —
    consumers that want to filter noise should drop ids whose membership
    is ``-1`` rather than relying on absence from the dict.

    ``metadata`` keys are cluster ids and are populated lazily by the
    naming pipeline; a clusterer that doesn't pre-fill metadata still
    returns a valid result. ``algorithm`` is the registry name of the
    clusterer that produced the result (e.g. ``"walktrap"``,
    ``"louvain"``, ``"topic_model"``) so downstream consumers can tell
    them apart without reading the ``Cluster`` step config.

    ``coords_2d`` is an *optional* per-paper 2D display projection
    (``paper_id → (x, y)``). Only embedding-based clusterers fill it in
    (:class:`~citeclaw.cluster.topic_model.TopicModelClusterer` runs a
    second, 2-component UMAP purely for display); graph clusterers leave
    it empty and the artifact writer borrows the topic result's
    coordinates instead. Consumers must treat an empty dict as "this run
    has no topic map", never as "the paper sits at the origin".

    ``quality`` carries scalar diagnostics about the partition itself
    (e.g. ``{"modularity": 0.5558}``) — free-form so different
    algorithms can report what they actually measured.
    """

    membership: dict[str, int]
    metadata: dict[int, ClusterMetadata] = field(default_factory=dict)
    algorithm: str = ""
    coords_2d: dict[str, tuple[float, float]] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)


def relabel_by_size(membership: dict[str, int]) -> dict[str, int]:
    """Renumber cluster ids so ``0`` is the largest cluster, ``1`` the next, …

    Ties break on the smallest paper id in the cluster, so the mapping
    is a pure function of the input and never depends on dict iteration
    order — the property the UI's ``C##`` codes and every golden test
    rely on. ``-1`` (noise) is passed through untouched and is not
    considered for a rank.

    Only :class:`~citeclaw.cluster.leiden.LeidenClusterer` applies this;
    ``walktrap`` / ``louvain`` keep their raw igraph ids so their
    existing behaviour (and the graphml columns downstream of it) is
    bit-identical to before.
    """
    sizes: dict[int, int] = {}
    first_pid: dict[int, str] = {}
    for pid, cid in membership.items():
        if cid == -1:
            continue
        sizes[cid] = sizes.get(cid, 0) + 1
        if cid not in first_pid or pid < first_pid[cid]:
            first_pid[cid] = pid
    order = sorted(sizes, key=lambda c: (-sizes[c], first_pid[c]))
    remap = {old: new for new, old in enumerate(order)}
    return {pid: (-1 if cid == -1 else remap[cid]) for pid, cid in membership.items()}


@runtime_checkable
class Clusterer(Protocol):
    """Protocol every clustering algorithm in CiteClaw implements.

    Concrete clusterers live in sibling modules
    (:mod:`citeclaw.cluster.walktrap`, :mod:`.louvain`, :mod:`.topic_model`)
    and are registered in :mod:`citeclaw.cluster` for the ``Cluster`` step
    builder.
    """

    name: str

    def cluster(
        self, signal: list[PaperRecord], ctx: "Context",
    ) -> ClusterResult:
        """Assign each paper in ``signal`` to a cluster.

        Returns a :class:`ClusterResult` with at minimum the ``membership``
        dict populated. Implementations may also pre-fill ``metadata`` with
        per-cluster ``size`` so the naming pipeline can skip empty clusters.
        Implementations should consult ``ctx.s2`` / ``ctx.cache`` rather
        than re-fetching graph or embedding data.
        """
        ...
