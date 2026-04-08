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
from typing import Protocol, runtime_checkable

from citeclaw.models import PaperRecord


@dataclass
class ClusterMetadata:
    """Optional human-readable description of one cluster.

    Filled in by the naming pipeline (c-TF-IDF for keywords, then optional
    LLM call for label + summary). Empty defaults so a clusterer that doesn't
    care about naming can ship a result with metadata={cluster_id: ClusterMetadata(size=N)}.
    """
    label: str = ""
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    size: int = 0
    representative_papers: list[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    """The output of running a clusterer over a list of papers.

    ``membership`` is the only required field — every paper id maps to its
    integer cluster id. Cluster id ``-1`` is reserved for "noise" / "unassigned"
    (HDBSCAN convention; reused by every clusterer that has a notion of
    unclustered papers).
    """
    membership: dict[str, int]
    metadata: dict[int, ClusterMetadata] = field(default_factory=dict)
    algorithm: str = ""


@runtime_checkable
class Clusterer(Protocol):
    name: str

    def cluster(self, signal: list[PaperRecord], ctx) -> ClusterResult:
        """Assign each paper in ``signal`` to a cluster.

        Returns a :class:`ClusterResult` with at minimum the ``membership``
        dict populated. Implementations may also pre-fill ``metadata`` with
        per-cluster ``size`` so the naming pipeline can skip empty clusters.
        """
        ...
