"""Pipeline context — mutable state shared across steps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from citeclaw.config import BudgetTracker, Settings
from citeclaw.models import PaperRecord

from citeclaw.progress import DashboardLike, NullDashboard

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.cluster.base import ClusterResult
    from citeclaw.clients.s2 import SemanticScholarClient


@dataclass
class Context:
    config: Settings
    s2: "SemanticScholarClient"
    cache: "Cache"
    budget: BudgetTracker

    collection: dict[str, PaperRecord] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    seed_ids: set[str] = field(default_factory=set)
    expanded_forward: set[str] = field(default_factory=set)
    expanded_backward: set[str] = field(default_factory=set)

    rejection_counts: Counter[str] = field(default_factory=Counter)

    # Per-edge metadata indexed by (src_paper_id, dst_paper_id) where src
    # is the cited paper and dst is the citing paper. Each value is
    # ``{contexts: list[str], intents: list[str], is_influential: bool}``.
    edge_meta: dict[tuple[str, str], dict] = field(default_factory=dict)

    # Alias map populated by the ``MergeDuplicates`` step. Maps a
    # non-canonical paper_id (e.g. an arXiv preprint) to its canonical
    # paper_id (e.g. the conference version). Graph construction consults
    # this to collapse duplicate edges that point at the preprint ID.
    alias_map: dict[str, str] = field(default_factory=dict)

    # Named cluster results populated by the ``Cluster`` step. Each entry
    # is a ``ClusterResult`` keyed by the step's ``store_as`` name. Any
    # downstream consumer (Rerank diversity, GraphML export, future
    # cluster-aware filters) can read this dict to look up a precomputed
    # clustering by name. Stored on the context (not the collection) so
    # the artifact survives across steps but doesn't pollute PaperRecord.
    clusters: dict[str, "ClusterResult"] = field(default_factory=dict)

    iteration: int = 1
    prior_dir: Path | None = None
    new_seed_ids: list[str] = field(default_factory=list)

    # Live terminal dashboard. ``NullDashboard`` is the default no-op so
    # steps can call ``ctx.dashboard.<anything>`` unconditionally; the
    # pipeline runner swaps in a real :class:`citeclaw.progress.Dashboard`
    # at start of run when stdout is interactive.
    dashboard: DashboardLike = field(default_factory=NullDashboard)
