"""Pipeline context — mutable state shared across steps."""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from citeclaw.budget import BudgetTracker

from citeclaw.config import Settings
from citeclaw.models import PaperRecord

from citeclaw.progress import DashboardLike, NullDashboard

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.cluster.base import ClusterResult
    from citeclaw.clients.s2 import SemanticScholarClient
    from citeclaw.event_sink import EventSink


@dataclass
class HitlGate:
    """Synchronization gate for web-mode HITL.

    The ``HumanInTheLoop`` step sets up a gate, emits an ``hitl_request``
    event via the event sink, then blocks on ``event.wait()``. The web
    backend's ``POST /api/runs/{run_id}/hitl`` handler writes user labels
    into ``labels`` and calls ``event.set()`` to unblock the pipeline
    thread.
    """

    event: threading.Event = field(default_factory=threading.Event)
    labels: dict[str, bool] = field(default_factory=dict)
    stop_requested: bool = False
    timeout_sec: float = 600.0


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

    # Paper IDs produced by ``ResolveSeeds`` (resolves title-only YAML
    # entries to S2 paper IDs, optionally adding preprint / published
    # siblings via ``external_ids``). When non-empty, ``LoadSeeds``
    # consumes this list instead of ``cfg.seed_papers``. Order matches
    # ``cfg.seed_papers`` with siblings appended after their primary.
    resolved_seed_ids: list[str] = field(default_factory=list)

    # Per-paper rejection categories — keyed by paper id so
    # ``HumanInTheLoop`` can sample papers rejected by a specific
    # filter. Populated by ``record_rejections``; may contain duplicate
    # category strings if a paper was rejected at multiple stages.
    rejection_ledger: dict[str, list[str]] = field(default_factory=dict)

    # Per-LLM-filter screening trace. ``papers_screened_by_filter`` lists
    # which paper_ids each LLM filter actually saw (regardless of
    # outcome) — needed by ``HumanInTheLoop`` so per-filter agreement
    # is computed only over papers a given filter ran on, not papers a
    # later filter never got to see. ``papers_accepted_by_filter`` lists
    # the per-filter accept set, used by HITL to sample only LLM-accepted
    # papers (rather than also pulling in hard-rule accepts from
    # YearFilter / CitationFilter / etc.). Populated by the LLMFilter
    # branch of ``filters.runner.apply_block``.
    papers_screened_by_filter: dict[str, set[str]] = field(default_factory=dict)
    papers_accepted_by_filter: dict[str, set[str]] = field(default_factory=dict)

    # Wallclock anchor (``time.monotonic()``) at the moment
    # ``run_pipeline`` started. Steps that gate on "wait N minutes since
    # pipeline start" — currently only ``HumanInTheLoop`` — read this to
    # decide whether they should sleep before sampling. Populated by
    # ``run_pipeline``; ``None`` outside of an active run.
    pipeline_started_at: float | None = None

    # Idempotency set for the ``ExpandBy*`` family. Each step adds a
    # fingerprint over (step name, signal ids, agent config) so a
    # repeat invocation with identical inputs becomes a no-op.
    searched_signals: set[str] = field(default_factory=set)

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

    # Web-mode HITL synchronization gate. When set, ``HumanInTheLoop``
    # emits an ``hitl_request`` event and blocks on ``hitl_gate.event``
    # instead of prompting via rich CLI. The web backend writes labels
    # into ``hitl_gate.labels`` and sets the event.
    hitl_gate: HitlGate | None = None

    # Event sink for streaming pipeline events to the web UI. Steps can
    # call ``ctx.event_sink.<method>`` to emit events; defaults to None
    # (steps should check before calling, or the pipeline runner sets it).
    event_sink: Any = None
