"""Typed state + hand-off dataclasses for the v2 ExpandBySearch agents.

These classes are pure data carriers — no behaviour, no IO. They are
the skeleton that the supervisor / worker loops fill in as a run
progresses, and they are what ``SubTopicResult`` captures for
postmortem analysis (see ``QueryResult`` for per-query detail).

Post-refactor notes (simplified tool surface):

- **No more "angle"**. The v1 AngleState + per-angle checklist
  (checked_top_cited / checked_random / checked_years /
  checked_topic_model / refinement_count) was scaffolding for tools
  the LLM was forced to orchestrate. Inspection is now automatic
  inside ``fetch_results`` — the worker just decides *which* query
  to run, not how to unpack it. What remains of the old abstraction:

  - ``query_fingerprint(query, filters)`` still uniquely IDs a
    (query, filters) pair. Used to dedupe DataFrames.
  - :class:`QueryMeta` holds the minimal per-query record
    (fingerprint, query, filters, total, df_id, n_fetched). No
    checklist flags.

- ``WorkerState.queries: dict[fingerprint, QueryMeta]`` is the
  dedup cache. ``active_fingerprint`` is just the most recently
  size-checked query — no transition semantics.

- :class:`SubTopicResult.query_results: list[QueryResult]` — one
  entry per distinct query the worker ran.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def query_fingerprint(query: str, filters: dict[str, Any] | None) -> str:
    """Stable sha256 fingerprint over ``(query, filters)``.

    Normalises filter dict keys by sorting so callers that happen to
    build the same filter dict in different key orders still hash the
    same. Filters are serialised through ``json.dumps(sort_keys=True)``
    for the same reason. Used by the dispatcher to dedupe DataFrames
    (two identical check_query_size calls resolve to the same
    fingerprint; ``fetch_results`` reuses an existing df).
    """
    payload = {"query": query, "filters": filters or {}}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Structural priors (supervisor -> every worker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralPriors:
    """S2-filter priors that every worker's queries share.

    Only priors that map DIRECTLY to S2 filter parameters belong here.
    Keyword-shaped priors (``required_keywords`` / ``excluded_keywords``)
    were removed in the review: they were prose-level hints that the
    code never actually applied, and as hard filters they become
    false-negative traps for papers that use canonical-adjacent
    terminology (mechanistic analysis, saliency, attribution, etc. vs
    "interpretability"). Let the worker design queries freely; the
    downstream LLM screener handles topic fit.

    Each remaining prior is advisory — a too-tight prior rejects
    in-topic papers. Prefer fewer strong priors to many weak ones.
    """

    year_min: int | None = None
    year_max: int | None = None
    venue_filters: tuple[str, ...] = field(default_factory=tuple)
    fields_of_study: tuple[str, ...] = field(default_factory=tuple)

    def to_s2_filters(self) -> dict[str, Any]:
        """Convert to the dict shape s2.search_bulk expects.

        Emits only S2-recognised keys (``year``, ``fieldsOfStudy``,
        ``venue``). Callers — the worker's ``check_query_size`` and
        ``fetch_results`` — merge this dict into any per-call filters.
        """
        out: dict[str, Any] = {}
        if self.year_min is not None or self.year_max is not None:
            lo = self.year_min if self.year_min is not None else ""
            hi = self.year_max if self.year_max is not None else ""
            out["year"] = f"{lo}-{hi}"
        if self.fields_of_study:
            out["fieldsOfStudy"] = ",".join(self.fields_of_study)
        if self.venue_filters:
            out["venue"] = ",".join(self.venue_filters)
        return out


# ---------------------------------------------------------------------------
# Supervisor strategy: decomposition into sub-topics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubTopicSpec:
    """One sub-topic the supervisor dispatches to a worker.

    ``reference_papers`` are **diagnostic anchors**, not test targets.
    The worker's auto-verification step (inside ``fetch_results``) uses
    them to check whether its cumulative fetch covered a few
    known-relevant papers; a miss triggers a ``diagnose_miss``
    reasoning step, NOT a "keep refining until this paper appears"
    loop. Rationale: forcing a reference paper in via OR-clause
    gymnastics overfits to a tiny test set and degrades precision
    across the full topic.
    """

    id: str
    description: str
    initial_query_sketch: str
    reference_papers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SearchStrategy:
    """Supervisor's plan for one ExpandBySearch run."""

    structural_priors: StructuralPriors
    sub_topics: tuple[SubTopicSpec, ...]


# ---------------------------------------------------------------------------
# Worker-side state: query cache + cumulative set
# ---------------------------------------------------------------------------


@dataclass
class QueryMeta:
    """Per-query tracking record. Replaces v1 AngleState.

    The fingerprint uniquely IDs the (query, filters) pair. df_id is
    set once ``fetch_results`` runs on this query. No checklist flags
    — inspection is now automatic inside ``fetch_results``.
    """

    fingerprint: str
    query: str
    filters: dict[str, Any]
    total_in_corpus: int | None = None
    df_id: str | None = None
    n_fetched: int | None = None


@dataclass
class WorkerState:
    """Live worker state during one sub-topic run.

    Holds the query registry (fingerprint -> QueryMeta) — acting as a
    DataFrame cache that prevents duplicate fetches of the same
    (query, filters) pair — the worker-cumulative paper id set,
    pending verification misses detected by the auto-verifier, and
    call-log metadata.

    ``structural_priors`` is stored here so tool handlers can merge
    the run-wide S2 filters into every check_query_size / fetch_results
    call without the worker having to thread them through its JSON
    tool_args.
    """

    sub_topic_id: str
    structural_priors: "StructuralPriors | None" = None
    # Reference-paper titles from the sub_topic spec, stored here so
    # the auto-verifier inside ``fetch_results`` can resolve them
    # against each new DataFrame without threading the spec through
    # every tool handler.
    reference_papers: tuple[str, ...] = field(default_factory=tuple)
    queries: dict[str, QueryMeta] = field(default_factory=dict)
    active_fingerprint: str | None = None
    cumulative_paper_ids: set[str] = field(default_factory=set)
    # Titles the auto-verifier flagged as "matched-but-not-in-cumulative"
    # — i.e. resolved by search_match to a real paper_id but that
    # paper_id is not in our fetched set. Each one must be explained
    # via diagnose_miss before done() is accepted.
    pending_miss_titles: list[str] = field(default_factory=list)
    miss_diagnoses: list[dict[str, Any]] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0

    @property
    def active_query(self) -> QueryMeta | None:
        if self.active_fingerprint is None:
            return None
        return self.queries.get(self.active_fingerprint)


# ---------------------------------------------------------------------------
# Per-query postmortem record (lives on SubTopicResult)
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Per-query outcome — read offline for postmortem."""

    query: str
    filters: dict[str, Any]
    fingerprint: str
    n_fetched: int
    total_in_corpus: int
    papers_added_to_cumulative: int


# ---------------------------------------------------------------------------
# Result returned by a worker to the supervisor
# ---------------------------------------------------------------------------


CoverageAssessment = Literal["comprehensive", "acceptable", "limited"]
WorkerStatus = Literal["success", "failed", "budget_exhausted"]


@dataclass
class SubTopicResult:
    """Final report from one sub-topic worker.

    The supervisor sees this (not the transcript). ``paper_ids`` is
    the worker's cumulative fetch at done-time, deduped. ``query_results``
    has one entry per distinct query the worker ran.
    """

    spec_id: str
    status: WorkerStatus
    paper_ids: list[str]
    coverage_assessment: CoverageAssessment | None
    summary: str
    turns_used: int
    query_results: list[QueryResult]
    failure_reason: str = ""
    # True when the worker hit its turn budget and was rescued by the
    # penultimate-turn auto-closer rather than reaching ``done()`` on
    # its own. Coverage is implicitly ``limited`` and the verification
    # misses may not have been diagnosed. Supervisors should treat
    # ``auto_closed=True`` workers with lower confidence.
    auto_closed: bool = False


# ---------------------------------------------------------------------------
# Supervisor-side state across a run
# ---------------------------------------------------------------------------


@dataclass
class SupervisorState:
    """Live state for the supervisor across one ExpandBySearch run."""

    strategy: SearchStrategy | None = None
    sub_topic_results: list[SubTopicResult] = field(default_factory=list)
    worker_failures: dict[str, int] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0

    def record_result(self, result: SubTopicResult) -> None:
        self.sub_topic_results.append(result)

    def aggregate_paper_ids(self) -> list[str]:
        """Dedup union of every sub-topic's paper_ids, stable order."""
        seen: set[str] = set()
        out: list[str] = []
        for r in self.sub_topic_results:
            for pid in r.paper_ids:
                if pid not in seen:
                    seen.add(pid)
                    out.append(pid)
        return out


# ---------------------------------------------------------------------------
# Run-level config (user-tunable knobs)
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Caller-tunable knobs for one ExpandBySearch v2 run.

    Post-refactor: the per-angle checklist caps
    (``max_angles_per_worker``, ``max_refinement_per_angle``) are
    gone — inspection is automatic so those were scaffolding for a
    problem the worker no longer orchestrates. Replaced by a flat
    ``max_queries_per_worker`` (dedup cap on distinct fingerprints)
    and the worker_max_turns soft ceiling.
    """

    # Worker loop
    worker_max_turns: int = 15
    # Cap on distinct (query, filters) pairs a single worker may
    # open. Dedup applies — re-running the same size-check does NOT
    # consume a slot. This replaces the old angle cap; the smaller
    # dispatcher means each "query" does more work (auto-inspection
    # + auto-verification inside fetch_results), so 4 is enough.
    max_queries_per_worker: int = 4

    # Supervisor loop
    supervisor_max_turns: int = 20
    supervisor_retries_per_failed_worker: int = 1

    # Search fan-out — bounds the cost of ONE fetch_results call.
    # Each fetch issues two S2 search_bulk calls (top_cited + paperId),
    # so with 500 each and ~4 queries per worker and ~6 workers per run
    # we get at most ~24,000 search-result rows over the wire per run.
    fetch_results_limit_per_strategy: int = 500

    # Per-strategy sample size surfaced in fetch_results' inspection
    # digest (separately for top_cited and random). Bumped from 10 →
    # 100 in the refactor: 10 was too few to judge topic drift or
    # diversity; 100 costs ~1.5K tokens per strategy (~3K total for
    # both) which is cheap insurance on 2-3 fetches per worker.
    inspection_sample_size: int = 100

    # Hard ceiling on ``total`` in the S2 corpus for a query to be
    # fetchable. See docstring in search_tools.py for rationale — 50K
    # is the point above which sampled fetch stops being statistically
    # reliable.
    fetch_total_cap: int = 50_000

    # Budget (delta from start-of-run)
    max_llm_tokens: int = 2_000_000
    max_s2_requests: int = 10_000

    # Model override (falls back through ctx.config.search_model ->
    # ctx.config.screening_model — same cascade as v1).
    model: str | None = None
    reasoning_effort: str | None = "high"

    # Seed sharing (supervisor -> worker). True shows the raw seed
    # papers to workers; False hides them (calibration still comes
    # through downstream_filters_summary + topic_description).
    share_seeds_with_agents: bool = True
