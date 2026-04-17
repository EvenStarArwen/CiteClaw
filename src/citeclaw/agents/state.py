"""Typed state + hand-off dataclasses for the v2 ExpandBySearch agents.

These classes are pure data carriers — no behaviour, no IO. They are
the skeleton that the supervisor / worker loops fill in as a run
progresses, and they are what ``SubTopicResult`` captures for
postmortem analysis (see ``QueryAngleResult`` for per-angle detail).

Key design points:

- An **angle** is uniquely identified by its
  :func:`query_fingerprint` over ``(query, filters)``. The dispatcher
  uses the fingerprint for object-consistency enforcement (a
  ``fetch_results`` call is only accepted if its recomputed
  fingerprint matches one already registered by
  ``check_query_size``). This is the v2 replacement for the v1
  "fetch_results must follow check_query_size within the last 2
  turns" temporal rule, which had both false-accept and false-reject
  failure modes.

- Per-angle checklist enforcement lives on :class:`AngleState`. The
  worker dispatcher flips the ``checked_*`` flags as tools run against
  that angle's ``df_id``; both ``done()`` and the angle-transition
  hook read them.

- :class:`WorkerState` owns the union of all angles plus the
  worker-cumulative paper id set. The cumulative set is what
  ``contains(paper_id)`` queries — verification is a worker-level
  invariant, not an angle-level one.

- :class:`SubTopicResult` is the only thing the supervisor ever sees
  from a worker. The full tool-call transcript is archived to disk
  (see ``search_logging``), not consumed by the supervisor's prompt.
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
    for the same reason. Used by the dispatcher to enforce object
    consistency across :func:`check_query_size` → :func:`fetch_results`
    without a temporal window.
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
    The worker's verification step (Step 6) uses them to check whether
    its cumulative fetch covered a few known-relevant papers; a miss
    triggers a ``diagnose_miss`` reasoning step, NOT a "keep refining
    until this paper appears" loop. Rationale: forcing a reference
    paper in via OR-clause gymnastics overfits to a tiny test set and
    degrades precision across the full topic.
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
# Worker-side state: angles + cumulative set
# ---------------------------------------------------------------------------


@dataclass
class AngleState:
    """Per-angle tracking for dispatcher-level checklist enforcement.

    An angle is uniquely identified by its :func:`query_fingerprint`
    over ``(query, filters)``. The dispatcher maintains one
    ``AngleState`` per fingerprint in
    ``WorkerState.angles: dict[fingerprint, AngleState]``. The
    ``active_fingerprint`` of the worker is the fingerprint of the
    most recent ``check_query_size`` call — subsequent
    ``fetch_results`` / inspection tools are attributed to that
    fingerprint.

    Calling ``check_query_size`` with a **new** fingerprint is the
    "angle transition" signal; the dispatcher only accepts it if the
    outgoing angle's inspection checklist is complete (or the
    outgoing angle never ran ``fetch_results`` — a worker may open an
    angle, reconsider, and not fetch).
    """

    fingerprint: str
    query: str
    filters: dict[str, Any]
    df_id: str | None = None
    n_fetched: int | None = None
    total_in_corpus: int | None = None
    checked_top_cited: bool = False
    checked_random: bool = False
    checked_years: bool = False
    checked_topic_model: bool = False
    refinement_count: int = 0
    inspection_notes: str = ""

    @property
    def requires_topic_model(self) -> bool:
        return (self.n_fetched or 0) >= 500

    def is_checklist_complete(self) -> bool:
        """True iff this angle can be closed without blocking done() / transition.

        An angle that never ran ``fetch_results`` (``df_id is None``)
        is trivially complete — the worker opened it and walked away.
        An angle with a DataFrame must have top_cited + random +
        years checked, and topic_model iff n_fetched >= 500.
        """
        if self.df_id is None:
            return True
        if not (self.checked_top_cited and self.checked_random and self.checked_years):
            return False
        if self.requires_topic_model and not self.checked_topic_model:
            return False
        return True


@dataclass
class WorkerState:
    """Live worker state during one sub-topic run.

    Holds the angle registry (fingerprint -> AngleState), the active
    angle pointer, the worker-cumulative paper id set (what
    ``contains(paper_id)`` queries), the verification cycle log, and
    call-log metadata. Passed to every tool handler via the
    dispatcher.

    ``structural_priors`` is stored here so tool handlers can merge
    the run-wide S2 filters into every check_query_size / fetch_results
    call without the worker having to thread them through its JSON
    tool_args.
    """

    sub_topic_id: str
    structural_priors: "StructuralPriors | None" = None
    angles: dict[str, AngleState] = field(default_factory=dict)
    active_fingerprint: str | None = None
    cumulative_paper_ids: set[str] = field(default_factory=set)
    verification_misses: list[str] = field(default_factory=list)
    miss_diagnoses: list[dict[str, Any]] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0

    @property
    def active_angle(self) -> AngleState | None:
        if self.active_fingerprint is None:
            return None
        return self.angles.get(self.active_fingerprint)


# ---------------------------------------------------------------------------
# Per-angle postmortem record (lives on SubTopicResult)
# ---------------------------------------------------------------------------


@dataclass
class QueryAngleResult:
    """Per-angle outcome — read offline for postmortem + angle-count tuning."""

    query: str
    filters: dict[str, Any]
    fingerprint: str
    n_fetched: int
    total_in_corpus: int
    papers_added_to_cumulative: int
    refinement_count: int
    topic_model_ran: bool
    inspection_notes: str


# ---------------------------------------------------------------------------
# Result returned by a worker to the supervisor
# ---------------------------------------------------------------------------


CoverageAssessment = Literal["comprehensive", "acceptable", "limited"]
WorkerStatus = Literal["success", "failed", "budget_exhausted"]


@dataclass
class SubTopicResult:
    """Final report from one sub-topic worker.

    The supervisor sees this (not the transcript). ``paper_ids`` is
    the union of every angle's ``cumulative_paper_ids`` at done-time,
    deduplicated. ``query_angles`` replaces the v1 ``final_query:
    str`` singular field — with multi-angle workers, one entry per
    angle in execution order.
    """

    spec_id: str
    status: WorkerStatus
    paper_ids: list[str]
    coverage_assessment: CoverageAssessment | None
    summary: str
    turns_used: int
    query_angles: list[QueryAngleResult]
    failure_reason: str = ""
    # True when the worker hit its turn budget and was rescued by the
    # penultimate-turn auto-closer rather than reaching ``done()`` on
    # its own. Coverage is implicitly ``limited`` and the verification
    # cycle may have been synthesised. Supervisors should treat
    # ``auto_closed=True`` workers with lower confidence when deciding
    # retries or coverage claims — the paper_ids are real but the
    # agent's own coverage judgement was never exercised.
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

    Hard caps reflect the settled parameters table in the v2 design
    doc. ``max_angles_per_worker`` and ``max_refinement_per_angle`` are
    dispatcher-enforced; the others are soft guidance / budgets.
    """

    # Worker loop
    worker_max_turns: int = 15
    max_angles_per_worker: int = 4
    max_refinement_per_angle: int = 1

    # Supervisor loop
    supervisor_max_turns: int = 20
    supervisor_retries_per_failed_worker: int = 1

    # Search fan-out — these bound the cost of ONE fetch_results call.
    # Each fetch issues two S2 search_bulk calls (top_cited + paperId),
    # so with 500 each and ~4 angles per worker and ~6 workers per run
    # we get at most ~24,000 search-result rows over the wire per run
    # (plus batch enrich for the unique subset). Caller may tune down
    # for tight budgets.
    fetch_results_limit_per_strategy: int = 500

    # Hard ceiling on ``total`` in the S2 corpus for a query to be
    # fetchable. Queries matching more than this are refused with a
    # teaching hint pointing at structural filters. Rationale: our
    # fetch only samples top_cited + paperId-ordered up to
    # ``fetch_results_limit_per_strategy`` each, so a query matching
    # 100K+ papers would be poorly represented and the result set
    # would be a coincidental slice of the long tail. Better to force
    # the agent to narrow first. 50K is a pragmatic floor above which
    # coverage becomes statistically unreliable for this kind of
    # sampled fetch — tuned from empirical runs rather than theory.
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
