"""V3 typed state — minimal, no holdover from V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AgentConfigV3:
    """V3 caller-tunable knobs.

    Worker runs max_iter queries regardless (no self-stop); each query
    fetches up to max_papers_per_query via S2 search_bulk.
    """

    max_subtopics: int = 6
    max_iter: int = 5
    max_papers_per_query: int = 10_000
    max_llm_tokens: int = 2_000_000
    max_s2_requests: int = 10_000
    model: str | None = None
    reasoning_effort: str | None = "high"
    # Topic modelling knob — k for MiniBatchKMeans. Set to None for
    # adaptive (heuristic: sqrt(n) / 2 clamped 4..10).
    topic_k: int | None = None


# ---------------------------------------------------------------------------
# Supervisor strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Facet:
    """One AND'd concept in a sub-topic's query.

    ``seed_terms`` is a minimal starter list the supervisor writes
    while deciding decomposition; the worker expands each into a full
    OR group during query construction. ``concept`` is the
    human-readable name of the dimension (e.g. "prime editing
    technique") — used for diagnostics and supervisor review, never
    sent to S2 verbatim.
    """

    id: str
    concept: str
    seed_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class FacetSkeleton:
    """Supervisor-written facet set for one sub-topic. Makes the
    anchor-shared decomposition judgement concrete: if every candidate
    sub-topic would use the same facets (synonym variation only),
    decomposition isn't justified."""

    facets: tuple[Facet, ...]


@dataclass(frozen=True)
class SubTopicSpecV3:
    """One sub-topic assigned to a worker. ``facet_skeleton`` is the
    supervisor's proposed AND'd-concept set; the worker gets one
    amendment turn before query construction (can add/remove a facet
    or reshape seed terms)."""

    id: str
    description: str
    facet_skeleton: FacetSkeleton | None = None


@dataclass(frozen=True)
class StrategyV3:
    sub_topics: tuple[SubTopicSpecV3, ...]


# ---------------------------------------------------------------------------
# Anchor papers (discovered before the worker runs — see anchor_discovery.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorPaper:
    """A canonical paper for the sub-topic, confirmed on-topic by the
    anchor-discovery agent. The worker reads these during
    ``propose_first`` to ground its OR-group expansion in the actual
    vocabulary authors use, and the system rechecks their coverage
    every iteration via :func:`check_anchor_coverage`."""

    paper_id: str
    title: str
    abstract: str = ""
    citation_count: int = 0


# ---------------------------------------------------------------------------
# Structured query plan (what transformations operate on)
# ---------------------------------------------------------------------------


TermStrictness = Literal["and_words", "proximity", "phrase"]


@dataclass
class TermSpec:
    """One entry inside an OR group. ``strictness`` picks the matching
    mode: ``and_words`` splits a multi-word term into AND'd words,
    ``proximity`` keeps them within ``slop`` positions, ``phrase`` is
    the exact contiguous quoted form. Single-word terms are always
    rendered bare regardless of strictness."""

    raw: str
    strictness: TermStrictness = "phrase"
    slop: int = 0


@dataclass
class MutableFacet:
    id: str
    concept: str
    terms: list[TermSpec] = field(default_factory=list)


@dataclass
class QueryPlan:
    """Mutable tree the transformation ops act on. Rendered to Lucene
    for S2 and to AND/OR/NOT form for worker-facing displays."""

    facets: list[MutableFacet] = field(default_factory=list)
    exclusions: list[TermSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-query iteration record
# ---------------------------------------------------------------------------


@dataclass
class QueryTreeNode:
    """One row of the query-tree breakdown.

    ``clause`` is the human-readable form (AND/OR/NOT, not the Lucene
    symbols) of the sub-clause that was measured. ``count`` is the
    paper count for that clause in isolation — S2's ``total`` from a
    count-only probe for top-level facets, or an in-memory regex-match
    count over the enriched fetched papers for OR-alternative children.

    ``children`` carry per-alternative breakdown inside an OR group:
    for ``+(A | B | "C D")`` the top node reports the facet's full
    count and each child reports how many fetched papers match just
    that one alternative.
    """

    clause: str
    count: int
    children: list["QueryTreeNode"] = field(default_factory=list)


@dataclass
class TopicCluster:
    cluster_id: int
    count: int
    keywords: list[str]  # top c-TF-IDF terms for this cluster
    representative_titles: list[str]  # top-3 titles in the cluster
    # Populated by v3.analysis.topic_model so downstream tools
    # (inspect_topic, in-cluster query tree) can grab the cluster's
    # actual member papers without re-running the clusterer.
    paper_ids: list[str] = field(default_factory=list)


@dataclass
class QueryIterationV3:
    """Everything captured for one iteration of a worker's loop."""

    iter_idx: int
    query: str  # natural-language form (AND/OR/NOT) — what the LLM wrote
    query_lucene: str  # Lucene form actually sent to S2
    total_count: int  # S2's reported total (capped at max_papers_per_query for fetch)
    fetched_count: int
    paper_ids: list[str]  # stable order = S2 relevance order
    query_tree: list[QueryTreeNode]
    clusters: list[TopicCluster]
    top_titles_100: list[str]  # top-100 by citationCount
    diff_new: int  # papers not seen in any prior iter's paper_ids (this worker)
    diff_seen: int  # papers seen in a prior iter
    # Per-phase one-sentence reasoning from this iter's diagnostics.
    # These are surfaced in subsequent iters' write-next history block
    # so the worker doesn't regress to mistakes it already diagnosed.
    reasoning_clusters: str = ""
    reasoning_top100: str = ""
    diagnosis: str = ""
    intended_change: str = ""
    # Transformations applied this iter (empty on iter 0 — the initial
    # plan came from propose_first, not from a transformation list).
    transformations: list[dict[str, Any]] = field(default_factory=list)
    # {title: "present" | "absent" | "ambiguous"} for every auto-injected
    # anchor plus any worker-proposed title checked this iter.
    anchor_coverage: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Worker result (one sub-agent output, visible to supervisor)
# ---------------------------------------------------------------------------


WorkerStatusV3 = Literal["success", "failed", "budget_exhausted"]


@dataclass
class SubTopicResultV3:
    spec_id: str
    description: str  # echoed back for supervisor view
    status: WorkerStatusV3
    paper_ids: list[str]  # union across iterations
    iterations: list[QueryIterationV3] = field(default_factory=list)
    # Top-cited titles on the final union (shown to supervisor).
    top_titles_final: list[str] = field(default_factory=list)
    # Topic clusters computed over the final union (shown to supervisor).
    clusters_final: list[TopicCluster] = field(default_factory=list)
    # Anchor papers resolved by anchor_discovery (before worker dispatch).
    anchor_papers: list[AnchorPaper] = field(default_factory=list)
    # Worker amendments to the supervisor's skeleton — visible to
    # supervisor in its next react turn so repeated big amendments
    # signal the skeleton was miscast.
    skeleton_amendments: list[dict[str, Any]] = field(default_factory=list)
    # Final anchor coverage ({title: present/absent/ambiguous}).
    anchor_coverage_final: dict[str, str] = field(default_factory=dict)
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Worker live state
# ---------------------------------------------------------------------------


@dataclass
class WorkerStateV3:
    sub_topic_id: str
    description: str
    iterations: list[QueryIterationV3] = field(default_factory=list)
    # Post-amendment skeleton + live query plan (filled on iter 0,
    # mutated in place by §3 transformations on subsequent iters).
    skeleton: FacetSkeleton | None = None
    plan: QueryPlan | None = None
    # Anchor titles checked every iteration by check_anchor_coverage;
    # sourced from pre-worker anchor_discovery plus anything the worker
    # explicitly adds later.
    anchor_titles: list[str] = field(default_factory=list)
    skeleton_amendments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def aggregate_paper_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for it in self.iterations:
            for pid in it.paper_ids:
                if pid not in seen:
                    seen.add(pid)
                    out.append(pid)
        return out


# ---------------------------------------------------------------------------
# Supervisor live state
# ---------------------------------------------------------------------------


@dataclass
class SupervisorStateV3:
    strategy: StrategyV3 | None = None
    sub_topic_results: list[SubTopicResultV3] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0

    def aggregate_paper_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for r in self.sub_topic_results:
            for pid in r.paper_ids:
                if pid not in seen:
                    seen.add(pid)
                    out.append(pid)
        return out
