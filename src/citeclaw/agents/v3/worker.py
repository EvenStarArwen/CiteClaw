"""V3 worker — tutorial-style loop, now operating on a structured QueryPlan.

Per iteration each phase issues a fresh LLM call (system + a
phase-specific user message) rather than accumulating one long
conversation. The heavy data dump (top-10 cited + random-10 +
topic clusters + query tree + anchor coverage) is shown ONLY in the
diagnose_plan phase.

V4 changes:

- Worker first runs an amendment turn against the supervisor's
  facet skeleton, then produces an initial :class:`QueryPlan` in
  ``propose_first`` seeded by skeleton + anchor papers.
- Subsequent iterations pick up to two transformations from the
  closed op set in :mod:`citeclaw.agents.v3.transformations`; the
  plan is mutated in place, never retyped.
- Every iteration runs :func:`check_anchor_coverage` against the
  auto-injected anchors (plus anything the worker explicitly adds).
  Coverage feeds into the diagnose phase and can trigger
  ``satisfied=true`` early termination.
- No standalone ``check_total`` phase — total and query tree are
  context in the diagnose prompt.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, TYPE_CHECKING

from citeclaw.agents.v3.analysis import (
    build_in_cluster_tree,
    build_query_tree,
    diff_vs_prev,
    render_clusters,
    render_in_cluster_tree,
    render_query_tree,
    topic_model,
)
from citeclaw.agents.v3.anchor_coverage import (
    check_anchor_coverage,
    coverage_ratio,
    render_anchor_coverage,
    titles_for_coverage,
)
from citeclaw.agents.v3.anchor_discovery import render_anchors
from citeclaw.agents.v3.query_plan import (
    plan_from_propose_first,
    render_plan_lucene,
    render_plan_natural,
    render_plan_tree,
)
from citeclaw.agents.v3.query_translate import to_lucene
from citeclaw.agents.v3.state import (
    AgentConfigV3,
    AnchorPaper,
    QueryIterationV3,
    SubTopicResultV3,
    SubTopicSpecV3,
    WorkerStateV3,
)
from citeclaw.agents.v3.syntax_check import (
    has_blocking_error,
    render_issues,
    syntax_check,
)
from citeclaw.agents.v3.transformations import (
    apply_transformations,
    render_transformations,
)
from citeclaw.prompts.search_agent_v3 import (
    WORKER_CHECK_CLUSTERS,
    WORKER_CHECK_TOP100,
    WORKER_DIAGNOSE_PLAN,
    WORKER_FINAL_SUMMARY,
    WORKER_NOISE_PAPER_DIAG,
    WORKER_NOISE_TOPIC_DIAG,
    WORKER_PROPOSE_FIRST,
    WORKER_SELECT_TRANSFORMATIONS,
    WORKER_SYNTAX_ERROR,
    WORKER_SYSTEM,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.clients.s2.api import SemanticScholarClient

log = logging.getLogger("citeclaw.agents.v3.worker")


_ENRICH_CAP = 2000
_RANDOM_SAMPLE_SIZE = 10
_TOP_ABSTRACTS_SIZE = 10
_COVERAGE_SATISFIED_THRESHOLD = 0.80

# Each LLM call already has a per-call tenacity budget (6 attempts or
# 600s total, whichever comes first — see clients/llm/openai_client.py).
# When that budget is exhausted, _single_call returns an empty string.
# If we see 2 consecutive empty replies, the endpoint is almost
# certainly down / saturated, so abort the worker rather than keep
# burning ~10 min per dead call through the remaining phases.
_MAX_CONSECUTIVE_LLM_FAILURES = 2


class _LLMCallTracker:
    def __init__(self, threshold: int = _MAX_CONSECUTIVE_LLM_FAILURES) -> None:
        self.threshold = threshold
        self.consec_empty = 0

    def record(self, text: str) -> None:
        if text and text.strip():
            self.consec_empty = 0
        else:
            self.consec_empty += 1

    def dead(self) -> bool:
        return self.consec_empty >= self.threshold


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}
    try:
        return json.loads(s[start:end])
    except json.JSONDecodeError:
        return {}


def _single_call(
    *,
    llm_client: "LLMClient",
    system: str,
    user: str,
    category: str,
) -> tuple[str, dict]:
    """Fresh LLM call — no history accumulation. Returns (text, parsed_json)."""
    try:
        resp = llm_client.call(system, user, category=category)
    except Exception as exc:  # noqa: BLE001
        log.warning("V3 LLM call failed (%s): %s", category, exc)
        return "", {}
    text = (resp.text or "").strip()
    return text, _extract_json(text)


# ---------------------------------------------------------------------------
# S2 fetch + analyse one query
# ---------------------------------------------------------------------------


def _fetch_and_analyse(
    *,
    query_lucene: str,
    iter_idx: int,
    s2_client: "SemanticScholarClient",
    prior_iters: list[QueryIterationV3],
    config: AgentConfigV3,
) -> tuple[list[str], int, list, list[dict]]:
    """Fetch paper_ids for the query (up to max_papers_per_query),
    enrich the first _ENRICH_CAP, compute query tree and return.

    Returns (paper_ids, total_count, query_tree_nodes, enriched_dicts).
    """
    paper_ids: list[str] = []
    total = 0
    token: str | None = None
    per_call_limit = 500
    while len(paper_ids) < config.max_papers_per_query:
        try:
            resp = s2_client.search_bulk(query=query_lucene, limit=per_call_limit, token=token)
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 search_bulk failed: %s", exc)
            break
        data = resp.get("data") or []
        if token is None:
            total = int(resp.get("total") or 0)
        for row in data:
            pid = row.get("paperId")
            if pid:
                paper_ids.append(pid)
        token = resp.get("token")
        if not token or not data:
            break
    paper_ids = paper_ids[: config.max_papers_per_query]

    enrich_ids = paper_ids[:_ENRICH_CAP]
    enriched: list[dict] = []
    if enrich_ids:
        try:
            records = s2_client.enrich_batch([{"paper_id": pid} for pid in enrich_ids])
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 enrich_batch failed: %s", exc)
            records = []
        by_id = {r.paper_id: r for r in records}
        for pid in enrich_ids:
            r = by_id.get(pid)
            if r is None:
                continue
            enriched.append({
                "paperId": pid,
                "title": r.title or "",
                "abstract": r.abstract or "",
                "citationCount": r.citation_count or 0,
            })

    query_tree = build_query_tree(
        query_lucene,
        full_total=total,
        s2_client=s2_client,
        enriched_papers=enriched,
    )
    return paper_ids, total, query_tree, enriched


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _render_top10_abstracts(enriched_sorted: list[dict]) -> str:
    if not enriched_sorted:
        return "  (no enriched papers)"
    lines: list[str] = []
    for i, p in enumerate(enriched_sorted[:_TOP_ABSTRACTS_SIZE], start=1):
        cc = p.get("citationCount") or 0
        title = (p.get("title") or "").strip() or "(no title)"
        abstract = _truncate(p.get("abstract") or "(no abstract)", 400)
        lines.append(f"  {i}. [{cc}c] {title}")
        lines.append(f"     {abstract}")
    return "\n".join(lines)


def _render_random10(enriched: list[dict], seed: int) -> str:
    if not enriched:
        return "  (no enriched papers)"
    rng = random.Random(seed)
    n = len(enriched)
    k = min(_RANDOM_SAMPLE_SIZE, n)
    sample_idx = rng.sample(range(n), k)
    lines: list[str] = []
    for i, idx in enumerate(sample_idx, start=1):
        p = enriched[idx]
        cc = p.get("citationCount") or 0
        title = (p.get("title") or "").strip() or "(no title)"
        abstract = _truncate(p.get("abstract") or "(no abstract)", 400)
        lines.append(f"  {i}. [{cc}c] {title}")
        lines.append(f"     {abstract}")
    return "\n".join(lines)


def _render_top100_titles(enriched_sorted: list[dict]) -> str:
    if not enriched_sorted:
        return "  (no enriched papers)"
    lines: list[str] = []
    for i, p in enumerate(enriched_sorted[:100], start=1):
        title = (p.get("title") or "").strip() or "(no title)"
        lines.append(f"  {i}. {title}")
    return "\n".join(lines)


def _render_noise_diagnoses(
    topic_diagnoses: list[dict[str, Any]],
    paper_diagnoses: list[dict[str, Any]],
) -> str:
    if not topic_diagnoses and not paper_diagnoses:
        return "  (no cluster or paper was flagged as noisy this iteration)"
    lines: list[str] = []
    if topic_diagnoses:
        lines.append("  Flagged clusters (you answered what let each one in):")
        for td in topic_diagnoses:
            lines.append(f"    · cluster {td['cluster_id']}: {td.get('reason') or '(no answer)'}")
    if paper_diagnoses:
        lines.append("  Flagged off-topic papers (you answered what let each one through):")
        for pd in paper_diagnoses:
            title = pd.get("title") or ""
            short = title if len(title) <= 100 else title[:97] + "..."
            lines.append(f"    · {short}")
            lines.append(f"       → {pd.get('reason') or '(no answer)'}")
    return "\n".join(lines)


def _render_transformation_history(iters: list[QueryIterationV3]) -> str:
    if not iters:
        return "  (this is iteration 0 — no prior transformations)"
    blocks: list[str] = []
    for it in iters:
        cov = it.anchor_coverage or {}
        n_present = sum(1 for v in cov.values() if v == "present")
        n_total = len(cov)
        coverage_note = f"{n_present}/{n_total} anchors present" if n_total else "no coverage data"
        ops = render_transformations(it.transformations)
        blocks.append(
            f"iter {it.iter_idx}:\n"
            f"  total: {it.total_count} (new +{it.diff_new}, seen {it.diff_seen})\n"
            f"  coverage: {coverage_note}\n"
            f"  transformations:\n{ops}\n"
            f"  diagnosis: {it.diagnosis or '(no plan)'}\n"
            f"  intended change: {it.intended_change or '(n/a)'}"
        )
    return "\n\n".join(blocks)


def _inspect_topic(
    iteration: QueryIterationV3,
    cluster_id: int,
    *,
    enriched_by_id: dict[str, dict],
    query_lucene: str,
) -> str:
    cluster = next((c for c in iteration.clusters if c.cluster_id == cluster_id), None)
    if cluster is None:
        valid = [c.cluster_id for c in iteration.clusters]
        return f"(no cluster {cluster_id}; valid ids: {valid})"
    cluster_papers = [
        enriched_by_id[pid] for pid in cluster.paper_ids if pid in enriched_by_id
    ]
    lines = [
        f"Cluster {cluster_id} — {cluster.count} papers.",
        f"Keywords: {', '.join(cluster.keywords[:8])}",
    ]
    tree = build_in_cluster_tree(cluster_papers, query_lucene)
    lines.append("")
    lines.append("In-cluster query-tree (which OR alternatives brought papers in):")
    lines.append(render_in_cluster_tree(tree))
    lines.append("")
    lines.append("Representative papers:")
    for title in cluster.representative_titles[:5]:
        match = next(
            (p for p in enriched_by_id.values() if (p.get("title") or "").strip() == title.strip()),
            None,
        )
        if match:
            abs_text = _truncate(match.get("abstract") or "", 400)
            lines.append(f"  · {title}")
            lines.append(f"    {abs_text}")
        else:
            lines.append(f"  · {title}")
    return "\n".join(lines)


def _inspect_paper(
    title: str,
    *,
    s2_client: "SemanticScholarClient",
    iteration: QueryIterationV3,
) -> str:
    try:
        matched = s2_client.search_match(title)
    except Exception as exc:  # noqa: BLE001
        return f"(search_match failed: {exc})"
    if not matched:
        return f"(S2 found no paper matching title: {title!r})"
    matched_id = matched.get("paperId", "")
    matched_title = matched.get("title", "")
    abstract = _truncate(matched.get("abstract") or "", 400)
    in_results = "YES" if matched_id in set(iteration.paper_ids) else "NO"
    return (
        f"Matched paper: {matched_title}\n"
        f"Paper ID: {matched_id}\n"
        f"In current iter's fetched set: {in_results}\n"
        f"Abstract: {abstract or '(no abstract)'}"
    )


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def run_v3_worker(
    *,
    spec: SubTopicSpecV3,
    config: AgentConfigV3,
    s2_client: "SemanticScholarClient",
    llm_client: "LLMClient",
    logger: "SearchLogger",
    anchors: list[AnchorPaper] | None = None,
) -> SubTopicResultV3:
    system_prompt = WORKER_SYSTEM.format(description=spec.description)
    anchors = anchors or []
    state = WorkerStateV3(sub_topic_id=spec.id, description=spec.description)
    state.anchor_titles = titles_for_coverage(anchors)
    enriched_pool: dict[str, dict] = {}

    _tracker = _LLMCallTracker()

    def _llm(*, user: str, category: str, system: str | None = None) -> tuple[str, dict]:
        if _tracker.dead():
            return "", {}
        text, parsed = _single_call(
            llm_client=llm_client,
            system=system if system is not None else system_prompt,
            user=user,
            category=category,
        )
        _tracker.record(text)
        return text, parsed

    # -- Iteration loop ---------------------------------------------------
    satisfied_early = False
    for iter_idx in range(config.max_iter):
        if _tracker.dead():
            log.warning(
                "V3 worker %s: aborting at iter %d — %d consecutive LLM failures",
                spec.id, iter_idx, _tracker.consec_empty,
            )
            break

        # -- Phase 1: PROPOSE / TRANSFORM ------------------------------------
        if iter_idx == 0:
            _, parsed = _llm(
                user=WORKER_PROPOSE_FIRST.format(
                    description=spec.description,
                    anchors_block=render_anchors(anchors),
                ),
                category="v3_worker_propose",
            )
            if not parsed:
                log.warning("V3 worker %s iter 0: propose returned empty", spec.id)
                break
            state.plan = plan_from_propose_first(parsed)
            applied_ops: list[dict[str, Any]] = []
        else:
            # Transformations: shown coverage + noise diagnoses from prior iter
            last = state.iterations[-1]
            _, parsed = _llm(
                user=WORKER_SELECT_TRANSFORMATIONS.format(
                    plan_tree=render_plan_tree(state.plan) if state.plan else "(no plan)",
                    query=render_plan_natural(state.plan) if state.plan else "",
                    total=last.total_count,
                    diff_new=last.diff_new,
                    diff_seen=last.diff_seen,
                    anchor_coverage=render_anchor_coverage(last.anchor_coverage),
                    ans_clusters=last.reasoning_clusters or "(no answer)",
                    ans_top100=last.reasoning_top100 or "(no answer)",
                    noise_diagnoses="  (surfaced in the diagnose phase, not re-shown here)",
                    transformation_history=_render_transformation_history(state.iterations),
                    diagnosis=last.diagnosis or "(none)",
                    intended_change=last.intended_change or "(none)",
                ),
                category="v3_worker_transform",
            )
            if parsed.get("satisfied") is True:
                satisfied_early = True
                break
            raw_ops = parsed.get("transformations") or []
            raw_ops = raw_ops if isinstance(raw_ops, list) else []
            result = apply_transformations(state.plan, [o for o in raw_ops if isinstance(o, dict)])
            applied_ops = result.applied
            if result.rejected:
                log.info(
                    "V3 worker %s iter %d: %d transformation(s) rejected: %s",
                    spec.id, iter_idx, len(result.rejected),
                    [r["reason"] for r in result.rejected],
                )

        if state.plan is None or not any(f.terms for f in state.plan.facets):
            log.warning("V3 worker %s iter %d: plan has no non-empty facet", spec.id, iter_idx)
            break

        query_nl = render_plan_natural(state.plan)
        query_lucene = render_plan_lucene(state.plan)

        # Pre-S2 syntax check. We can't retry as a natural-language
        # rewrite any more (transformations own the plan), but every
        # blocking error is a bug in the renderer, not the worker —
        # surface it in the log and bail this iter rather than spin.
        issues = syntax_check(query_lucene)
        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="syntax_check",
            args={"query_nl": query_nl},
            result={
                "n_errors": sum(1 for i in issues if i.severity == "error"),
                "n_warnings": sum(1 for i in issues if i.severity == "warning"),
                "codes": [i.code for i in issues],
            },
        )
        if has_blocking_error(issues):
            log.warning(
                "V3 worker %s iter %d: plan renders to a syntactically invalid "
                "Lucene query — this indicates a renderer bug. Issues: %s",
                spec.id, iter_idx, render_issues(issues),
            )
            break

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="propose_query" if iter_idx == 0 else "apply_transformations",
            args={"iter": iter_idx, "ops": applied_ops},
            result={"query_nl": query_nl, "query_lucene": query_lucene},
        )

        # -- Phase 2: FETCH + ANALYSE (system-side, no LLM) -----------------
        paper_ids, total, query_tree, enriched = _fetch_and_analyse(
            query_lucene=query_lucene,
            iter_idx=iter_idx,
            s2_client=s2_client,
            prior_iters=state.iterations,
            config=config,
        )
        for p in enriched:
            enriched_pool.setdefault(p["paperId"], p)

        enriched_sorted = sorted(enriched, key=lambda p: -(p.get("citationCount") or 0))
        top_titles = [p["title"] for p in enriched_sorted[:100] if p.get("title")]
        clusters = topic_model(enriched, s2_client=s2_client, k=config.topic_k)

        prior_ids_set: set[str] = set()
        for it in state.iterations:
            prior_ids_set.update(it.paper_ids)
        diff_new, diff_seen = diff_vs_prev(paper_ids, prior_ids_set)

        # -- Phase 3: ANCHOR COVERAGE (system-side, no LLM) -----------------
        fetched_titles = [p.get("title") or "" for p in enriched]
        anchor_cov = check_anchor_coverage(state.anchor_titles, fetched_titles)
        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="check_anchor_coverage",
            args={"n_anchors": len(state.anchor_titles)},
            result={
                "present": sum(1 for v in anchor_cov.values() if v == "present"),
                "absent": sum(1 for v in anchor_cov.values() if v == "absent"),
                "ambiguous": sum(1 for v in anchor_cov.values() if v == "ambiguous"),
            },
        )

        iteration = QueryIterationV3(
            iter_idx=iter_idx,
            query=query_nl,
            query_lucene=query_lucene,
            total_count=total,
            fetched_count=len(paper_ids),
            paper_ids=paper_ids,
            query_tree=query_tree,
            clusters=clusters,
            top_titles_100=top_titles,
            diff_new=diff_new,
            diff_seen=diff_seen,
            transformations=applied_ops,
            anchor_coverage=anchor_cov,
        )

        # -- Phase 4: CHECK_CLUSTERS ---------------------------------------
        _, parsed = _llm(
            user=WORKER_CHECK_CLUSTERS.format(
                query=query_nl,
                description=spec.description,
                clusters=render_clusters(clusters),
            ),
            category="v3_worker_check_clusters",
        )
        ans_clusters_reasoning = str(parsed.get("reasoning") or "").strip()
        unrelated = parsed.get("unrelated_clusters") or []
        missing = parsed.get("missing_areas") or []
        extras: list[str] = []
        if unrelated:
            extras.append(f"unrelated_clusters={unrelated}")
        if missing:
            extras.append(f"missing_areas={missing}")
        iteration.reasoning_clusters = (
            ans_clusters_reasoning + ((" (" + "; ".join(extras) + ")") if extras else "")
        ).strip()

        # -- Phase 4b: forced per-cluster noise diagnosis ------------------
        noise_topic_diagnoses: list[dict[str, Any]] = []
        for raw_cid in unrelated[:5]:
            try:
                cid_int = int(raw_cid)
            except (TypeError, ValueError):
                continue
            inspect_output = _inspect_topic(
                iteration, cid_int,
                enriched_by_id=enriched_pool,
                query_lucene=query_lucene,
            )
            _, parsed_diag = _llm(
                user=WORKER_NOISE_TOPIC_DIAG.format(
                    cluster_id=cid_int,
                    description=spec.description,
                    inspect_output=inspect_output,
                ),
                category="v3_worker_noise_topic_diag",
            )
            reason = str(parsed_diag.get("reason") or "").strip()
            noise_topic_diagnoses.append({"cluster_id": cid_int, "reason": reason})

        # -- Phase 5: CHECK_TOP100 -----------------------------------------
        titles_block = _render_top100_titles(enriched_sorted)
        _, parsed = _llm(
            user=WORKER_CHECK_TOP100.format(
                query=query_nl,
                titles=titles_block,
            ),
            category="v3_worker_check_top100",
        )
        ans_top100_reasoning = str(parsed.get("reasoning") or "").strip()
        off_topic = parsed.get("off_topic_titles") or []
        if off_topic:
            ans_top100_reasoning += f" (off_topic_titles={off_topic[:5]})"
        iteration.reasoning_top100 = ans_top100_reasoning.strip()

        # -- Phase 5b: forced per-paper noise diagnosis --------------------
        noise_paper_diagnoses: list[dict[str, Any]] = []
        for raw_title in off_topic[:5]:
            title = str(raw_title or "").strip()
            if not title:
                continue
            inspect_output = _inspect_paper(title, s2_client=s2_client, iteration=iteration)
            _, parsed_diag = _llm(
                user=WORKER_NOISE_PAPER_DIAG.format(
                    title=title,
                    inspect_output=inspect_output,
                ),
                category="v3_worker_noise_paper_diag",
            )
            reason = str(parsed_diag.get("reason") or "").strip()
            noise_paper_diagnoses.append({"title": title, "reason": reason})

        # -- Phase 6: DIAGNOSE + PLAN --------------------------------------
        top10_block = _render_top10_abstracts(enriched_sorted)
        random10_block = _render_random10(enriched, seed=iter_idx)
        clusters_block = render_clusters(clusters)
        noise_diag_block = _render_noise_diagnoses(
            noise_topic_diagnoses, noise_paper_diagnoses,
        )
        diagnose_msg = WORKER_DIAGNOSE_PLAN.format(
            query=query_nl,
            ans_clusters=iteration.reasoning_clusters or "(no answer)",
            ans_top100=iteration.reasoning_top100 or "(no answer)",
            anchor_coverage=render_anchor_coverage(anchor_cov),
            noise_diagnoses=noise_diag_block,
            total=total,
            diff_new=diff_new,
            diff_seen=diff_seen,
            query_tree=render_query_tree(query_tree),
            top10_block=top10_block,
            random10_block=random10_block,
            clusters=clusters_block,
        )
        _, parsed = _llm(
            user=diagnose_msg,
            category="v3_worker_diagnose_plan",
        )
        iteration.diagnosis = str(parsed.get("diagnosis") or "").strip()
        iteration.intended_change = str(parsed.get("intended_change") or "").strip()
        coverage_ok_flag = bool(parsed.get("coverage_ok") is True)

        state.iterations.append(iteration)

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="iter_summary",
            args={"iter": iter_idx, "query_nl": query_nl, "query_lucene": query_lucene},
            result={
                "total_count": total,
                "fetched": len(paper_ids),
                "new": diff_new,
                "seen": diff_seen,
                "n_clusters": len(clusters),
                "reasoning_clusters": iteration.reasoning_clusters[:200],
                "reasoning_top100": iteration.reasoning_top100[:200],
                "diagnosis": iteration.diagnosis[:200],
                "intended_change": iteration.intended_change[:200],
                "coverage_ratio": round(coverage_ratio(anchor_cov), 3),
                "n_transformations": len(applied_ops),
            },
        )

        # Early termination: strong coverage + LLM confirms coverage_ok AND
        # no flagged noise clusters/papers.
        if (
            coverage_ok_flag
            and coverage_ratio(anchor_cov) >= _COVERAGE_SATISFIED_THRESHOLD
            and not unrelated
            and not off_topic
        ):
            satisfied_early = True
            break

    # -- Final summary -----------------------------------------------------
    last_iter = state.iterations[-1] if state.iterations else None
    final_ids: list[str] = list(last_iter.paper_ids) if last_iter else []
    final_papers = [
        enriched_pool[pid] for pid in final_ids[:_ENRICH_CAP] if pid in enriched_pool
    ]
    final_sorted = sorted(final_papers, key=lambda p: -(p.get("citationCount") or 0))
    top_titles_final = [p["title"] for p in final_sorted[:30] if p.get("title")]
    clusters_final = (
        topic_model(final_papers, s2_client=s2_client, k=config.topic_k)
        if final_papers else []
    )

    final_coverage: dict[str, str] = {}
    if final_papers and state.anchor_titles:
        final_coverage = check_anchor_coverage(
            state.anchor_titles,
            [p.get("title") or "" for p in final_papers],
        )

    try:
        _, parsed = _llm(
            user=WORKER_FINAL_SUMMARY.format(
                n_unique=len(final_ids),
                clusters=render_clusters(clusters_final),
                anchor_coverage=render_anchor_coverage(final_coverage),
            ),
            category="v3_worker_final",
        )
        _ = str(parsed.get("summary") or "")
    except Exception:  # noqa: BLE001
        pass

    status = "success" if state.iterations else "failed"
    failure = "" if state.iterations else "no iterations completed"
    return SubTopicResultV3(
        spec_id=spec.id,
        description=spec.description,
        status=status,
        paper_ids=final_ids,
        iterations=state.iterations,
        top_titles_final=top_titles_final,
        clusters_final=clusters_final,
        anchor_papers=list(anchors),
        anchor_coverage_final=final_coverage,
        failure_reason=failure if not satisfied_early else "",
    )
