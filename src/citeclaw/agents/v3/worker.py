"""V3 worker — tutorial-style 6-phase loop with ISOLATED context.

Per iteration each phase issues a fresh LLM call (system + a
phase-specific user message) rather than accumulating one long
conversation. The heavy data dump (top-10 cited + random-10 +
topic clusters + query tree) is shown ONLY in the diagnose_plan
phase, paired with one-sentence recaps of the earlier diagnostic
answers.

The worker writes queries in natural-language (AND / OR / NOT);
the translator converts to Lucene before hitting S2, and translates
back when the query is shown in the write-next history block.
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
from citeclaw.agents.v3.query_translate import to_lucene, to_natural
from citeclaw.agents.v3.state import (
    AgentConfigV3,
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
from citeclaw.prompts.search_agent_v3 import (
    WORKER_CHECK_CLUSTERS,
    WORKER_CHECK_TOP100,
    WORKER_CHECK_TOTAL,
    WORKER_DIAGNOSE_PLAN,
    WORKER_FINAL_SUMMARY,
    WORKER_NOISE_PAPER_DIAG,
    WORKER_NOISE_TOPIC_DIAG,
    WORKER_PROPOSE_FIRST,
    WORKER_SYNTAX_ERROR,
    WORKER_SYSTEM,
    WORKER_WRITE_NEXT,
)

if TYPE_CHECKING:
    from citeclaw.agents.search_logging import SearchLogger
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.clients.s2.api import SemanticScholarClient

log = logging.getLogger("citeclaw.agents.v3.worker")


_ENRICH_CAP = 2000
_RANDOM_SAMPLE_SIZE = 10
_TOP_ABSTRACTS_SIZE = 10


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
    """Render the per-item diagnostic answers gathered during Phase 4b / 5b."""
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


def _render_history(iters: list[QueryIterationV3]) -> str:
    """Prior-iter history for the write-next prompt — includes the
    natural-language query (translated back), total_count, the query
    tree, and the iter's one-sentence diagnosis. Keeps the worker from
    regressing to queries it already tried and from re-discovering
    problems it already flagged."""
    if not iters:
        return "  (this is iteration 0 — no prior iterations)"
    blocks: list[str] = []
    for it in iters:
        q_nl = to_natural(it.query_lucene) or it.query
        problems = " | ".join(
            p for p in (
                it.reasoning_total,
                it.reasoning_clusters,
                it.reasoning_top100,
            ) if p
        ) or "(no issues recorded)"
        tree = render_query_tree(it.query_tree)
        block = (
            f"iter {it.iter_idx}:\n"
            f"  query: {q_nl}\n"
            f"  total: {it.total_count} (new +{it.diff_new}, seen {it.diff_seen})\n"
            f"  query tree:\n{tree}\n"
            f"  issues found: {problems}\n"
            f"  diagnosis: {it.diagnosis or '(no plan)'}\n"
            f"  intended change: {it.intended_change or '(n/a)'}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Diagnose-plan tool helpers
# ---------------------------------------------------------------------------


def _inspect_topic(
    iteration: QueryIterationV3,
    cluster_id: int,
    *,
    enriched_by_id: dict[str, dict],
    query_lucene: str,
) -> str:
    """Render a worker-facing report for one cluster, including:

    - cluster keywords + representative titles with abstracts
    - an IN-CLUSTER query tree that shows WHICH OR-alternatives
      (across facets) are responsible for pulling papers into this
      cluster — per-facet marginals + Cartesian-product combinations
      (see :func:`citeclaw.agents.v3.analysis.build_in_cluster_tree`).

    No S2 calls — all counts come from regex matching the
    already-fetched papers' title + abstract text.
    """
    cluster = next((c for c in iteration.clusters if c.cluster_id == cluster_id), None)
    if cluster is None:
        valid = [c.cluster_id for c in iteration.clusters]
        return f"(no cluster {cluster_id}; valid ids: {valid})"

    # Pull the actual member papers (populated by topic_model).
    cluster_papers = [
        enriched_by_id[pid] for pid in cluster.paper_ids if pid in enriched_by_id
    ]

    lines = [
        f"Cluster {cluster_id} — {cluster.count} papers.",
        f"Keywords: {', '.join(cluster.keywords[:8])}",
    ]

    # In-cluster breakdown: which OR-alternative combinations are in this cluster.
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
) -> SubTopicResultV3:
    system_prompt = WORKER_SYSTEM.format(description=spec.description)
    state = WorkerStateV3(sub_topic_id=spec.id, description=spec.description)
    enriched_pool: dict[str, dict] = {}

    for iter_idx in range(config.max_iter):
        # -- Phase 1: PROPOSE ------------------------------------------------
        if iter_idx == 0:
            user_msg = WORKER_PROPOSE_FIRST.format(description=spec.description)
        else:
            history_block = _render_history(state.iterations)
            user_msg = WORKER_WRITE_NEXT.format(
                history_block=history_block,
                diagnosis=state.iterations[-1].diagnosis or "(none)",
                intended_change=state.iterations[-1].intended_change or "(none)",
            )
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=user_msg,
            category="v3_worker_propose",
        )
        query_nl = (parsed.get("query") or "").strip()
        if not query_nl:
            log.warning(
                "V3 worker %s: iter %d propose returned no query",
                spec.id, iter_idx,
            )
            break

        # Pre-S2 syntax check loop — up to 3 rewrites if the query has
        # errors S2 would reject (short wildcard, unmatched parens /
        # quotes, no mandatory clause, etc.).
        query_lucene = to_lucene(query_nl)
        for syntax_retry in range(3):
            issues = syntax_check(query_lucene)
            logger.log_tool_call(
                scope=f"v3_worker::{spec.id}",
                turn=iter_idx,
                tool_name="syntax_check",
                args={"retry": syntax_retry, "query_nl": query_nl},
                result={
                    "n_errors": sum(1 for i in issues if i.severity == "error"),
                    "n_warnings": sum(1 for i in issues if i.severity == "warning"),
                    "codes": [i.code for i in issues],
                },
            )
            if not has_blocking_error(issues):
                break
            _, parsed_fix = _single_call(
                llm_client=llm_client,
                system=system_prompt,
                user=WORKER_SYNTAX_ERROR.format(
                    query=query_nl,
                    issues=render_issues(issues),
                ),
                category="v3_worker_syntax_fix",
            )
            new_nl = (parsed_fix.get("query") or "").strip()
            if not new_nl:
                log.warning(
                    "V3 worker %s iter %d: syntax-fix retry returned no query",
                    spec.id, iter_idx,
                )
                break
            query_nl = new_nl
            query_lucene = to_lucene(query_nl)

        logger.log_tool_call(
            scope=f"v3_worker::{spec.id}",
            turn=iter_idx,
            tool_name="propose_query",
            args={"iter": iter_idx},
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
        )

        # -- Phase 3: CHECK_TOTAL (fresh, isolated context) ----------------
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_CHECK_TOTAL.format(
                query=query_nl,
                total=total,
                diff_new=diff_new,
                diff_seen=diff_seen,
                query_tree=render_query_tree(query_tree),
            ),
            category="v3_worker_check_total",
        )
        iteration.reasoning_total = str(parsed.get("reasoning") or "").strip()

        # -- Phase 4: CHECK_CLUSTERS ---------------------------------------
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
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

        # -- Phase 4b: FORCED per-cluster noise diagnosis -------------------
        # When the agent flagged clusters as unrelated, we MUST show it the
        # in-cluster query-tree breakdown and force a per-cluster diagnosis
        # of WHICH part of the query let the cluster through. The agent does
        # NOT propose a fix yet — fixes are chosen holistically in
        # diagnose_plan after all per-item diagnoses are in.
        noise_topic_diagnoses: list[dict[str, Any]] = []
        for raw_cid in unrelated[:5]:  # cap to 5 to bound LLM calls
            try:
                cid_int = int(raw_cid)
            except (TypeError, ValueError):
                continue
            inspect_output = _inspect_topic(
                iteration, cid_int,
                enriched_by_id=enriched_pool,
                query_lucene=query_lucene,
            )
            _, parsed_diag = _single_call(
                llm_client=llm_client,
                system=system_prompt,
                user=WORKER_NOISE_TOPIC_DIAG.format(
                    cluster_id=cid_int,
                    description=spec.description,
                    inspect_output=inspect_output,
                ),
                category="v3_worker_noise_topic_diag",
            )
            reason = str(parsed_diag.get("reason") or "").strip()
            noise_topic_diagnoses.append({
                "cluster_id": cid_int,
                "reason": reason,
            })

        # -- Phase 5: CHECK_TOP100 -----------------------------------------
        titles_block = _render_top100_titles(enriched_sorted)
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
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

        # -- Phase 5b: FORCED per-paper noise diagnosis ---------------------
        noise_paper_diagnoses: list[dict[str, Any]] = []
        for raw_title in off_topic[:5]:
            title = str(raw_title or "").strip()
            if not title:
                continue
            inspect_output = _inspect_paper(title, s2_client=s2_client, iteration=iteration)
            _, parsed_diag = _single_call(
                llm_client=llm_client,
                system=system_prompt,
                user=WORKER_NOISE_PAPER_DIAG.format(
                    title=title,
                    inspect_output=inspect_output,
                ),
                category="v3_worker_noise_paper_diag",
            )
            reason = str(parsed_diag.get("reason") or "").strip()
            noise_paper_diagnoses.append({"title": title, "reason": reason})

        # -- Phase 6: DIAGNOSE + PLAN (fresh, no tool loop) ----------------
        # Agent now has: global query tree, per-cluster keywords +
        # representative titles, top-10 abstracts, 10-random abstracts,
        # PLUS per-noise-item diagnostic answers from Phase 4b / 5b.
        top10_block = _render_top10_abstracts(enriched_sorted)
        random10_block = _render_random10(enriched, seed=iter_idx)
        clusters_block = render_clusters(clusters)

        noise_diag_block = _render_noise_diagnoses(
            noise_topic_diagnoses, noise_paper_diagnoses,
        )

        diagnose_msg = WORKER_DIAGNOSE_PLAN.format(
            query=query_nl,
            ans_total=iteration.reasoning_total or "(no answer)",
            ans_clusters=iteration.reasoning_clusters or "(no answer)",
            ans_top100=iteration.reasoning_top100 or "(no answer)",
            noise_diagnoses=noise_diag_block,
            total=total,
            query_tree=render_query_tree(query_tree),
            top10_block=top10_block,
            random10_block=random10_block,
            clusters=clusters_block,
        )
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=diagnose_msg,
            category="v3_worker_diagnose_plan",
        )
        diagnosis = str(parsed.get("diagnosis") or "").strip()
        intended_change = str(parsed.get("intended_change") or "").strip()
        iteration.diagnosis = diagnosis
        iteration.intended_change = intended_change

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
                "reasoning_total": iteration.reasoning_total[:200],
                "reasoning_clusters": iteration.reasoning_clusters[:200],
                "reasoning_top100": iteration.reasoning_top100[:200],
                "diagnosis": diagnosis[:200],
                "intended_change": intended_change[:200],
            },
        )

    # -- Final summary -----------------------------------------------------
    # Return the LAST iteration's paper_ids — that is the agent's
    # committed query after refinement. Unioning across iterations
    # would permanently bake in iter-0's exploratory noise, defeating
    # the purpose of refinement.
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

    summary_text = ""
    try:
        _, parsed = _single_call(
            llm_client=llm_client,
            system=system_prompt,
            user=WORKER_FINAL_SUMMARY.format(
                n_unique=len(final_ids),
                clusters=render_clusters(clusters_final),
            ),
            category="v3_worker_final",
        )
        summary_text = str(parsed.get("summary") or "")
    except Exception:  # noqa: BLE001
        summary_text = ""

    return SubTopicResultV3(
        spec_id=spec.id,
        description=spec.description,
        status="success" if state.iterations else "failed",
        paper_ids=final_ids,
        iterations=state.iterations,
        top_titles_final=top_titles_final,
        clusters_final=clusters_final,
        failure_reason="" if state.iterations else "no iterations completed",
    )
