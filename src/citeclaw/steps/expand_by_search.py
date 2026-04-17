"""ExpandBySearch v2 — supervisor/worker flagship Phase C step.

Wires the v2 supervisor loop (:func:`citeclaw.agents.supervisor.run_supervisor`)
into the pipeline as a composable step. Unlike v1's single-agent
iterative loop, v2 decomposes the topic into sub-topics and dispatches
a sequential worker per sub-topic, each running a per-angle
dispatcher-enforced checklist.

Step flow:
  1. Idempotency fingerprint over (step name, sorted signal ids,
     agent config). Skip if already searched.
  2. Derive seed papers from the input signal (up to
     ``max_anchor_papers``) — supervisor + workers see these as
     context.
  3. Build the LLMClient via the same cascade as v1: per-step model
     override > ctx.config.search_model > ctx.config.screening_model.
  4. Render the downstream filter summary (for calibration context).
  5. Open a SearchLogger that writes to
     ``<data_dir>/search_agent_transcripts/<timestamp>/``.
  6. Run the supervisor, collect aggregate paper_ids.
  7. Screen the hydrated candidates through the shared expand helpers
     (matches ExpandByForward / ExpandByBackward). apply_local_query
     is available as a post-hydrate trim.
  8. Mark fingerprint as searched.
  9. Return StepResult with stats including the agent-level bookkeeping.

The old v1 flow (``run_iterative_search``, single-loop) is gone —
this step now *always* goes through supervisor/worker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from citeclaw.agents.filter_summary import render_filter_summary
from citeclaw.agents.search_logging import SearchLogger
from citeclaw.agents.state import AgentConfig
from citeclaw.agents.supervisor import run_supervisor
from citeclaw.clients.llm.factory import build_llm_client
from citeclaw.models import PaperRecord
from citeclaw.search.query_engine import apply_local_query
from citeclaw.steps._expand_helpers import (
    check_already_searched,
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_search")


class ExpandBySearch:
    """Supervisor/worker meta-LLM search expansion step (v2)."""

    name = "ExpandBySearch"

    def __init__(
        self,
        *,
        agent: AgentConfig,
        screener: Any = None,
        topic_description: str | None = None,
        max_anchor_papers: int = 20,
        apply_local_query_args: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.screener = screener
        self.topic_description = topic_description
        self.max_anchor_papers = max_anchor_papers
        self.apply_local_query_args = apply_local_query_args or None

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        fp = fingerprint_signal(
            self.name, signal,
            agent=asdict(self.agent),
            max_anchor_papers=self.max_anchor_papers,
            topic=self.topic_description or "",
        )
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        anchor_papers = signal[: self.max_anchor_papers]
        topic = (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )

        llm = build_llm_client(
            ctx.config,
            ctx.budget,
            model=(
                self.agent.model
                or ctx.config.search_model
                or ctx.config.screening_model
            ),
            reasoning_effort=self.agent.reasoning_effort,
            cache=getattr(ctx, "cache", None),
        )

        # Seed papers for the agents — derived from input signal. Using
        # already-hydrated PaperRecords avoids a second S2 hit.
        seed_papers = []
        for rec in anchor_papers:
            seed_papers.append({
                "paper_id": rec.paper_id,
                "title": rec.title or "",
                "abstract": rec.abstract or "",
                "year": rec.year,
                "venue": rec.venue or "",
            })

        filter_summary = render_filter_summary(self.screener)

        # Logger dir: one subdirectory per run, stamped with ISO date.
        data_dir = Path(getattr(ctx.config, "data_dir", ".") or ".")
        run_dir = data_dir / "search_agent_transcripts" / f"run_{int(time.time())}_{fp[:10]}"
        logger = SearchLogger(run_dir)
        logger.log_run_started(
            topic=topic,
            seed_count=len(seed_papers),
            filter_summary=filter_summary,
            agent_config=asdict(self.agent),
            model=(
                self.agent.model
                or ctx.config.search_model
                or ctx.config.screening_model
            ),
        )

        t_start = time.time()
        tokens_before = ctx.budget.llm_total_tokens
        s2_before = ctx.budget.s2_requests
        # Full snapshot so the cost summary can diff run-total vs
        # search-only spend per model. to_dict() is cheap — a bounded
        # dict over existing tracker state.
        budget_before_snapshot = ctx.budget.to_dict()

        try:
            sup_state, aggregate_ids = run_supervisor(
                topic_description=topic,
                filter_summary=filter_summary,
                seed_papers=seed_papers,
                llm_client=llm,
                ctx=ctx,
                agent_config=self.agent,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001 — LLM/outage safety net
            log.warning(
                "ExpandBySearch: supervisor crashed (%s); returning empty signal",
                exc,
            )
            logger.log_run_finished(
                n_papers_found=0,
                n_sub_topics=0,
                duration_s=time.time() - t_start,
                llm_tokens=ctx.budget.llm_total_tokens - tokens_before,
                s2_requests=ctx.budget.s2_requests - s2_before,
                summary=f"Supervisor crashed: {exc}",
            )
            logger.finalize()
            logger.write_cost_summary(
                ctx.budget, before_snapshot=budget_before_snapshot,
            )
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "reason": "supervisor_failed",
                    "error": str(exc)[:200],
                    "anchor_papers": len(anchor_papers),
                    "transcript_dir": str(run_dir),
                },
            )

        # Turn aggregate paper_ids into the list[dict] shape screen_expand_candidates expects.
        raw_hits = [{"paperId": pid} for pid in aggregate_ids]

        post_trim = (
            (lambda recs: apply_local_query(recs, **self.apply_local_query_args))
            if self.apply_local_query_args
            else None
        )
        screened = screen_expand_candidates(
            raw_hits=raw_hits,
            source_label="search",
            screener=self.screener,
            ctx=ctx,
            post_hydrate_fn=post_trim,
        )

        ctx.searched_signals.add(fp)

        duration = time.time() - t_start
        tokens_used = ctx.budget.llm_total_tokens - tokens_before
        s2_used = ctx.budget.s2_requests - s2_before

        done_summary = (
            (sup_state.call_log[-1] or {}).get("result", {}).get("summary", "")
            if sup_state.call_log else ""
        ) or f"{len(sup_state.sub_topic_results)} sub-topics dispatched"

        logger.log_run_finished(
            n_papers_found=len(aggregate_ids),
            n_sub_topics=len(sup_state.sub_topic_results),
            duration_s=duration,
            llm_tokens=tokens_used,
            s2_requests=s2_used,
            summary=done_summary,
        )
        logger.finalize()
        logger.write_cost_summary(
            ctx.budget, before_snapshot=budget_before_snapshot,
        )

        ctx.dashboard.note(
            f"ExpandBySearch v2 done · {len(sup_state.sub_topic_results)} sub-topics · "
            f"{len(aggregate_ids)} unique hits · "
            f"{screened.base_stats.get('accepted', 0)} accepted after screening"
        )

        return StepResult(
            signal=screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "n_sub_topics": len(sup_state.sub_topic_results),
                "n_successful_workers": sum(
                    1 for r in sup_state.sub_topic_results if r.status == "success"
                ),
                "n_failed_workers": sum(
                    1 for r in sup_state.sub_topic_results if r.status != "success"
                ),
                "aggregate_paper_ids": len(aggregate_ids),
                "supervisor_turns": sup_state.turn_index,
                "tokens_used": tokens_used,
                "s2_requests_used": s2_used,
                "transcript_dir": str(run_dir),
            },
        )
