"""ExpandBySearch — flagship Phase C step.

Wires :func:`citeclaw.agents.iterative_search.run_iterative_search`
into the pipeline as a composable step. Unlike ``ExpandForward`` /
``ExpandBackward``, this step has *no source paper*: the agent is
grounded by the input signal as anchor context, not by an upstream
citation edge. The ``FilterContext`` therefore carries
``source=None`` / ``source_refs=None`` / ``source_citers=None``, so
every filter atom in the screener must tolerate source-less mode
(audited in PC-05).

Run loop (matches the roadmap spec line by line):

  1. Compute a fingerprint over (step name, sorted signal IDs, agent
     config). If already in ``ctx.searched_signals``, return a no-op
     so a second invocation with identical inputs doesn't re-spend
     budget.
  2. ``anchor_papers = signal[:max_anchor_papers]`` — callers can put
     a ``Rerank`` (with diversity) upstream to control what gets fed
     in.
  3. ``topic = self.topic_description or ctx.config.topic_description``.
  4. Build the LLM client once and call ``run_iterative_search``.
  5. Hydrate the agent's raw hits via ``ctx.s2.enrich_batch``.
  6. Fill in abstracts via ``ctx.s2.enrich_with_abstracts``.
  7. (Optional) trim via :func:`citeclaw.search.apply_local_query` for
     post-fetch predicates that the S2 search API can't express.
  8. Dedup against ``ctx.seen``, stamp ``source="search"`` on the
     novel ones, add them to ``ctx.seen``.
  9. Build a source-less ``FilterContext``.
  10. ``apply_block`` the screener, then ``record_rejections``.
  11. Add survivors to ``ctx.collection`` with ``llm_verdict="accept"``.
  12. Mark the fingerprint in ``ctx.searched_signals``.
  13. Return ``StepResult(signal=passed, in_count=len(hydrated),
      stats={...})`` carrying the agent's bookkeeping totals so the
      shape-summary table can show what the run cost.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from citeclaw.agents.iterative_search import AgentConfig, run_iterative_search
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
    """Iterative meta-LLM search expansion step.

    Composable at the same level as ``ExpandForward`` /
    ``ExpandBackward`` — users mix all three freely in YAML pipelines.
    """

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

        # 1. Idempotency.
        fp = fingerprint_signal(
            self.name, signal,
            agent=asdict(self.agent),
            max_anchor_papers=self.max_anchor_papers,
            topic=self.topic_description or "",
        )
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        # 2. Anchor papers — callers control the order via an upstream Rerank.
        anchor_papers = signal[: self.max_anchor_papers]

        # 3. Topic — explicit override > Settings.topic_description.
        topic = (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )

        # 4. Build the LLM client once and run the agent.
        # Cascade: per-step model > Settings.search_model > screening_model.
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
        # Wrap the agent run so that LLM API outages, malformed
        # structured-output responses, schema-validation crashes etc.
        # don't kill the entire pipeline. ExpandBySemantics already does
        # this around its fetch_recommendations call; this brings
        # ExpandBySearch in line with that crash-isolation contract.
        try:
            result = run_iterative_search(
                topic, anchor_papers, llm, ctx, self.agent,
            )
        except Exception as exc:  # noqa: BLE001 — agent crash safety net
            log.warning(
                "ExpandBySearch: iterative search agent crashed (%s); "
                "returning empty signal so the pipeline can continue.",
                exc,
            )
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "reason": "agent_failed",
                    "error": str(exc)[:160],
                    "anchor_papers": len(anchor_papers),
                },
            )

        # Surface the agent's exit reason so the user knows whether the
        # loop terminated because the corpus was saturated or because
        # the iteration / token cap was hit.
        ctx.dashboard.note(
            f"search agent done · {len(result.transcript)} turns · "
            f"{len(result.hits)} unique hits · → {result.final_decision or 'unknown'}"
        )

        # 5. Shared screening pipeline. ``apply_local_query`` runs as
        # the post-hydrate trim so the screener only sees post-trim
        # candidates (matches the previous behaviour).
        post_trim = (
            (lambda recs: apply_local_query(recs, **self.apply_local_query_args))
            if self.apply_local_query_args
            else None
        )
        screened = screen_expand_candidates(
            raw_hits=result.hits,
            source_label="search",
            screener=self.screener,
            ctx=ctx,
            post_hydrate_fn=post_trim,
        )

        # 6. Mark the fingerprint so re-runs are no-ops.
        ctx.searched_signals.add(fp)

        return StepResult(
            signal=screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "agent_iterations": len(result.transcript),
                "agent_decision": result.final_decision,
                "tokens_used": result.tokens_used,
                "s2_requests_used": result.s2_requests_used,
            },
        )
