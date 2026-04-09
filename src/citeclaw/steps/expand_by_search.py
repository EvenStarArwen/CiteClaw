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

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any

from citeclaw.agents.iterative_search import AgentConfig, run_iterative_search
from citeclaw.clients.llm.factory import build_llm_client
from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.search.query_engine import apply_local_query
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

    def _fingerprint(self, signal: list[PaperRecord]) -> str:
        """Stable hash over (step name, sorted signal IDs, agent config,
        topic, anchor cap). Used as the idempotency key in
        ``ctx.searched_signals`` so re-running the step on the same
        input is a true no-op."""
        signal_ids = sorted(p.paper_id for p in signal if p.paper_id)
        payload = {
            "step": self.name,
            "signal_ids": signal_ids,
            "agent": asdict(self.agent),
            "max_anchor_papers": self.max_anchor_papers,
            "topic": self.topic_description or "",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        # 1. Idempotency
        fingerprint = self._fingerprint(signal)
        if fingerprint in ctx.searched_signals:
            log.info(
                "ExpandBySearch: signal fingerprint already searched, "
                "skipping (no-op)",
            )
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "already_searched", "fingerprint": fingerprint[:12]},
            )

        # 2. Anchor papers — callers control the order via an upstream Rerank.
        anchor_papers = signal[: self.max_anchor_papers]

        # 3. Topic — explicit override > Settings.topic_description.
        topic = (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )

        # 4. Build the LLM client once and run the agent.
        # Cascade: per-step model > Settings.search_model (PC-06) > screening_model.
        llm = build_llm_client(
            ctx.config,
            ctx.budget,
            model=(
                self.agent.model
                or getattr(ctx.config, "search_model", None)
                or ctx.config.screening_model
            ),
            reasoning_effort=self.agent.reasoning_effort,
        )
        result = run_iterative_search(topic, anchor_papers, llm, ctx, self.agent)

        # 5. Hydrate the agent's raw hits into PaperRecord instances.
        candidates: list[dict[str, Any]] = []
        for hit in result.hits:
            if not isinstance(hit, dict):
                continue
            pid = hit.get("paperId")
            if isinstance(pid, str) and pid:
                candidates.append({"paper_id": pid})
        hydrated: list[PaperRecord] = (
            ctx.s2.enrich_batch(candidates) if candidates else []
        )

        # 6. Fill in abstracts so any ``title_abstract`` LLMFilter in the
        # screener has something to read.
        if hydrated:
            ctx.s2.enrich_with_abstracts(hydrated)

        # 7. Optional post-fetch trim via apply_local_query for predicates
        # the S2 API can't express (regex on title/abstract, etc.).
        if self.apply_local_query_args and hydrated:
            hydrated = apply_local_query(hydrated, **self.apply_local_query_args)

        # 8. Dedup against ctx.seen + stamp the source label.
        new_records: list[PaperRecord] = []
        for rec in hydrated:
            if not rec.paper_id:
                continue
            if rec.paper_id in ctx.seen:
                continue
            rec.source = "search"
            new_records.append(rec)
            ctx.seen.add(rec.paper_id)

        # 9. Source-less filter context — every screener atom in PC-05's
        # audit must handle this case.
        fctx = FilterContext(
            ctx=ctx, source=None, source_refs=None, source_citers=None,
        )

        # 10. Apply the screener cascade and record per-paper rejections.
        passed, rejected = apply_block(new_records, self.screener, fctx)
        record_rejections(rejected, fctx)

        # 11. Add survivors to the collection.
        for p in passed:
            p.llm_verdict = "accept"
            ctx.collection[p.paper_id] = p

        # 12. Mark the fingerprint so re-runs are no-ops.
        ctx.searched_signals.add(fingerprint)

        # 13. Return the StepResult with the agent's bookkeeping in stats.
        return StepResult(
            signal=passed,
            in_count=len(hydrated),
            stats={
                "agent_iterations": len(result.transcript),
                "agent_decision": result.final_decision,
                "raw_hits": len(result.hits),
                "hydrated": len(hydrated),
                "after_local_query": len(new_records),
                "accepted": len(passed),
                "rejected": len(rejected),
                "tokens_used": result.tokens_used,
                "s2_requests_used": result.s2_requests_used,
            },
        )
