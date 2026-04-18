"""ExpandBySearchV3 pipeline step — tutorial-style worker + clean supervisor.

Thin wrapper that:
1. Builds the LLM client via the same cascade as V2.
2. Resolves the S2 client from ``ctx``.
3. Runs ``run_v3_supervisor`` and collects aggregate paper_ids.
4. Returns the union as a bare list of ``{paperId: ...}`` dicts so
   downstream ExpandBy* / finalize steps can hydrate on demand.

Intentionally does NOT call the downstream screener — the V3 testing
loop only exercises the multi-agent query-design module, per the
design brief. If a caller wants filter pipelines + citation network
on top, they can swap back to ``ExpandBySearch`` (V2) or add a
post-processing step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from citeclaw.agents.search_logging import SearchLogger
from citeclaw.agents.v3.state import AgentConfigV3
from citeclaw.agents.v3.supervisor import run_v3_supervisor
from citeclaw.clients.llm.factory import build_llm_client
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_search_v3")


class ExpandBySearchV3:
    """Tutorial-style V3 search expansion.

    YAML form::

        - step: ExpandBySearchV3
          agent:
            max_subtopics: 6
            max_iter: 5
            max_papers_per_query: 10000
            reasoning_effort: "high"
    """

    name = "ExpandBySearchV3"

    def __init__(
        self,
        *,
        agent: AgentConfigV3 | dict | None = None,
        topic_description: str | None = None,
        screener: Any = None,  # accepted for YAML compat but unused
        supervisor_max_turns: int = 20,
    ) -> None:
        if agent is None:
            self.agent = AgentConfigV3()
        elif isinstance(agent, dict):
            known = {k: v for k, v in agent.items() if k in AgentConfigV3.__dataclass_fields__}
            self.agent = AgentConfigV3(**known)
        else:
            self.agent = agent
        self.topic_description = topic_description
        self.supervisor_max_turns = supervisor_max_turns

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        topic = (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )
        if not topic.strip():
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no topic_description"},
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
        s2_client = ctx.s2

        data_dir = Path(getattr(ctx.config, "data_dir", ".") or ".")
        run_dir = data_dir / "search_agent_transcripts" / f"v3_run_{int(time.time())}"
        logger = SearchLogger(run_dir)
        logger.log_run_started(
            topic=topic,
            seed_count=0,
            filter_summary="(V3: workers do not see filters)",
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

        try:
            sup_state, aggregate_ids = run_v3_supervisor(
                topic_description=topic,
                config=self.agent,
                s2_client=s2_client,
                llm_client=llm,
                logger=logger,
                supervisor_max_turns=self.supervisor_max_turns,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("V3 supervisor crashed: %s", exc)
            logger.log_run_finished(
                n_papers_found=0,
                n_sub_topics=0,
                duration_s=time.time() - t_start,
                llm_tokens=ctx.budget.llm_total_tokens - tokens_before,
                s2_requests=ctx.budget.s2_requests - s2_before,
                summary=f"Supervisor crashed: {exc}",
            )
            logger.finalize()
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "reason": "v3_supervisor_failed",
                    "error": str(exc)[:200],
                    "transcript_dir": str(run_dir),
                },
            )

        duration = time.time() - t_start
        tokens_used = ctx.budget.llm_total_tokens - tokens_before
        s2_used = ctx.budget.s2_requests - s2_before

        logger.log_run_finished(
            n_papers_found=len(aggregate_ids),
            n_sub_topics=len(sup_state.sub_topic_results),
            duration_s=duration,
            llm_tokens=tokens_used,
            s2_requests=s2_used,
            summary=f"{len(sup_state.sub_topic_results)} sub-topics dispatched",
        )
        logger.finalize()

        # V3 is a test-loop for the query-design module — we don't
        # hydrate records here. Aggregate ids live on the transcript
        # and in step stats; downstream steps (if any) can re-fetch.
        out_records: list[PaperRecord] = []

        stats = {
            "n_sub_topics": len(sup_state.sub_topic_results),
            "n_aggregate_papers": len(aggregate_ids),
            "duration_s": round(duration, 2),
            "tokens_used": tokens_used,
            "s2_requests": s2_used,
            "transcript_dir": str(run_dir),
        }
        return StepResult(
            signal=out_records,
            in_count=len(signal),
            stats=stats,
        )
