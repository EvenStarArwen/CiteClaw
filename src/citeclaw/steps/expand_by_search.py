"""ExpandBySearch step shell — no agent backend wired in.

The pipeline-side wiring (idempotency fingerprint, screener, anchor
slicing, post-agent screen-and-finalize) is intact and ready for the
next agent backend to plug into ``run``. Until then the step raises
:class:`NotImplementedError` when invoked.
"""

from __future__ import annotations

import logging
from typing import Any

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
    """Meta-LLM search expansion step (no agent backend wired in)."""

    name = "ExpandBySearch"

    def __init__(
        self,
        *,
        agent: dict[str, Any] | None = None,
        screener: Any = None,
        topic_description: str | None = None,
        max_anchor_papers: int = 20,
        apply_local_query_args: dict[str, Any] | None = None,
    ) -> None:
        self.agent = dict(agent or {})
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

        fp = self._fingerprint(signal)
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        anchor_papers = signal[: self.max_anchor_papers]
        topic = self._resolve_topic(ctx)

        # No agent backend yet. A future backend should consume
        # (topic, anchor_papers, ctx, **self.agent), produce a list of
        # S2 paper_ids, and call ``self._screen_and_finalize(...)`` to
        # return a StepResult.
        raise NotImplementedError(
            "ExpandBySearch has no agent backend. Wire one in here: feed "
            "(topic, anchor_papers, ctx) to the agent, then return "
            "self._screen_and_finalize(aggregate_ids=..., signal=signal, "
            "ctx=ctx, fp=fp)."
        )

    # ---- Scaffolding helpers for a future agent ---------------------

    def _fingerprint(self, signal: list[PaperRecord]) -> str:
        return fingerprint_signal(
            self.name, signal,
            agent=self.agent,
            max_anchor_papers=self.max_anchor_papers,
            topic=self.topic_description or "",
        )

    def _resolve_topic(self, ctx) -> str:
        return (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )

    def _screen_and_finalize(
        self,
        *,
        aggregate_ids: list[str],
        signal: list[PaperRecord],
        ctx,
        fp: str,
        extra_stats: dict[str, Any] | None = None,
    ) -> StepResult:
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
        return StepResult(
            signal=screened.passed,
            in_count=len(screened.hydrated),
            stats={**screened.base_stats, **(extra_stats or {})},
        )
