"""ExpandBySemantics — semantic kNN expansion via S2 Recommendations API.

The semantics step skips both the LLM and the iterative agent: instead
of designing queries, it asks Semantic Scholar's
``POST /recommendations/v1/papers`` endpoint to do a SPECTER2 nearest-
neighbour search over its full corpus, anchored on the input signal.
That keeps the step pure-S2 (zero LLM tokens, zero local embedding
infrastructure) and lets users compose semantic expansion alongside
the LLM-driven ``ExpandBySearch`` and the citation-graph
``ExpandForward`` / ``ExpandBackward`` in one pipeline.

Run loop (matches the roadmap spec):

  1. Compute a fingerprint over (step name, sorted signal IDs, anchor
     cap, recommendation limit, use_rejected_as_negatives flag). If
     already in ``ctx.searched_signals``, no-op.
  2. ``anchor_ids = [p.paper_id for p in signal[:max_anchor_papers]]``.
  3. ``negative_ids = list(ctx.rejected)[:50]`` if
     ``use_rejected_as_negatives``, else ``None``. The 50-cap matches
     the spec; if a real run accumulates more rejections than that,
     the trim picks an arbitrary set deterministically per Python's
     set ordering — good enough as a hint signal.
  4. ``raw = ctx.s2.fetch_recommendations(anchor_ids,
     negative_ids=..., limit=self.limit)``.
  5. Hydrate ``raw`` via ``ctx.s2.enrich_batch``.
  6. Fill abstracts via ``ctx.s2.enrich_with_abstracts``.
  7. Dedup against ``ctx.seen``, stamp ``source="semantic"`` on the
     novel ones, add them to ``ctx.seen``.
  8. Build a source-less ``FilterContext``.
  9. ``apply_block`` the screener, then ``record_rejections``.
  10. Add survivors to ``ctx.collection`` with ``llm_verdict="accept"``.
  11. Mark the fingerprint in ``ctx.searched_signals``.
  12. Return a ``StepResult`` whose stats carry the per-run totals so
     the shape-summary table can show the semantics step's footprint.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.models import PaperRecord
from citeclaw.steps._expand_helpers import (
    check_already_searched,
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_semantics")


class ExpandBySemantics:
    """Semantic kNN expansion via S2 Recommendations API.

    Composable at the same level as ``ExpandForward`` /
    ``ExpandBackward`` / ``ExpandBySearch``. Like ``ExpandBySearch``
    this step has no source paper, so its screener cascade must
    tolerate ``fctx.source=None`` (audited in PC-05).
    """

    name = "ExpandBySemantics"

    def __init__(
        self,
        *,
        screener: Any = None,
        max_anchor_papers: int = 10,
        limit: int = 100,
        use_rejected_as_negatives: bool = False,
    ) -> None:
        self.screener = screener
        self.max_anchor_papers = max_anchor_papers
        self.limit = limit
        self.use_rejected_as_negatives = use_rejected_as_negatives

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        # 1. Idempotency.
        fp = fingerprint_signal(
            self.name, signal,
            max_anchor_papers=self.max_anchor_papers,
            limit=self.limit,
            use_rejected_as_negatives=self.use_rejected_as_negatives,
        )
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        # 2. Anchors — first N from the (caller-reranked) signal.
        anchor_ids = [
            p.paper_id for p in signal[: self.max_anchor_papers] if p.paper_id
        ]
        if not anchor_ids:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no_anchors"},
            )

        # 3. Optional negative anchors from prior rejections.
        negative_ids: list[str] | None = None
        if self.use_rejected_as_negatives and ctx.rejected:
            negative_ids = list(ctx.rejected)[:50]

        # 4. Retriever — single S2 call to the Recommendations API.
        try:
            raw = ctx.s2.fetch_recommendations(
                anchor_ids,
                negative_ids=negative_ids,
                limit=self.limit,
            )
        except Exception as exc:  # noqa: BLE001 — keep the pipeline alive
            log.warning("ExpandBySemantics: fetch_recommendations failed: %s", exc)
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "fetch_failed", "error": str(exc)[:120]},
            )

        # 5. Shared screening pipeline (hydrate → enrich → dedup →
        # screen → commit). See _expand_helpers.screen_expand_candidates.
        screened = screen_expand_candidates(
            raw_hits=raw,
            source_label="semantic",
            screener=self.screener,
            ctx=ctx,
        )

        # 6. Mark fingerprint so re-runs are no-ops.
        ctx.searched_signals.add(fp)

        return StepResult(
            signal=screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "anchor_count": len(anchor_ids),
                "negative_count": len(negative_ids) if negative_ids else 0,
            },
        )
