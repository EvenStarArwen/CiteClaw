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

import hashlib
import json
import logging
from typing import Any

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
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

    def _fingerprint(self, signal: list[PaperRecord]) -> str:
        """Stable hash over (step name, sorted signal IDs, anchor cap,
        recommendation limit, use_rejected_as_negatives). Excludes
        ``ctx.rejected`` itself so re-running on the same signal is a
        clean no-op even if the rejection set has grown — callers who
        want a fresh fetch should use a different signal."""
        signal_ids = sorted(p.paper_id for p in signal if p.paper_id)
        payload = {
            "step": self.name,
            "signal_ids": signal_ids,
            "max_anchor_papers": self.max_anchor_papers,
            "limit": self.limit,
            "use_rejected_as_negatives": self.use_rejected_as_negatives,
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
                "ExpandBySemantics: signal fingerprint already searched, "
                "skipping (no-op)",
            )
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "already_searched", "fingerprint": fingerprint[:12]},
            )

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

        # 4. Single S2 call — Recommendations API does the SPECTER2 kNN.
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

        # 5. Hydrate hits → PaperRecord instances.
        candidates: list[dict[str, Any]] = []
        for hit in raw or []:
            if not isinstance(hit, dict):
                continue
            pid = hit.get("paperId")
            if isinstance(pid, str) and pid:
                candidates.append({"paper_id": pid})
        hydrated: list[PaperRecord] = (
            ctx.s2.enrich_batch(candidates) if candidates else []
        )

        # 6. Fill abstracts so any title_abstract LLMFilter has text to read.
        if hydrated:
            ctx.s2.enrich_with_abstracts(hydrated)

        # 7. Dedup against ctx.seen + stamp the source label.
        new_records: list[PaperRecord] = []
        for rec in hydrated:
            if not rec.paper_id:
                continue
            if rec.paper_id in ctx.seen:
                continue
            rec.source = "semantic"
            new_records.append(rec)
            ctx.seen.add(rec.paper_id)

        # 8. Source-less filter context.
        fctx = FilterContext(
            ctx=ctx, source=None, source_refs=None, source_citers=None,
        )

        # 9. Apply screener cascade and record rejections.
        passed, rejected = apply_block(new_records, self.screener, fctx)
        record_rejections(rejected, fctx)

        # 10. Survivors → collection.
        for p in passed:
            p.llm_verdict = "accept"
            ctx.collection[p.paper_id] = p

        # 11. Mark fingerprint so re-runs are no-ops.
        ctx.searched_signals.add(fingerprint)

        # 12. Return result with stats.
        return StepResult(
            signal=passed,
            in_count=len(hydrated),
            stats={
                "anchor_count": len(anchor_ids),
                "negative_count": len(negative_ids) if negative_ids else 0,
                "raw_hits": len(raw or []),
                "hydrated": len(hydrated),
                "novel": len(new_records),
                "accepted": len(passed),
                "rejected": len(rejected),
            },
        )
