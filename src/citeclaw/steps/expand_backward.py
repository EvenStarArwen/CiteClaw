"""ExpandBackward step — for each paper in signal, fetch references and screen."""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_backward")


class ExpandBackward:
    name = "ExpandBackward"

    def __init__(self, *, screener=None) -> None:
        self.screener = screener

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(signal=[], in_count=len(signal), stats={"reason": "no screener"})

        dash = ctx.dashboard
        dash.enable_outer_bar(total=len(signal), description="source papers")

        accepted: list[PaperRecord] = []
        for source in signal:
            if source.paper_id in ctx.expanded_backward:
                dash.advance_outer(1)
                continue
            ctx.expanded_backward.add(source.paper_id)

            dash.begin_phase("fetch refs", total=1)
            try:
                ref_records = ctx.s2.fetch_references(source.paper_id)
            except Exception as exc:
                log.warning("backward: failed for %s: %s", source.paper_id[:20], exc)
                dash.advance_outer(1)
                continue
            dash.tick_inner(1)
            source.references = [r.paper_id for r in ref_records if r.paper_id]

            cands: list[PaperRecord] = []
            for r in ref_records:
                if not r.paper_id or r.paper_id in ctx.seen:
                    continue
                ctx.seen.add(r.paper_id)
                r.depth = source.depth + 1
                r.source = "backward"
                r.supporting_papers = [source.paper_id]
                cands.append(r)

            if not cands:
                dash.advance_outer(1)
                continue
            dash.note_candidates_seen(len(cands))

            dash.begin_phase("enrich · abstracts", total=1)
            ctx.s2.enrich_with_abstracts(cands)
            dash.tick_inner(1)

            fctx = FilterContext(ctx=ctx, source=source)
            passed, rejected = apply_block(cands, self.screener, fctx)
            record_rejections(rejected, fctx)
            for p in passed:
                p.llm_verdict = "accept"
                ctx.collection[p.paper_id] = p
                accepted.append(p)
                dash.paper_accepted(p, saturation=saturation_for_paper(p, ctx))

            dash.advance_outer(1)

        return StepResult(
            signal=accepted, in_count=len(signal),
            stats={"accepted": len(accepted)},
        )
