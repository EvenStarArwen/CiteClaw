"""ExpandForward step — for each paper in signal, fetch citers and screen them."""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_forward")


class ExpandForward:
    name = "ExpandForward"

    def __init__(self, *, max_citations: int = 100, screener=None) -> None:
        self.max_citations = max_citations
        self.screener = screener

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(signal=[], in_count=len(signal), stats={"reason": "no screener"})

        dash = ctx.dashboard
        dash.enable_outer_bar(total=len(signal), description="source papers")

        accepted: list[PaperRecord] = []
        for source in signal:
            if source.paper_id in ctx.expanded_forward:
                dash.advance_outer(1)
                continue
            ctx.expanded_forward.add(source.paper_id)

            dash.begin_phase("fetch citers", total=1)
            try:
                citers = ctx.s2.fetch_citation_ids_and_counts(source.paper_id)
            except Exception as exc:
                log.warning("forward: failed to fetch citers for %s: %s", source.paper_id[:20], exc)
                dash.advance_outer(1)
                continue
            dash.tick_inner(1)

            citers.sort(key=lambda x: x.get("citation_count") or 0, reverse=True)
            citers = citers[: self.max_citations]
            unseen = [c for c in citers if c.get("paper_id") and c["paper_id"] not in ctx.seen]
            if not unseen:
                dash.advance_outer(1)
                continue
            for c in unseen:
                ctx.seen.add(c["paper_id"])
            dash.note_candidates_seen(len(unseen))

            dash.begin_phase("fetch source refs", total=1)
            try:
                source_refs = set(ctx.s2.fetch_reference_ids(source.paper_id))
            except Exception:
                source_refs = set()
            dash.tick_inner(1)

            source_citers = {c.get("paper_id") for c in citers if c.get("paper_id")}

            dash.begin_phase("enrich · batch", total=1)
            records = ctx.s2.enrich_batch(unseen)
            for rec in records:
                rec.depth = source.depth + 1
                rec.source = "forward"
                rec.supporting_papers = [source.paper_id]
                if source.paper_id not in rec.references:
                    rec.references.append(source.paper_id)
            dash.tick_inner(1)

            dash.begin_phase("enrich · abstracts", total=1)
            # Need abstracts for title_abstract LLMFilters
            ctx.s2.enrich_with_abstracts(records)
            dash.tick_inner(1)

            fctx = FilterContext(
                ctx=ctx, source=source, source_refs=source_refs, source_citers=source_citers,
            )
            # apply_block drives the inner bar through the screener cascade.
            passed, rejected = apply_block(records, self.screener, fctx)
            record_rejections(rejected, fctx)
            for p in passed:
                p.llm_verdict = "accept"
                ctx.collection[p.paper_id] = p
                accepted.append(p)
                # Saturation: cache-only ref lookup so we never trigger
                # surprise S2 calls for the metric. Will be None for any
                # paper whose references aren't already cached (e.g.
                # because the screener didn't include a CitSim/RefSim).
                dash.paper_accepted(p, saturation=saturation_for_paper(p, ctx))

            dash.advance_outer(1)

        return StepResult(
            signal=accepted, in_count=len(signal),
            stats={"accepted": len(accepted)},
        )
