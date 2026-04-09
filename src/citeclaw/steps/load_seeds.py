"""LoadSeeds step — fetch seed paper metadata and add to ctx.collection."""

from __future__ import annotations

import logging

from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.load_seeds")


class LoadSeeds:
    name = "LoadSeeds"

    def __init__(self, file: str | None = None) -> None:
        self.file = file

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        cfg = ctx.config
        dash = ctx.dashboard

        # PC-04: prefer the IDs resolved by an upstream ``ResolveSeeds``
        # step (which may have expanded title-only entries and added
        # preprint/published siblings). Fall back to ``cfg.seed_papers``
        # for the legacy single-step pipeline. Each item is
        # ``(paper_id, source_seed_paper_or_None)`` so the existing
        # title/abstract fallback logic still applies for direct entries.
        items: list[tuple[str, object | None]]
        if ctx.resolved_seed_ids:
            items = [(pid, None) for pid in ctx.resolved_seed_ids if pid]
        else:
            items = [
                (sp.paper_id.strip(), sp)
                for sp in cfg.seed_papers
                if sp.paper_id.strip()
            ]
        dash.begin_phase("fetch seed metadata", total=max(1, len(items)))

        records: list[PaperRecord] = []
        for pid, sp in items:
            try:
                rec = ctx.s2.fetch_metadata(pid)
            except Exception as exc:
                log.error("Failed to fetch seed %s: %s", pid, exc)
                dash.tick_inner(1)
                continue
            if rec.paper_id in ctx.collection:
                log.info("Seed %s already in collection — skipping", rec.paper_id[:20])
                dash.tick_inner(1)
                continue
            if sp is not None:
                if sp.title and not rec.title:
                    rec.title = sp.title
                if sp.abstract and not rec.abstract:
                    rec.abstract = sp.abstract
            rec.depth = 0
            rec.source = "seed"
            rec.llm_verdict = "accept_seed"
            ctx.collection[rec.paper_id] = rec
            ctx.seen.add(rec.paper_id)
            ctx.seed_ids.add(rec.paper_id)
            records.append(rec)
            dash.note_candidates_seen(1)
            dash.paper_accepted(rec, saturation=saturation_for_paper(rec, ctx))
            dash.tick_inner(1)
        ctx.new_seed_ids = [r.paper_id for r in records]
        return StepResult(signal=records, in_count=len(signal), stats={"loaded": len(records)})
