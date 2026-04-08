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
        seeds = [sp for sp in cfg.seed_papers if sp.paper_id.strip()]
        dash.begin_phase("fetch seed metadata", total=max(1, len(seeds)))

        records: list[PaperRecord] = []
        for sp in seeds:
            pid = sp.paper_id.strip()
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
