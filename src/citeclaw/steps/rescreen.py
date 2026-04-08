"""ReScreen — destructive global cleanup pass: apply a screener block to ctx.collection."""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.rescreen")


class ReScreen:
    name = "ReScreen"

    def __init__(self, *, screener=None) -> None:
        self.screener = screener

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(signal=signal, in_count=len(signal), stats={"removed": 0})

        candidates = [p for p in ctx.collection.values() if p.source != "seed"]
        if not candidates:
            return StepResult(signal=signal, in_count=len(signal), stats={"removed": 0})

        dash = ctx.dashboard
        dash.note_candidates_seen(len(candidates))

        # apply_block walks the screener cascade and drives the inner bar.
        fctx = FilterContext(ctx=ctx)
        passed, rejected = apply_block(candidates, self.screener, fctx)

        for p, outcome in rejected:
            if p.paper_id in ctx.collection:
                del ctx.collection[p.paper_id]
                ctx.rejected.add(p.paper_id)
                key = f"rescreen_{outcome.category or 'unknown'}"
                ctx.rejection_counts[key] = ctx.rejection_counts.get(key, 0) + 1

        out_signal = [p for p in signal if p.paper_id in ctx.collection]
        return StepResult(
            signal=out_signal,
            in_count=len(signal),
            stats={"removed": len(rejected), "kept": len(passed)},
        )
