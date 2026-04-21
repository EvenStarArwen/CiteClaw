"""``ReScreen`` step — apply a screener over the whole ``ctx.collection``.

This is the **only step that removes papers from ``ctx.collection``**
once they've been accepted. Useful as a final pruning pass after
expansions have collected a wide net of candidates — e.g. tightening
year bounds, adding a new LLM filter, or applying a stricter venue
preset to drop earlier-accepted papers that wouldn't survive the new
criterion.

Important invariants:

* **Seed papers are exempt** — anything stamped ``source="seed"`` by
  :class:`~citeclaw.steps.load_seeds.LoadSeeds` is excluded from the
  candidate pool and stays in the collection regardless of the
  screener verdict. This protects the user-supplied entry points from
  accidental removal.
* **Rejected papers move to ``ctx.rejected``** with rejection
  category ``rescreen_<original-category>`` so the dashboard /
  ledger can distinguish a fresh rejection from a re-screen drop.
* **The returned signal is filtered to surviving papers only** —
  callers downstream of ReScreen never see a paper that's been
  removed from ``ctx.collection``.
* When ``screener`` is ``None`` the step is a no-op (signal passes
  through unchanged, ``stats={"removed": 0}``).

Source-less :class:`FilterContext` (``source=None``) — the screener
cascade must tolerate it (PC-05 invariant; same as the ``ExpandBy*``
family).
"""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.rescreen")


class ReScreen:
    """Apply a screener block to ``ctx.collection`` and remove the rejects."""

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
