"""``LoadSeeds`` step — fetch seed paper metadata and seed ``ctx.collection``.

This is the first step of every pipeline. It reads the resolved seed
IDs (from an upstream :class:`~citeclaw.steps.resolve_seeds.ResolveSeeds`
when present, falling back to ``Settings.seed_papers``), fetches each
paper's metadata via the S2 client, stamps ``depth=0`` /
``source="seed"`` / ``llm_verdict="accept_seed"``, and adds them to
``ctx.collection`` / ``ctx.seen`` / ``ctx.seed_ids``. Returns the
loaded records as the next step's input signal.

Per-seed S2 errors are logged at ERROR (not silent) and the seed is
skipped — one bad DOI doesn't abort the whole run. Seeds already in
the collection (e.g. from a ``--continue-from`` checkpoint) are
preserved unchanged.
"""

from __future__ import annotations

import logging

from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.load_seeds")


class LoadSeeds:
    """Pipeline step that hydrates seed papers into ``ctx.collection``."""

    name = "LoadSeeds"

    def __init__(self, file: str | None = None) -> None:
        # ``file`` is a placeholder for a future "load seeds from file"
        # mode (currently always None — the YAML builder accepts the
        # field for forward compatibility but the run() method ignores
        # it; seeds come from ``Settings.seed_papers`` or the upstream
        # ResolveSeeds step). Kept as a typed kwarg so the YAML schema
        # stays stable when the file mode lands.
        self.file = file

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        cfg = ctx.config
        dash = ctx.dashboard

        # Prefer the IDs resolved by an upstream ``ResolveSeeds`` step
        # (which may have expanded title-only entries and added
        # preprint/published siblings). Fall back to ``cfg.seed_papers``
        # when there's no resolver upstream. Each item is
        # ``(paper_id, source_seed_paper_or_None)`` so direct entries
        # can still pass title/abstract fallbacks down.
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
