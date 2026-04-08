"""RefSimMeasure — Jaccard-like reference overlap with the source paper."""

from __future__ import annotations

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord


class RefSimMeasure:
    def __init__(self, name: str = "ref_sim") -> None:
        self.name = name

    def compute(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        if not fctx.source_refs:
            return None
        try:
            cand_refs = set(fctx.ctx.s2.fetch_reference_ids(paper.paper_id))
        except Exception:
            return None
        if not cand_refs:
            return None
        denom = min(len(fctx.source_refs), len(cand_refs))
        if denom == 0:
            return None
        return len(fctx.source_refs & cand_refs) / denom
