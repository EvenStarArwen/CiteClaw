"""CitSimMeasure — Jaccard-like citer overlap with the source paper."""

from __future__ import annotations

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord


class CitSimMeasure:
    def __init__(
        self,
        name: str = "cit_sim",
        *,
        pass_if_cited_at_least: int = 10**9,
    ) -> None:
        self.name = name
        self._pass_if = pass_if_cited_at_least

    def compute(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        # Heavily-cited shortcut: treat as max similarity (1.0).
        if (paper.citation_count or 0) >= self._pass_if:
            return 1.0
        if not fctx.source_citers:
            return None
        try:
            cand_citers = {
                x.get("paper_id", "")
                for x in fctx.ctx.s2.fetch_citation_ids_and_counts(paper.paper_id)
                if x.get("paper_id")
            }
        except Exception:
            return None
        if not cand_citers:
            return None
        denom = min(len(fctx.source_citers), len(cand_citers))
        if denom == 0:
            return None
        return len(fctx.source_citers & cand_citers) / denom
