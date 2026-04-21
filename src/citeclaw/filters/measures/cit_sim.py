"""CitSimMeasure — overlap of citers between candidate and source paper."""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.filters.measures.cit_sim")


class CitSimMeasure:
    """Normalised citer overlap: |citers(src) ∩ citers(cand)| / min(|citers|).

    Returns ``None`` when ``fctx.source_citers`` is unset (anchorless
    screening), when the candidate has no fetchable citers, or when the
    S2 citation fetch raises (network / rate limit / 404). The parent
    :class:`SimilarityFilter` treats ``None`` as "no signal" and skips
    this measure when computing the max.

    ``pass_if_cited_at_least`` is an opt-in shortcut: papers with at
    least that many total citations skip the citer fetch and return the
    maximum score (1.0) directly. Default ``10**9`` effectively
    disables the shortcut.
    """

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
        except Exception as exc:
            # Non-fatal — SimilarityFilter falls back to other measures.
            log.debug(
                "cit_sim: fetch_citation_ids_and_counts(%r) failed: %s",
                paper.paper_id, exc,
            )
            return None
        if not cand_citers:
            return None
        denom = min(len(fctx.source_citers), len(cand_citers))
        if denom == 0:
            return None
        return len(fctx.source_citers & cand_citers) / denom
