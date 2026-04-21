"""RefSimMeasure — overlap of references between candidate and source paper."""

from __future__ import annotations

import logging

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.filters.measures.ref_sim")


class RefSimMeasure:
    """Normalised reference overlap: |refs(src) ∩ refs(cand)| / min(|refs|).

    Returns ``None`` when ``fctx.source_refs`` is unset (anchorless
    screening), when the candidate has no fetchable references, or when
    the S2 reference fetch raises (network / rate limit / 404). The
    parent :class:`SimilarityFilter` treats ``None`` as "no signal" and
    skips this measure when computing the max.
    """

    def __init__(self, name: str = "ref_sim") -> None:
        self.name = name

    def compute(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        if not fctx.source_refs:
            return None
        try:
            cand_refs = set(fctx.ctx.s2.fetch_reference_ids(paper.paper_id))
        except Exception as exc:
            # Non-fatal — SimilarityFilter falls back to other measures.
            log.debug(
                "ref_sim: fetch_reference_ids(%r) failed: %s",
                paper.paper_id, exc,
            )
            return None
        if not cand_refs:
            return None
        denom = min(len(fctx.source_refs), len(cand_refs))
        if denom == 0:
            return None
        return len(fctx.source_refs & cand_refs) / denom
