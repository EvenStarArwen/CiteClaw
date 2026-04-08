"""CitationFilter atom (citation count must outpace age × beta)."""

from __future__ import annotations

from datetime import datetime

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


def _current_year() -> int:
    return datetime.now().year


class CitationFilter:
    def __init__(self, name: str = "citation", *, beta: float = 5.0) -> None:
        self.name = name
        self._beta = beta

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if paper.citation_count is None:
            return FilterOutcome(False, "missing citation count", "missing_data")
        years_since = max(_current_year() - (paper.year or 0), 1)
        threshold = years_since * self._beta
        if paper.citation_count < threshold:
            return FilterOutcome(
                False,
                f"cit {paper.citation_count} < {threshold:.0f} (β={self._beta})",
                "citation",
            )
        return PASS
