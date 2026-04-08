"""Route predicate atoms: VenueIn, CitAtLeast, YearAtLeast."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class VenueIn:
    def __init__(self, name: str = "venue_in", *, values: list[str]) -> None:
        self.name = name
        self._values = [v.lower() for v in values]

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        v = (paper.venue or "").lower()
        for needle in self._values:
            if needle and needle in v:
                return PASS
        return FilterOutcome(False, f"venue not in {self._values}", "venue_in")


class CitAtLeast:
    def __init__(self, name: str = "cit_at_least", *, n: int) -> None:
        self.name = name
        self._n = n

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if (paper.citation_count or 0) >= self._n:
            return PASS
        return FilterOutcome(False, f"cit < {self._n}", "cit_at_least")


class YearAtLeast:
    def __init__(self, name: str = "year_at_least", *, n: int) -> None:
        self.name = name
        self._n = n

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if (paper.year or 0) >= self._n:
            return PASS
        return FilterOutcome(False, f"year < {self._n}", "year_at_least")
