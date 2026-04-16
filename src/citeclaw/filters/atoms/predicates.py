"""Route predicate atoms: VenueIn, VenuePreset, CitAtLeast, YearAtLeast."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.filters.venue_presets import resolve_presets
from citeclaw.models import PaperRecord


def _normalize_venue(s: str) -> str:
    return " ".join(s.lower().split())


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


class VenuePreset:
    """Pass if paper's venue exactly matches any venue in the named presets.

    Presets live in ``citeclaw.filters.venue_presets`` (``nature``,
    ``science``, ``cell``, ``preprint``, …). Matching normalizes both
    sides (lowercase + whitespace-collapsed) then compares exactly —
    safer than substring match for large curated lists.
    """

    def __init__(self, name: str = "venue_preset", *, presets: list[str]) -> None:
        self.name = name
        self._presets = list(presets)
        self._venues = {_normalize_venue(v) for v in resolve_presets(self._presets)}

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        v = _normalize_venue(paper.venue or "")
        if v and v in self._venues:
            return PASS
        return FilterOutcome(
            False, f"venue not in presets {self._presets}", "venue_preset"
        )


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
