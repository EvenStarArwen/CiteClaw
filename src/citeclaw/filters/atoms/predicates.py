"""Route-predicate atoms — used in YAML ``Route`` blocks' ``if:`` clauses.

These are deliberately simpler than the full filter atoms (``YearFilter``,
``CitationFilter``): they implement the same :class:`Filter` Protocol but
get used as one-shot dispatch predicates inside ``Route.select(paper)``,
not as full screeners. The four exported here cover the common dispatch
keys a YAML config needs: venue substring (``VenueIn``), curated venue
family exact match (``VenuePreset``), and "value at least N" gates for
citations and years.

Missing-data convention: ``CitAtLeast`` and ``YearAtLeast`` both fall
back to ``0`` for ``None`` so a positive threshold rejects naturally
(no need for a separate ``missing_data`` category — Route only needs
yes/no).
"""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.filters.venue_presets import resolve_presets
from citeclaw.models import PaperRecord


def _normalize_venue(s: str) -> str:
    """Lowercase + whitespace-collapse for venue exact-match comparisons."""
    return " ".join(s.lower().split())


class VenueIn:
    """Substring (case-insensitive) match against ``paper.venue``.

    Loose by design — accepts ``"Nature"`` against ``"Nature Methods"``
    because the dispatch use case wants any Nature-family venue to
    match. Use :class:`VenuePreset` instead when you need exact match
    against a curated list (e.g. excluding Nature-Inspired Computing).
    """

    def __init__(self, name: str = "VenueIn", *, values: list[str]) -> None:
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

    def __init__(self, name: str = "VenuePreset", *, presets: list[str]) -> None:
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
    """Pass if ``paper.citation_count >= n``. Missing count is treated as 0."""

    def __init__(self, name: str = "CitAtLeast", *, n: int) -> None:
        self.name = name
        self._n = n

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if (paper.citation_count or 0) >= self._n:
            return PASS
        return FilterOutcome(False, f"cit < {self._n}", "cit_at_least")


class YearAtLeast:
    """Pass if ``paper.year >= n``. Missing year is treated as 0."""

    def __init__(self, name: str = "YearAtLeast", *, n: int) -> None:
        self.name = name
        self._n = n

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if (paper.year or 0) >= self._n:
            return PASS
        return FilterOutcome(False, f"year < {self._n}", "year_at_least")
