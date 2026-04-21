"""YearFilter — pass papers whose ``year`` is in ``[min, max]``."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class YearFilter:
    """Accept papers whose publication year is inside ``[min, max]``.

    Either bound can be ``None`` to disable that side. Papers with
    ``year is None`` are always rejected (strict — prevents S2's
    occasional missing-year records from silently slipping through).
    """

    def __init__(
        self,
        name: str = "year",
        *,
        min: int | None = None,
        max: int | None = None,
    ) -> None:
        self.name = name
        self._min = min
        self._max = max

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        if paper.year is None:
            return FilterOutcome(False, "year is None", "year")
        if self._min is not None and paper.year < self._min:
            return FilterOutcome(False, f"year {paper.year} < {self._min}", "year")
        if self._max is not None and paper.year > self._max:
            return FilterOutcome(False, f"year {paper.year} > {self._max}", "year")
        return PASS
