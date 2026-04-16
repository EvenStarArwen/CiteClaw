"""CitationFilter atom (citation count must outpace age × beta)."""

from __future__ import annotations

from datetime import datetime

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


def _current_year() -> int:
    return datetime.now().year


class CitationFilter:
    """Reject papers whose citation count is below ``beta × years_since_publication``.

    The age-aware threshold (``beta × age``) penalises young papers far less
    than old ones, so a 2-year-old preprint with 60 citations passes a
    ``beta=30`` bar that a 10-year-old paper would need 300 citations to
    clear. ``years_since`` is computed against ``reference_year`` (defaulting
    to the current calendar year) and floored at 1 so brand-new papers still
    have a non-zero bar.

    Two knobs cover the recency-skip case:

    * ``reference_year`` — anchor year for both the age math and the
      exemption window. Defaults to the current calendar year. Pin it
      to keep results reproducible across runs that span a year
      boundary.
    * ``exemption_years`` — when set to an int ``N >= 0``, papers
      published in the last ``N`` years (i.e. ``year >= reference_year - N``)
      skip the citation check entirely. ``N=0`` grants the exemption
      only to papers from ``reference_year`` itself; ``N=1`` adds the
      year before, etc. Defaults to ``None`` (no exemption — preserves
      the original strict behaviour).
    """

    def __init__(
        self,
        name: str = "citation",
        *,
        beta: float = 5.0,
        exemption_years: int | None = None,
        reference_year: int | None = None,
    ) -> None:
        if exemption_years is not None and exemption_years < 0:
            raise ValueError(
                f"exemption_years must be >= 0 when set, got {exemption_years}"
            )
        self.name = name
        self._beta = beta
        self._exemption_years = exemption_years
        self._reference_year = reference_year

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        anchor = self._reference_year if self._reference_year is not None else _current_year()
        if (
            self._exemption_years is not None
            and paper.year is not None
            and paper.year >= anchor - self._exemption_years
        ):
            return PASS
        if paper.citation_count is None:
            return FilterOutcome(False, "missing citation count", "missing_data")
        years_since = max(anchor - (paper.year or 0), 1)
        threshold = years_since * self._beta
        if paper.citation_count < threshold:
            return FilterOutcome(
                False,
                f"cit {paper.citation_count} < {threshold:.0f} (β={self._beta})",
                "citation",
            )
        return PASS
