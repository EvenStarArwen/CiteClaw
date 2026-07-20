"""CitationFilter — accept papers whose citation count outpaces ``beta × f(age)``."""

from __future__ import annotations

import math
from datetime import datetime

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


def _current_year() -> int:
    """Wall-clock year — extracted so tests can monkeypatch it deterministically."""
    return datetime.now().year


# Age→multiplier shapes for the threshold ``beta * f(age)``. Every curve is
# normalized so ``f(1) == 1`` — beta always means "citations required at age
# one year", and the curve only controls how the bar grows beyond that.
CURVES = ("linear", "sqrt", "log", "exp")


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
    * ``exemption_years`` — papers from the last ``N+1`` years skip
      the citation check. The semantics work as follows:

        * ``N = 0`` (the default): only papers published in
          ``reference_year`` itself skip — current-calendar-year work
          almost always has zero citations on S2, so without this the
          SPECTER2 / search-agent expansion has nothing to feed into
          the citation gate.
        * ``N = 1``: ``reference_year`` AND ``reference_year - 1`` skip
          (i.e. last-two-years window).
        * ``N = -1``: disable the exemption entirely — every paper
          must clear ``years_since * beta``, matching the original
          pre-2026-05 strict behaviour. Use this when you genuinely
          want to filter out brand-new work that hasn't earned its
          citations yet.

      Note that the implementation accepts any integer; values < -1 are
      treated the same as -1 (no exemption).
    """

    def __init__(
        self,
        name: str = "citation",
        *,
        beta: float = 5.0,
        exemption_years: int = 0,
        reference_year: int | None = None,
        curve: str = "linear",
        exp_base: float = 2.0,
    ) -> None:
        """``curve`` picks the age→threshold shape (all satisfy f(1)=1, so
        ``beta`` is always the bar at age one year):

        * ``linear`` — ``beta * age`` (the historical behaviour)
        * ``sqrt``   — ``beta * sqrt(age)``: sub-linear, gentler on old papers
        * ``log``    — ``beta * log2(1 + age)``: near-flat for old papers
        * ``exp``    — ``beta * exp_base**(age - 1)``: compounding bar for
          old papers (``exp_base`` per extra year; must be > 1)
        """
        if curve not in CURVES:
            raise ValueError(f"CitationFilter curve must be one of {CURVES}, got {curve!r}")
        if curve == "exp" and exp_base <= 1:
            raise ValueError("CitationFilter exp_base must be > 1")
        self.name = name
        self._beta = beta
        self._exemption_years = exemption_years
        self._reference_year = reference_year
        self._curve = curve
        self._exp_base = exp_base

    def _threshold(self, years_since: int) -> float:
        a = float(years_since)
        if self._curve == "sqrt":
            mult = math.sqrt(a)
        elif self._curve == "log":
            mult = math.log2(1.0 + a)
        elif self._curve == "exp":
            mult = self._exp_base ** (a - 1.0)
        else:
            mult = a
        return self._beta * mult

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        """Evaluate ``paper`` in three stages.

        1. Recency exemption — if configured and the paper falls inside
           the window, pass without consulting the citation count.
        2. Missing-data reject — papers with no ``citation_count`` reject
           in the ``missing_data`` category, distinct from a real
           citation-floor failure.
        3. Threshold — ``citation_count >= max(anchor - paper.year, 1) * beta``.
           Papers with ``year is None`` are effectively treated as
           ancient (``years_since = anchor``), so only extreme citation
           counts can pass.
        """
        anchor = self._reference_year if self._reference_year is not None else _current_year()
        if (
            self._exemption_years >= 0
            and paper.year is not None
            and paper.year >= anchor - self._exemption_years
        ):
            return PASS
        if paper.citation_count is None:
            return FilterOutcome(False, "missing citation count", "missing_data")
        years_since = max(anchor - (paper.year or 0), 1)
        threshold = self._threshold(years_since)
        if paper.citation_count < threshold:
            tag = f"β={self._beta}" if self._curve == "linear" else f"β={self._beta}·{self._curve}"
            return FilterOutcome(
                False,
                f"cit {paper.citation_count} < {threshold:.0f} ({tag})",
                "citation",
            )
        return PASS
