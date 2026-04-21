"""Route filter block — if/elif/else dispatch over predicate cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from citeclaw.filters.base import FilterContext, FilterOutcome
from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.filters.base import Filter


@dataclass
class RouteCase:
    """One branch of a Route block.

    ``predicate`` is the test (a Filter atom — typically a Route
    predicate from :mod:`citeclaw.filters.atoms.predicates`); ``target``
    is the Filter to dispatch matching papers to; ``is_default=True``
    marks the catch-all case (no predicate, always matches). The YAML
    builder appends the default case last so :meth:`Route.select` reads
    it as the "else" branch.
    """

    predicate: "Filter | None"
    target: "Filter"
    is_default: bool = False


class Route:
    """Dispatch each paper to the first matching case's ``target`` filter.

    Cases are evaluated top-to-bottom. The first ``RouteCase`` whose
    ``predicate.check()`` passes (or ``is_default=True``) wins; all
    subsequent cases are unreachable for that paper. Papers that match
    no case fall through and reject with ``"no_route_match"``.
    """

    def __init__(
        self, name: str = "route", *, cases: list[RouteCase] | None = None,
    ) -> None:
        self.name = name
        self.cases = cases or []

    def select(
        self, paper: PaperRecord, fctx: FilterContext,
    ) -> "Filter | None":
        """Return the chosen target Filter, or None if no case matched."""
        for case in self.cases:
            if case.is_default:
                return case.target
            if case.predicate.check(paper, fctx).passed:
                return case.target
        return None

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        target = self.select(paper, fctx)
        if target is None:
            return FilterOutcome(False, "no_route_match", "no_route_match")
        return target.check(paper, fctx)
