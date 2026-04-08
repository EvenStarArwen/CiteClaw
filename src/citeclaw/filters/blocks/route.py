"""Route filter block — if/elif/else dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citeclaw.filters.base import FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


@dataclass
class RouteCase:
    predicate: Any  # Filter (None for default)
    target: Any     # Filter
    is_default: bool = False


class Route:
    def __init__(self, name: str = "route", *, cases: list[RouteCase] = None) -> None:
        self.name = name
        self.cases = cases or []

    def select(self, paper: PaperRecord, fctx: FilterContext):
        """Return the chosen target Filter, or None if none matches."""
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
