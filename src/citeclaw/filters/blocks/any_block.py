"""Any filter block — OR with short-circuit."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class Any_:
    def __init__(self, name: str = "any", *, layers: list = None) -> None:
        self.name = name
        self.layers = layers or []

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        last: FilterOutcome | None = None
        for layer in self.layers:
            outcome = layer.check(paper, fctx)
            if outcome.passed:
                return PASS
            last = outcome
        return last or FilterOutcome(False, "no layers passed", "any")
