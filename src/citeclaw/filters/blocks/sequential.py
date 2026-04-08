"""Sequential filter block — AND with short-circuit."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class Sequential:
    def __init__(self, name: str = "sequential", *, layers: list = None) -> None:
        self.name = name
        self.layers = layers or []

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        for layer in self.layers:
            outcome = layer.check(paper, fctx)
            if not outcome.passed:
                return outcome
        return PASS
