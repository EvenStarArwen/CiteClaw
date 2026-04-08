"""Not_ — invert a child filter block."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class Not_:
    def __init__(self, name: str = "not", *, layer) -> None:
        self.name = name
        self.layer = layer

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        # Synchronous fallback path; the runner short-circuits via apply_block
        # for batched LLM efficiency.
        outcome = self.layer.check(paper, fctx)
        if outcome.passed:
            return FilterOutcome(
                False,
                f"not({self.layer.name})",
                f"not_{self.layer.name}",
            )
        return PASS
