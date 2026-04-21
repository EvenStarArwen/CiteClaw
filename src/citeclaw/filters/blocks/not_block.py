"""Not_ — invert the verdict of a single child filter block."""

from __future__ import annotations

from typing import TYPE_CHECKING

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.filters.base import Filter


class Not_:
    """Negate ``layer``: pass iff the wrapped filter would have rejected.

    Carries a single child (``layer:`` singular in YAML, contrast
    Sequential / Any which take ``layers:``). The rejection category
    is ``"not_<child name>"`` so the dashboard can distinguish a
    "rejected by the inverse" reason from a vanilla rejection by the
    same child filter elsewhere.
    """

    def __init__(self, name: str = "not", *, layer: "Filter") -> None:
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
