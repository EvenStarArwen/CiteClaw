"""Any filter block — OR of children with short-circuit on first pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.filters.base import Filter


class Any_:
    """OR-compose ``layers`` — pass iff any child passes.

    The runner's batched ``apply_block`` path is the production
    dispatcher; this class's ``check()`` is the synchronous
    Filter-Protocol fallback. Both paths short-circuit on first pass.
    The empty-layers and all-layers-fail cases return a
    ``FilterOutcome`` with category ``"any"`` so the dashboard groups
    them together rather than scattering across each child's category.
    """

    def __init__(
        self, name: str = "any", *, layers: "list[Filter] | None" = None,
    ) -> None:
        self.name = name
        self.layers: list = layers or []

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        last: FilterOutcome | None = None
        for layer in self.layers:
            outcome = layer.check(paper, fctx)
            if outcome.passed:
                return PASS
            last = outcome
        return last or FilterOutcome(False, "no layers passed", "any")
