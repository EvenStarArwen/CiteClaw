"""Sequential filter block — AND of children with short-circuit on first reject."""

from __future__ import annotations

from typing import TYPE_CHECKING

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.filters.base import Filter


class Sequential:
    """AND-compose ``layers`` — pass iff every child passes.

    The runner's batched ``apply_block`` path is the production
    dispatcher; this class's ``check()`` is the synchronous
    Filter-Protocol fallback used when a single paper falls through
    a non-batched code path. Both paths short-circuit on first
    rejection.
    """

    def __init__(
        self, name: str = "sequential", *, layers: "list[Filter] | None" = None,
    ) -> None:
        self.name = name
        self.layers: list = layers or []

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        for layer in self.layers:
            outcome = layer.check(paper, fctx)
            if not outcome.passed:
                return outcome
        return PASS
