"""SimilarityMeasure Protocol — uniform interface over normalized similarity scores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord


@runtime_checkable
class SimilarityMeasure(Protocol):
    name: str

    def compute(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        """Return normalized score in [0, 1], or None if data unavailable."""
        ...
