"""Filter Protocol + FilterContext + FilterOutcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.context import Context


@dataclass
class FilterOutcome:
    """Result of running one filter on one paper."""

    passed: bool
    reason: str = ""
    category: str = ""


PASS = FilterOutcome(passed=True)


@dataclass
class FilterContext:
    """Per-screening context: shared Context + (optional) source paper info."""

    ctx: "Context"
    source: PaperRecord | None = None
    source_refs: set[str] | None = None
    source_citers: set[str] | None = None


@runtime_checkable
class Filter(Protocol):
    name: str

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome: ...
