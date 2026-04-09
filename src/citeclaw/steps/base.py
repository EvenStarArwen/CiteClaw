"""BaseStep + StepResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from citeclaw.models import PaperRecord


@dataclass
class StepResult:
    signal: list[PaperRecord]
    in_count: int
    stats: dict[str, Any] = field(default_factory=dict)
    # When True, the pipeline runner short-circuits to ``Finalize`` after
    # this step. Used by ``HumanInTheLoop`` to surface a "user clicked
    # stop" signal cleanly — every other step leaves it False.
    stop_pipeline: bool = False


class BaseStep(Protocol):
    name: str
    def run(self, signal: list[PaperRecord], ctx) -> StepResult: ...
