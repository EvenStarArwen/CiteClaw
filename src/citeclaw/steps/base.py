"""BaseStep Protocol + StepResult — the contract every pipeline step satisfies.

Every concrete step (LoadSeeds, ExpandForward, Rerank, Cluster, Finalize, …)
implements :class:`BaseStep` and is registered in
:data:`citeclaw.steps.STEP_REGISTRY` so the YAML builder
(:func:`citeclaw.steps.build_step`) can dispatch by ``step:`` name.

A step is a pure signal transformer: in-list -> out-list (+ stats).
Steps SHOULD mutate ``ctx.collection`` / ``ctx.rejected`` / ``ctx.expanded_*``
when adding or removing papers from the cumulative state, but MUST NOT
mutate the input ``signal`` list. The pipeline runner threads the
returned :attr:`StepResult.signal` to the next step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.context import Context


@dataclass(frozen=True)
class StepResult:
    """One step's output — frozen so the pipeline runner can pass it
    around without defensive copying.

    ``signal`` is the (possibly filtered, reranked, expanded) paper list
    handed to the next step. ``in_count`` is the size of the input
    signal — the runner uses it for the shape table and event sink.
    ``stats`` is a free-form per-step diagnostic dict rendered into the
    dashboard and the run summary; the runner defensively copies it
    before forwarding to event sinks. ``stop_pipeline``, when True,
    short-circuits the runner to ``Finalize`` after this step — used
    today only by ``HumanInTheLoop`` to surface a "user clicked stop"
    signal cleanly.
    """

    signal: list[PaperRecord]
    in_count: int
    stats: dict[str, Any] = field(default_factory=dict)
    stop_pipeline: bool = False


@runtime_checkable
class BaseStep(Protocol):
    """Protocol every pipeline step implements.

    Concrete implementations live in sibling modules
    (:mod:`citeclaw.steps.expand_forward`, :mod:`.rerank`, :mod:`.cluster`,
    …) and are registered via :data:`citeclaw.steps.STEP_REGISTRY`.
    The runtime-checkable variant lets the runner ``isinstance``-test a
    builder-produced object during shape-log debugging.
    """

    name: str

    def run(
        self, signal: list[PaperRecord], ctx: "Context",
    ) -> StepResult:
        """Transform ``signal`` and return a :class:`StepResult`.

        Implementations should consult ``ctx.s2`` / ``ctx.cache`` /
        ``ctx.collection`` rather than re-fetching, and update
        ``ctx.collection`` / ``ctx.rejected`` / ``ctx.expanded_*`` for
        any cumulative-state changes. Must not mutate ``signal`` itself.
        """
        ...
