"""Filter Protocol, FilterContext, FilterOutcome — the bottom of the filter layer.

Every concrete filter (atoms in :mod:`citeclaw.filters.atoms`, blocks in
:mod:`citeclaw.filters.blocks`) implements the :class:`Filter` Protocol.
The runner in :mod:`citeclaw.filters.runner` does the bulk dispatch.

Three shapes live here:

* :class:`FilterOutcome` — the immutable verdict returned by ``Filter.check``.
* :class:`FilterContext` — the per-screening context passed alongside each
  paper. Carries the shared :class:`citeclaw.context.Context` plus an
  optional ``source`` paper for filters whose decision depends on a graph
  edge (e.g. SimilarityFilter measures Jaccard against ``source.references``).
* :class:`Filter` — the runtime-checkable Protocol every filter satisfies.

The module-level :data:`PASS` singleton lets atom implementations return
the common "pass with no reason" verdict without re-allocating the
dataclass on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from citeclaw.models import PaperRecord

if TYPE_CHECKING:
    from citeclaw.context import Context


@dataclass(frozen=True)
class FilterOutcome:
    """Immutable verdict from one filter on one paper.

    ``passed`` is the only required field. ``reason`` is a human-readable
    explanation rendered into the dashboard and the rejection ledger;
    ``category`` is the bucket key the dashboard groups rejections by
    (e.g. ``"year"``, ``"citation"``, ``"llm"``). Frozen so the shared
    :data:`PASS` singleton can't be mutated by accident.
    """

    passed: bool
    reason: str = ""
    category: str = ""


PASS = FilterOutcome(passed=True)
"""Shared "pass with no reason" verdict — return this from any filter
that has no diagnostic to attach. Re-used to avoid per-call allocation."""


@dataclass
class FilterContext:
    """Per-screening context handed to every :meth:`Filter.check` call.

    ``ctx`` is always populated. The ``source`` triple is optional — it
    is set to a real paper only when the filter is being asked about an
    *edge* (e.g. "should we accept this citation OF ``source``?") and
    left ``None`` for source-less screening (the ExpandBy* family
    anchors on the signal, not on individual edges). ``source_refs`` and
    ``source_citers`` are pre-computed sets the runner derives once per
    source so that per-paper measures (RefSim, CitSim) don't re-fetch.
    """

    ctx: "Context"
    source: PaperRecord | None = None
    source_refs: set[str] | None = None
    source_citers: set[str] | None = None


@runtime_checkable
class Filter(Protocol):
    """The Protocol every filter atom and block implements.

    A filter is a *pure* paper-to-verdict function: given one paper and
    the screening context, return a :class:`FilterOutcome`. No I/O, no
    state mutation on the paper itself; rejection bookkeeping is handled
    by the runner via :func:`citeclaw.filters.runner.record_rejections`.
    """

    name: str

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        """Decide whether ``paper`` passes this filter.

        Implementations may consult ``fctx.ctx`` (e.g. for cached graph
        edges) and ``fctx.source`` when present (for edge-anchored
        decisions). Always return a :class:`FilterOutcome` — never raise
        for "this paper doesn't match"; raise only for bona fide
        programming or configuration errors.
        """
        ...
