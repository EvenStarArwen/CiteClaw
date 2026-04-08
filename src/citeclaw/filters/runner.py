"""apply_block: walk a Filter (atom or composite) over a list of papers.

Handles batched LLM dispatch and Route partitioning, so any nested
Sequential/Any/Route tree gets correct batching at every LLMFilter node.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.base import FilterContext, FilterOutcome
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.filters.runner")


def _is_llm_layer(block: Any) -> bool:
    """Return True iff the block is an LLM filter (or a Not_-wrapped one).

    Used by ``apply_block``'s Sequential branch to decide whether the
    dashboard's inner bar will be ticked by the dispatcher (LLM) or by
    apply_block itself (cheap filters).
    """
    if isinstance(block, LLMFilter):
        return True
    if isinstance(block, Not_) and isinstance(block.layer, LLMFilter):
        return True
    return False


def apply_block(
    papers: list[PaperRecord],
    block: Any,
    fctx: FilterContext,
) -> tuple[list[PaperRecord], list[tuple[PaperRecord, FilterOutcome]]]:
    """Walk ``block`` over ``papers``; return ``(passed, rejected)``."""
    if not papers:
        return [], []

    if isinstance(block, Sequential):
        passed = list(papers)
        rejected: list[tuple[PaperRecord, FilterOutcome]] = []
        dash = getattr(fctx.ctx, "dashboard", None)
        for layer in block.layers:
            if not passed:
                break
            in_count = len(passed)
            # Tell the dashboard which filter is running and how many
            # papers it's about to chew on. LLM filters re-set the total
            # in their dispatcher (to account for voting); cheap filters
            # are completed in one tick after they return.
            if dash is not None:
                layer_name = getattr(layer, "name", type(layer).__name__)
                dash.begin_phase(layer_name, total=in_count)
            p, r = apply_block(passed, layer, fctx)
            passed = p
            rejected.extend(r)
            if dash is not None and not _is_llm_layer(layer):
                # Cheap / non-LLM filters: complete the bar in one shot
                # since the layer ran synchronously to completion.
                dash.tick_inner(in_count)
        return passed, rejected

    if isinstance(block, Any_):
        decided_pass: list[PaperRecord] = []
        undecided = list(papers)
        last_rej: dict[str, FilterOutcome] = {}
        for layer in block.layers:
            if not undecided:
                break
            p, r = apply_block(undecided, layer, fctx)
            decided_pass.extend(p)
            for rec, outcome in r:
                last_rej[rec.paper_id] = outcome
            passed_ids = {rec.paper_id for rec in p}
            undecided = [x for x in undecided if x.paper_id not in passed_ids]
        rejected = [(x, last_rej.get(x.paper_id, FilterOutcome(False, "any: all failed", "any"))) for x in undecided]
        return decided_pass, rejected

    if isinstance(block, Not_):
        inner_passed, _inner_rejected = apply_block(papers, block.layer, fctx)
        inner_passed_ids = {p.paper_id for p in inner_passed}
        new_passed = [p for p in papers if p.paper_id not in inner_passed_ids]
        new_rejected = [
            (
                p,
                FilterOutcome(
                    False,
                    f"not({block.layer.name})",
                    f"not_{block.layer.name}",
                ),
            )
            for p in inner_passed
        ]
        return new_passed, new_rejected

    if isinstance(block, Route):
        groups: dict[int, tuple[Any, list[PaperRecord]]] = {}
        rejected: list[tuple[PaperRecord, FilterOutcome]] = []
        for paper in papers:
            target = block.select(paper, fctx)
            if target is None:
                rejected.append((paper, FilterOutcome(False, "no_route_match", "no_route_match")))
                continue
            key = id(target)
            groups.setdefault(key, (target, []))[1].append(paper)
        passed: list[PaperRecord] = []
        for _, (target, plist) in groups.items():
            p, r = apply_block(plist, target, fctx)
            passed.extend(p)
            rejected.extend(r)
        return passed, rejected

    if isinstance(block, LLMFilter):
        from citeclaw.screening.llm_runner import dispatch_batch

        verdicts = dispatch_batch(papers, block, fctx.ctx)
        passed = [p for p in papers if verdicts.get(p.paper_id, False)]
        rejected = [
            (p, FilterOutcome(False, f"llm:{block.name}", f"llm_{block.name}"))
            for p in papers if not verdicts.get(p.paper_id, False)
        ]
        return passed, rejected

    # Atom: prefer batched dispatch if the block offers check_batch (e.g.
    # SimilarityFilter, whose measures may prefetch). Else per-paper check.
    check_batch = getattr(block, "check_batch", None)
    if callable(check_batch):
        outcomes = check_batch(papers, fctx)
        passed = [p for p, o in zip(papers, outcomes) if o.passed]
        rejected = [(p, o) for p, o in zip(papers, outcomes) if not o.passed]
        return passed, rejected

    passed: list[PaperRecord] = []
    rejected: list[tuple[PaperRecord, FilterOutcome]] = []
    for paper in papers:
        outcome = block.check(paper, fctx)
        if outcome.passed:
            passed.append(paper)
        else:
            rejected.append((paper, outcome))
    return passed, rejected


def record_rejections(
    rejected: list[tuple[PaperRecord, FilterOutcome]],
    fctx: FilterContext,
) -> None:
    ctx = fctx.ctx
    for paper, outcome in rejected:
        ctx.rejected.add(paper.paper_id)
        key = outcome.category or "unknown"
        ctx.rejection_counts[key] = ctx.rejection_counts.get(key, 0) + 1
