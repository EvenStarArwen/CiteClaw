"""ReinforceGraph — rescue high-pagerank rejected papers (PD-01, v1).

The rationale: a screener cascade is necessarily imperfect, and some
rejected papers are *structurally important* — papers that the
collection is densely connected to via references and citations. This
step builds a combined citation graph over ``ctx.collection ∪ ctx.seen``,
runs PageRank, picks the rejected papers with the highest scores
(filtered by ``percentile_floor`` and capped at ``top_n``), re-screens
them through a (typically loosened) screener block, and restores any
that pass into ``ctx.collection`` with ``source="reinforced"``.

v1 design notes:

  * The metric is currently fixed to PageRank ('pagerank') — future
    versions may add betweenness, community-aware centrality, or
    learned metrics. The constructor takes ``metric=...`` so v2 can
    extend without breaking callers.
  * The graph is built via :func:`citeclaw.network.build_citation_graph`
    over a *combined* dict that includes ``ctx.collection.values()`` AND
    a hydrated record for every paper in ``ctx.seen \\ ctx.collection``.
    Hydration goes through ``ctx.s2.fetch_metadata`` (cache-first), so
    repeat runs against a warm S2 cache are cheap.
  * Without hydration, rejected papers would be orphan nodes (no
    references → no incoming edges → pagerank = teleport baseline) and
    PageRank couldn't distinguish them. Hydration makes the graph
    structure load-bearing.
  * ``percentile_floor=0.9`` keeps only the top 10% of rejected scores.
    Combined with ``top_n=30``, that bounds the per-run rescue budget.
  * The source-less ``FilterContext(source=None, ...)`` matches the
    contract for the ExpandBy* family — the rescue screener must
    tolerate the source-less mode (audited in PC-05).
  * Restored papers are appended to ``ctx.reinforcement_log`` so the
    final report can show which papers came back via reinforcement.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import build_citation_graph, compute_pagerank
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.reinforce_graph")


class ReinforceGraph:
    """Rescue high-pagerank rejected papers and re-screen them."""

    name = "ReinforceGraph"

    def __init__(
        self,
        *,
        screener: Any = None,
        metric: str = "pagerank",
        top_n: int = 30,
        percentile_floor: float = 0.9,
    ) -> None:
        if metric != "pagerank":
            # v1: only pagerank is supported. Constructor accepts the
            # arg so v2 can add metrics without breaking the YAML schema.
            raise ValueError(
                f"ReinforceGraph v1 only supports metric='pagerank'; got {metric!r}"
            )
        if not (0.0 <= percentile_floor <= 1.0):
            raise ValueError(
                f"percentile_floor must be in [0, 1]; got {percentile_floor}"
            )
        self.screener = screener
        self.metric = metric
        self.top_n = top_n
        self.percentile_floor = percentile_floor

    def _hydrate_rejected(
        self, rejected_ids: set[str], ctx,
    ) -> dict[str, PaperRecord]:
        """Best-effort hydration of every rejected paper id.

        Cache-first via ``ctx.s2.fetch_metadata``. Falls back to a bare
        placeholder when the lookup fails so the graph still includes
        the node (even though it'll be orphaned with no edges).
        """
        out: dict[str, PaperRecord] = {}
        for pid in rejected_ids:
            try:
                out[pid] = ctx.s2.fetch_metadata(pid)
            except Exception as exc:  # noqa: BLE001 — fall back, don't crash
                log.debug(
                    "ReinforceGraph: fetch_metadata(%s) failed, using placeholder: %s",
                    pid, exc,
                )
                out[pid] = PaperRecord(paper_id=pid, source="rejected")
        return out

    def _apply_percentile_floor(
        self,
        rejected_scores: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Drop rejected papers below the ``percentile_floor`` threshold.

        ``rejected_scores`` is sorted descending by score; we compute the
        floor against the full distribution and keep only entries at or
        above it. With floor=0.9 and 10 papers, the floor lands at the
        9th-from-top entry (top 10%). With 0 floor we keep everything.
        """
        if self.percentile_floor <= 0.0 or not rejected_scores:
            return rejected_scores
        scores_only = sorted(s for _, s in rejected_scores)
        idx = int(self.percentile_floor * len(scores_only))
        if idx >= len(scores_only):
            idx = len(scores_only) - 1
        floor_score = scores_only[idx]
        return [(pid, s) for pid, s in rejected_scores if s >= floor_score]

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        # 1. Identify the rejected pool (in seen, not in collection).
        rejected_ids = set(ctx.seen) - set(ctx.collection.keys())
        if not rejected_ids:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no_rejected"},
            )

        # 2. Hydrate rejected papers so the graph has meaningful edges.
        rejected_records = self._hydrate_rejected(rejected_ids, ctx)

        # 3. Build the combined collection used to construct the graph.
        combined: dict[str, PaperRecord] = dict(ctx.collection)
        combined.update(rejected_records)

        # 4. Build graph + compute pagerank.
        graph = build_citation_graph(combined)
        if graph.vcount() == 0:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "empty_graph"},
            )
        ranked = compute_pagerank(graph)
        score_by_id = dict(ranked)

        # 5. Filter to rejected only and sort descending.
        rejected_scores: list[tuple[str, float]] = sorted(
            ((pid, score_by_id.get(pid, 0.0)) for pid in rejected_ids),
            key=lambda x: x[1],
            reverse=True,
        )

        # 6. Apply percentile_floor + top_n cap.
        survivors = self._apply_percentile_floor(rejected_scores)
        candidates = survivors[: self.top_n]
        if not candidates:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "reason": "no_candidates_after_floor",
                    "rejected_pool": len(rejected_ids),
                },
            )

        # 7. Already hydrated above — pull the records into a list.
        candidate_records: list[PaperRecord] = []
        for pid, _score in candidates:
            rec = rejected_records.get(pid)
            if rec is not None and rec.paper_id:
                candidate_records.append(rec)

        # 8. Apply the (typically loose) rescue screener with a
        #    source-less FilterContext — every atom in PC-05's audit
        #    must honour this contract.
        fctx = FilterContext(
            ctx=ctx, source=None, source_refs=None, source_citers=None,
        )
        passed, rejected_again = apply_block(candidate_records, self.screener, fctx)
        record_rejections(rejected_again, fctx)

        # 9. Stamp source="reinforced" and restore to the collection.
        #    Append a one-line entry to ctx.reinforcement_log per rescue
        #    so the final report can show why the paper came back.
        for p in passed:
            p.source = "reinforced"
            p.llm_verdict = "accept"
            ctx.collection[p.paper_id] = p
            ctx.reinforcement_log.append(
                {
                    "paper_id": p.paper_id,
                    "metric": self.metric,
                    "score": float(score_by_id.get(p.paper_id, 0.0)),
                    "reason": "high_pagerank_rescued",
                }
            )

        return StepResult(
            signal=passed,
            in_count=len(signal),
            stats={
                "rejected_pool": len(rejected_ids),
                "after_floor": len(survivors),
                "candidates_screened": len(candidate_records),
                "accepted": len(passed),
                "rejected_again": len(rejected_again),
                "metric": self.metric,
                "percentile_floor": self.percentile_floor,
                "top_n": self.top_n,
            },
        )
