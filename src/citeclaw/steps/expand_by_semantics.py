"""ExpandBySemantics — semantic kNN expansion via S2 Recommendations API.

Two modes, chosen via the YAML ``mode:`` knob:

* ``multi_anchor`` (default, legacy) — ONE
  ``POST /recommendations/v1/papers`` call posts every anchor as
  ``positivePaperIds`` and returns a single shared top-``limit`` list.
  Cheap (one S2 call regardless of signal size) but the response is
  biased toward the anchors that happen to dominate the SPECTER2
  centroid of the batch — papers far from the centroid contribute
  little to the returned neighbours.

* ``per_paper`` — ONE
  ``GET /recommendations/v1/papers/forpaper/{paperId}`` call per anchor
  paper, in parallel (``max_workers``), each returning up to
  ``recs_per_paper`` neighbours. Every accepted paper in the signal
  gets its own neighbourhood, then the union is screened. Costs
  ``len(signal)`` S2 calls so re-runs are pricey, but every paper
  surfaces its own local cluster instead of being averaged out.

The pipeline contract is unchanged: source-less ``FilterContext`` (no
edge anchor), single shared ``screen_expand_candidates`` cascade,
``ctx.searched_signals`` fingerprint for idempotency. Mode is folded
into the fingerprint so switching modes between runs invalidates the
prior signature and forces a fresh expansion.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from citeclaw.models import PaperRecord
from citeclaw.steps._expand_helpers import (
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_semantics")


_VALID_MODES = ("multi_anchor", "per_paper")


class ExpandBySemantics:
    """Semantic kNN expansion via S2 Recommendations API.

    Composable at the same level as ``ExpandForward`` /
    ``ExpandBackward`` / ``ExpandBySearch``. Like ``ExpandBySearch``
    this step has no source paper, so its screener cascade must
    tolerate ``fctx.source=None``.
    """

    name = "ExpandBySemantics"

    def __init__(
        self,
        *,
        screener: Any = None,
        mode: str = "multi_anchor",
        max_anchor_papers: int = 10,
        limit: int = 100,
        use_rejected_as_negatives: bool = False,
        recs_per_paper: int = 10,
        max_workers: int = 4,
    ) -> None:
        """Configure the semantic kNN expansion.

        Parameters
        ----------
        screener:
            Filter block applied to recommendations before they reach
            ``ctx.collection``. ``None`` is a no-op (the step returns
            empty + ``reason="no screener"``).
        mode:
            ``"multi_anchor"`` (default, legacy) or ``"per_paper"`` (new).
            See module docstring for the trade-off.
        max_anchor_papers:
            ``multi_anchor`` only: cap on the number of anchor papers
            packed into the single API call. Ignored in ``per_paper``
            mode (per-paper expansion runs over the entire signal).
        limit:
            ``multi_anchor`` only: API-side cap on the single shared
            recommendation list (S2 default 100). Ignored in
            ``per_paper`` mode.
        use_rejected_as_negatives:
            ``multi_anchor`` only: when True, sends the first 50 papers
            in ``ctx.rejected`` as negative anchors so the kNN steers
            away from already-rejected material. The single-paper
            endpoint does not accept negatives, so this flag is a no-op
            in ``per_paper`` mode.
        recs_per_paper:
            ``per_paper`` only: how many neighbours to request per
            anchor (default 10).
        max_workers:
            ``per_paper`` only: concurrent in-flight S2 calls. The S2
            HTTP layer is globally rate-limited so this only controls
            how aggressively requests overlap on the I/O side; the
            actual rps cap is unaffected.
        """
        self.screener = screener
        self.mode = mode if mode in _VALID_MODES else "multi_anchor"
        if self.mode != mode:
            log.warning(
                "ExpandBySemantics: unknown mode=%r, falling back to "
                "multi_anchor. Valid modes: %s",
                mode, _VALID_MODES,
            )
        self.max_anchor_papers = max_anchor_papers
        self.limit = limit
        self.use_rejected_as_negatives = use_rejected_as_negatives
        self.recs_per_paper = max(1, int(recs_per_paper))
        self.max_workers = max(1, int(max_workers))

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Dispatch to the per-mode runner after the shared early-exit guards.

        Like ``ExpandBySearch``, this step is an *augmentation* (it adds
        kNN neighbours to the candidate pool) rather than a *traversal*
        (which consumes the signal as the set of papers to expand from
        — e.g. ExpandForward / ExpandBackward). So every exit path returns
        ``list(signal) + new_passed`` — losing the input would silently
        kill the downstream snowball when no neighbours pass the screener,
        even though all the iter-N citation-graph survivors are still in
        ``ctx.collection``.
        """
        if self.screener is None:
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        fp = fingerprint_signal(
            self.name, signal,
            mode=self.mode,
            max_anchor_papers=self.max_anchor_papers,
            limit=self.limit,
            use_rejected_as_negatives=self.use_rejected_as_negatives,
            recs_per_paper=self.recs_per_paper,
        )
        if fp in ctx.searched_signals:
            log.info(
                "%s: signal fingerprint already searched, passing input through",
                self.name,
            )
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "already_searched", "fingerprint": fp[:12]},
            )

        if self.mode == "per_paper":
            return self._run_per_paper(signal, ctx, fp)
        return self._run_multi_anchor(signal, ctx, fp)

    # ------------------------------------------------------------------
    # Mode: multi_anchor — legacy single batched call.
    # ------------------------------------------------------------------

    def _run_multi_anchor(
        self, signal: list[PaperRecord], ctx, fp: str,
    ) -> StepResult:
        """One ``POST /recommendations/v1/papers`` over up to
        ``max_anchor_papers`` anchors → shared screening cascade."""
        anchor_ids = [
            p.paper_id for p in signal[: self.max_anchor_papers] if p.paper_id
        ]
        if not anchor_ids:
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no_anchors"},
            )

        negative_ids: list[str] | None = None
        if self.use_rejected_as_negatives and ctx.rejected:
            negative_ids = list(ctx.rejected)[:50]

        try:
            raw = ctx.s2.fetch_recommendations(
                anchor_ids,
                negative_ids=negative_ids,
                limit=self.limit,
            )
        except Exception as exc:  # noqa: BLE001 — keep the pipeline alive
            log.warning("ExpandBySemantics: fetch_recommendations failed: %s", exc)
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "fetch_failed", "error": str(exc)[:120]},
            )

        screened = screen_expand_candidates(
            raw_hits=raw,
            source_label="semantic",
            screener=self.screener,
            ctx=ctx,
        )
        ctx.searched_signals.add(fp)
        return StepResult(
            signal=list(signal) + screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "mode": "multi_anchor",
                "anchor_count": len(anchor_ids),
                "negative_count": len(negative_ids) if negative_ids else 0,
            },
        )

    # ------------------------------------------------------------------
    # Mode: per_paper — one S2 call per signal paper, parallel.
    # ------------------------------------------------------------------

    def _run_per_paper(
        self, signal: list[PaperRecord], ctx, fp: str,
    ) -> StepResult:
        """One ``GET .../forpaper/{id}`` per signal paper (parallel) →
        union of neighbours → shared screening cascade."""
        anchor_ids = [p.paper_id for p in signal if p.paper_id]
        if not anchor_ids:
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={"reason": "no_anchors"},
            )

        dash = getattr(ctx, "dashboard", None)
        if dash is not None:
            try:
                dash.enable_outer_bar(
                    total=len(anchor_ids), description="per-paper kNN",
                )
            except Exception:
                pass

        # Per-anchor S2 fan-out. Failures of individual calls are logged
        # at DEBUG and treated as an empty neighbourhood — one missing
        # paper must not abort the rest of the iteration.
        per_anchor_counts: dict[str, int] = {}
        aggregate: list[dict[str, Any]] = []
        seen_in_step: set[str] = set()
        n_failed = 0

        def _fetch_one(anchor: str) -> tuple[str, list[dict[str, Any]], BaseException | None]:
            try:
                recs = ctx.s2.fetch_recommendations_for_paper(
                    anchor, limit=self.recs_per_paper,
                )
                return anchor, recs, None
            except Exception as exc:  # noqa: BLE001
                return anchor, [], exc

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="semantic-knn",
        ) as pool:
            futures = {pool.submit(_fetch_one, a): a for a in anchor_ids}
            for fut in as_completed(futures):
                anchor, recs, exc = fut.result()
                if exc is not None:
                    n_failed += 1
                    log.debug(
                        "ExpandBySemantics per-paper kNN failed for %s: %s",
                        anchor[:20], exc,
                    )
                else:
                    per_anchor_counts[anchor] = len(recs)
                    for rec in recs:
                        if not isinstance(rec, dict):
                            continue
                        pid = rec.get("paperId")
                        if isinstance(pid, str) and pid and pid not in seen_in_step:
                            seen_in_step.add(pid)
                            aggregate.append(rec)
                if dash is not None:
                    try:
                        dash.advance_outer(1)
                    except Exception:
                        pass

        if not aggregate:
            ctx.searched_signals.add(fp)
            return StepResult(
                signal=list(signal),
                in_count=len(signal),
                stats={
                    "mode": "per_paper",
                    "anchor_count": len(anchor_ids),
                    "recs_per_paper": self.recs_per_paper,
                    "calls_failed": n_failed,
                    "raw_hits": 0,
                    "hydrated": 0,
                    "novel": 0,
                    "accepted": 0,
                    "rejected": 0,
                },
            )

        screened = screen_expand_candidates(
            raw_hits=aggregate,
            source_label="semantic",
            screener=self.screener,
            ctx=ctx,
        )
        ctx.searched_signals.add(fp)
        return StepResult(
            signal=list(signal) + screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "mode": "per_paper",
                "anchor_count": len(anchor_ids),
                "recs_per_paper": self.recs_per_paper,
                "calls_failed": n_failed,
            },
        )
