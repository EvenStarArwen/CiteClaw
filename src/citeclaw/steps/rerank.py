"""``Rerank`` step — score the signal by a named metric and keep top-K.

Non-destructive: ``ctx.collection`` is never modified — Rerank only
filters the signal it returns. This invariant is what makes
:class:`~citeclaw.steps.parallel.Parallel` work; one branch can
rerank-and-forward while another sees the original input untouched.

Two modes:

* **Plain top-K** (``diversity: None``) — sort by score descending,
  keep the first ``k``.
* **Cluster-aware diversity** (``diversity: {cluster: <store_as>}``
  or any other inline-clusterer config) — delegate to
  :func:`citeclaw.rerank.diversity.cluster_diverse_top_k` for
  floor-then-proportional allocation across clusters.

Available metrics (registered in :mod:`citeclaw.rerank.metrics`):
``"citation"`` (raw count) and ``"pagerank"`` (over the full
``ctx.collection`` graph). Unknown metrics raise :class:`ValueError`
at runtime so YAML typos surface immediately.
"""

from __future__ import annotations

import logging

from citeclaw.models import PaperRecord
from citeclaw.rerank.diversity import cluster_diverse_top_k
from citeclaw.rerank.metrics import compute_metric
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.rerank")


class Rerank:
    """Score-then-truncate step with optional cluster-aware diversity."""

    name = "Rerank"

    def __init__(
        self,
        *,
        metric: str = "citation",
        k: int = 100,
        diversity: dict | str | None = None,
    ) -> None:
        """Configure scoring + truncation.

        Parameters
        ----------
        metric:
            Name of a metric registered in
            :mod:`citeclaw.rerank.metrics` (``"citation"`` /
            ``"pagerank"``). Unknown names raise :class:`ValueError`
            at run time, not at construction — by design, so a
            misconfigured Rerank in a Parallel branch fails when its
            branch runs rather than at pipeline-build.
        k:
            How many top-scored papers to keep.
        diversity:
            ``None`` for plain top-K (score → sort → truncate). A
            ``{"cluster": "<store_as>"}`` dict reuses an upstream
            Cluster step's stored result; any other dict / string is
            forwarded to :func:`citeclaw.cluster.build_clusterer` for
            an inline build. See
            :func:`citeclaw.rerank.diversity.cluster_diverse_top_k`
            for the floor-then-proportional algorithm.
        """
        self.metric = metric
        self.k = k
        self.diversity = diversity  # None = plain top-k

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if not signal:
            return StepResult(
                signal=[],
                in_count=0,
                stats={"metric": self.metric, "k": self.k, "kept": 0},
            )

        dash = ctx.dashboard

        dash.begin_phase(f"compute {self.metric}", total=len(signal))
        scores = compute_metric(self.metric, signal, ctx)
        dash.tick_inner(len(signal))

        if self.diversity is None:
            dash.begin_phase(f"rank top-{self.k}", total=1)
            ranked = sorted(signal, key=lambda p: scores.get(p.paper_id, 0), reverse=True)
            out = ranked[: self.k]
            dash.tick_inner(1)
        else:
            dash.begin_phase("cluster-diverse top-k", total=1)
            out = cluster_diverse_top_k(signal, scores, ctx, self.k, self.diversity)
            dash.tick_inner(1)

        return StepResult(
            signal=out,
            in_count=len(signal),
            stats={
                "metric": self.metric,
                "k": self.k,
                "diversity": self._div_label(),
                "kept": len(out),
            },
        )

    def _div_label(self) -> str:
        if self.diversity is None:
            return "off"
        if isinstance(self.diversity, str):
            return self.diversity
        if "cluster" in self.diversity:
            return f"ref:{self.diversity['cluster']}"
        return self.diversity.get("type", "?")
