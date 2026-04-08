"""Rerank step — non-destructive: score by metric, optional community-aware diversity."""

from __future__ import annotations

import logging

from citeclaw.models import PaperRecord
from citeclaw.rerank.diversity import cluster_diverse_top_k
from citeclaw.rerank.metrics import compute_metric
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.rerank")


class Rerank:
    name = "Rerank"

    def __init__(
        self,
        *,
        metric: str = "citation",
        k: int = 100,
        diversity: dict | str | None = None,
    ) -> None:
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
