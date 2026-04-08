"""Parallel step — broadcast signal to N branches, union outputs by paper_id."""

from __future__ import annotations

from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult


class Parallel:
    name = "Parallel"

    def __init__(self, *, branches: list[list]) -> None:
        self.branches = branches

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        snapshot = list(signal)
        merged: dict[str, PaperRecord] = {}
        per_branch: list[dict] = []
        dash = ctx.dashboard
        n_branches = len(self.branches)
        for i, branch in enumerate(self.branches):
            cur = list(snapshot)
            steps_log: list[dict] = []
            for j, step in enumerate(branch):
                # Surface a phase label so the dashboard's inner bar shows
                # which sub-step of which branch is running. The sub-step
                # itself will reset begin_phase/enable_outer_bar as it runs.
                dash.begin_phase(
                    f"branch {i + 1}/{n_branches} · {step.name}",
                    total=1,
                )
                res = step.run(cur, ctx)
                cur = res.signal
                steps_log.append({"name": step.name, "in": res.in_count, "out": len(res.signal)})
            per_branch.append({"branch": i, "steps": steps_log})
            for p in cur:
                merged.setdefault(p.paper_id, p)
        out = list(merged.values())
        return StepResult(
            signal=out, in_count=len(snapshot),
            stats={"branches": per_branch, "merged": len(out)},
        )
