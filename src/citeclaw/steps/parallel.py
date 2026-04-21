"""``Parallel`` step — broadcast signal to N branches, union outputs.

Each branch is a sequential list of steps that runs against an
**immutable snapshot** of the input signal — the snapshot is the
key correctness invariant that makes one branch's destructive ops
(e.g. ``Rerank`` truncating to top-K, or ``ReScreen`` removing
papers from ``ctx.collection``) invisible to the other branches'
inputs. ``ctx`` itself is shared across branches (so cumulative
state like ``ctx.collection`` / ``ctx.seen`` does see writes from
every branch in the order branches run); only the per-branch
**signal** is isolated via the snapshot.

Branch outputs are unioned by ``paper_id`` (first occurrence wins
for the dict-merge), so a paper that appears in multiple branches'
output is included exactly once. Sequential execution is intentional —
there's no thread/process pool here. The "parallel" name reflects
the *composition pattern* (alternate paths through the pipeline),
not concurrency.

The runner currently does NOT propagate ``StepResult.stop_pipeline``
from inner steps — branches that signal stop are absorbed by
Parallel. That's a known coverage gap noted in the architectural
audit.
"""

from __future__ import annotations

from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult


class Parallel:
    """Sequentially run N step-lists over a shared snapshot of the signal."""

    name = "Parallel"

    def __init__(self, *, branches: list[list]) -> None:
        """Configure the branch list.

        Parameters
        ----------
        branches:
            A list of step lists. Each inner list is a sequential
            mini-pipeline that receives the snapshotted input signal.
            Empty branches are allowed and just contribute nothing
            to the union.
        """
        self.branches = branches

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Run every branch over the same snapshot, union by ``paper_id``.

        Five-step flow: (1) snapshot the input signal so per-branch
        rerank/rescreen don't poison the next branch's input;
        (2) iterate branches in order, running each step in sequence;
        (3) surface a phase label per step so the dashboard shows
        which branch + sub-step is active; (4) collect the per-branch
        step-counts log into ``stats["branches"]``; (5) dict-merge
        outputs by ``paper_id`` (first occurrence wins) — the union
        becomes the returned signal.
        """
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
