"""``ShapeLog`` — PyTorch-summary-style in/out shape table for pipeline runs.

Every step produces an :class:`~citeclaw.steps.base.StepResult` whose
``in_count`` / ``len(signal)`` give one row, and the runner tracks how
much ``ctx.collection`` grew during the step (``delta_coll``). The
``ShapeLog`` accumulates those rows and emits two views:

* :meth:`to_dicts` — list of plain dicts suitable for the
  ``shape_summary.json`` artefact written by Finalize. Branch
  sub-step rows are nested under ``stats["branches"]`` (already
  JSON-friendly per :class:`~citeclaw.steps.parallel.Parallel.run`).
* :meth:`render` — pretty ASCII table for ``shape_summary.txt`` and
  the live dashboard. Branch sub-rows print as ``  └ branch[i]``
  groups so users can see what each branch did inside a
  :class:`~citeclaw.steps.parallel.Parallel`.

Notes column is truncated to 80 chars to keep the table usable on a
80-column terminal; the full stats dict survives in
:meth:`to_dicts`'s output for downstream consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Notes column hard-cap so the rendered table stays usable on an
# 80-column terminal even when steps emit verbose stats dicts.
_NOTES_MAX_CHARS = 80


@dataclass
class _Row:
    """One per-step entry: name, in/out counts, collection delta, stats dict."""

    name: str
    in_count: int
    out_count: int
    delta_collection: int
    stats: dict[str, Any] = field(default_factory=dict)


class ShapeLog:
    """Accumulator for per-step shape rows + dual JSON / ASCII renderers."""

    def __init__(self) -> None:
        self.rows: list[_Row] = []

    def record(self, name: str, in_count: int, out_count: int, delta_coll: int, stats: dict) -> None:
        """Append one step's row. Called by the runner after each ``step.run``."""
        self.rows.append(_Row(name, in_count, out_count, delta_coll, stats))

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the shape rows as a list of plain dicts, suitable for
        JSON serialisation alongside the pretty-printed ``render()`` text.

        Branch sub-rows are preserved under ``stats.branches`` unchanged —
        they're already JSON-friendly in :meth:`citeclaw.steps.parallel.Parallel.run`.
        """
        return [
            {
                "step": r.name,
                "in": r.in_count,
                "out": r.out_count,
                "delta_collection": r.delta_collection,
                "stats": dict(r.stats),
            }
            for r in self.rows
        ]

    def render(self) -> str:
        header = f"{'Step':<22}| {'In':>6} | {'Out':>6} | {'Δcoll':>6} | Notes"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in self.rows:
            notes_parts = []
            for k, v in r.stats.items():
                if k in ("branches",):
                    notes_parts.append(f"{k}={len(v) if isinstance(v, list) else v}")
                else:
                    notes_parts.append(f"{k}={v}")
            notes = " ".join(notes_parts)[:_NOTES_MAX_CHARS]
            lines.append(
                f"{r.name:<22}| {r.in_count:>6} | {r.out_count:>6} | {r.delta_collection:+6d} | {notes}"
            )
            # Branch sub-rows
            br = r.stats.get("branches")
            if isinstance(br, list):
                for branch in br:
                    bi = branch.get("branch", "?")
                    lines.append(f"  └ branch[{bi}]")
                    for s in branch.get("steps", []):
                        lines.append(
                            f"      {s['name']:<18}| {s['in']:>6} | {s['out']:>6} |        |"
                        )
        lines.append(sep)
        return "\n".join(lines)
