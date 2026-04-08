"""Pretty-printed in/out shape table for the pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Row:
    name: str
    in_count: int
    out_count: int
    delta_collection: int
    stats: dict[str, Any] = field(default_factory=dict)


class ShapeLog:
    def __init__(self) -> None:
        self.rows: list[_Row] = []

    def record(self, name: str, in_count: int, out_count: int, delta_coll: int, stats: dict) -> None:
        self.rows.append(_Row(name, in_count, out_count, delta_coll, stats))

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
            notes = " ".join(notes_parts)[:80]
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
