"""JSON output: ``literature_collection.json`` and ``run_state.json``.

Two writers, called from :class:`citeclaw.steps.finalize.Finalize`:

* :func:`build_output` + :func:`write_json` — the user-facing
  ``literature_collection.json``: summary block (counts, budget,
  per-field distributions) plus the full sorted paper list.
* :func:`write_run_state` — the ``run_state.json`` checkpoint that
  ``--continue-from`` reads to resume a previous run (collection /
  rejected / seen sets, the still-to-process queue, budget).

Neither writer is atomic — an interrupted process leaves a
partially-written file. The pipeline normally only writes these once
at the very end (Finalize step), so the window is small in practice;
upgrade to a tmp-file + rename if that ever changes.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from citeclaw.budget import BudgetTracker
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.output.json")


def _count_by_field(papers: list[PaperRecord], field: str) -> dict[str, int]:
    """Return ``{value: count}`` for ``getattr(p, field, None)`` over ``papers``.

    Values are stringified (so ``None`` → ``"None"``) so the result is
    JSON-serialisable even for fields that may carry enums / dataclasses.
    Sorted by descending count via :meth:`Counter.most_common`.
    """
    counter: Counter[str] = Counter()
    for p in papers:
        counter[str(getattr(p, field, None))] += 1
    return dict(counter.most_common())


def _paper_to_dict(p: PaperRecord) -> dict[str, Any]:
    """Project a :class:`PaperRecord` to its JSON-friendly dict shape.

    The 16 fields surfaced here form the public contract of
    ``literature_collection.json`` — downstream tools (web UI,
    annotation step, third-party consumers) read by these field names,
    so adding / removing a key is a breaking change for them.
    """
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "abstract": p.abstract,
        "year": p.year,
        "venue": p.venue,
        "citation_count": p.citation_count,
        "influential_citation_count": p.influential_citation_count,
        "references": p.references,
        "depth": p.depth,
        "source": p.source,
        "llm_verdict": p.llm_verdict,
        "llm_reasoning": p.llm_reasoning,
        "supporting_papers": p.supporting_papers,
        "expanded": p.expanded,
        "pdf_url": p.pdf_url,
        "authors": p.authors,
    }


def build_output(
    collection: dict[str, PaperRecord],
    rejected: set[str],
    seen: set[str],
    budget: BudgetTracker,
) -> dict[str, Any]:
    """Build the ``literature_collection.json`` payload.

    Top-level keys are ``summary`` (counts + budget + per-field
    distributions for ``depth`` / ``source`` / ``llm_verdict``) and
    ``papers`` (all accepted papers sorted descending by
    ``citation_count``; missing counts treated as 0).
    """
    sorted_papers = sorted(
        collection.values(), key=lambda p: p.citation_count or 0, reverse=True,
    )
    return {
        "summary": {
            "total_accepted": len(collection),
            "total_rejected": len(rejected),
            "total_seen": len(seen),
            "budget": budget.to_dict(),
            "depth_distribution": _count_by_field(sorted_papers, "depth"),
            "source_distribution": _count_by_field(sorted_papers, "source"),
            "verdict_distribution": _count_by_field(sorted_papers, "llm_verdict"),
        },
        "papers": [_paper_to_dict(p) for p in sorted_papers],
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON (UTF-8, default=str).

    Creates the parent directory if it doesn't exist. ``default=str``
    ensures any unexpected non-JSON-native value (e.g. ``Enum`` /
    ``Path``) round-trips as its ``str()`` form rather than raising.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log.info("Wrote JSON output: %s", path)


def with_iteration_suffix(path: Path, iteration: int) -> Path:
    """Add ``.expN`` suffix to a filename for iteration > 1."""
    if iteration <= 1:
        return path
    return path.with_name(f"{path.stem}.exp{iteration}{path.suffix}")


def write_run_state(
    collection: dict[str, PaperRecord],
    rejected: set[str],
    seen: set[str],
    queue_ids: list[str],
    budget: BudgetTracker,
    path: Path,
    *,
    iteration: int = 1,
    parent_dir: str = "",
    new_seed_ids: list[str] | None = None,
) -> None:
    """Write the ``run_state.json`` checkpoint that ``--continue-from`` reads.

    The eight payload keys (iteration / parent_dir / new_seed_ids /
    collection_ids / rejected_ids / seen_ids / queue_ids / budget) are
    consumed verbatim by :func:`citeclaw.steps.checkpoint.load_state`,
    so renames are breaking changes for anyone with a half-finished run.
    """
    state = {
        "iteration": iteration,
        "parent_dir": parent_dir,
        "new_seed_ids": new_seed_ids or [],
        "collection_ids": list(collection.keys()),
        "rejected_ids": list(rejected),
        "seen_ids": list(seen),
        "queue_ids": queue_ids,
        "budget": budget.to_dict(),
    }
    write_json(state, path)
    log.info("Wrote run state: %s", path)
