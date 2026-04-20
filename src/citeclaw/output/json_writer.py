"""JSON output: ``literature_collection.json`` and ``run_state.json``."""

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
    counter: Counter[str] = Counter()
    for p in papers:
        counter[str(getattr(p, field, None))] += 1
    return dict(counter.most_common())


def build_output(
    collection: dict[str, PaperRecord],
    rejected: set[str],
    seen: set[str],
    budget: BudgetTracker,
) -> dict[str, Any]:
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
        "papers": [
            {
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
            for p in sorted_papers
        ],
    }


def write_json(data: dict[str, Any], path: Path) -> None:
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
