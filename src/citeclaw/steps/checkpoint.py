"""Continuation: load prior literature_collection*.json + run_state*.json."""

from __future__ import annotations

import json as _json
import logging
import re
from pathlib import Path

from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.progress import phase_done

log = logging.getLogger("citeclaw.stages.checkpoint")


def load_checkpoint(ctx: Context, checkpoint_dir: Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def _exp_num(p: Path) -> int:
        m = re.search(r"\.exp(\d+)\.", p.name)
        return int(m.group(1)) if m else 0

    coll_candidates = sorted(checkpoint_dir.glob("literature_collection*.json"))
    if not coll_candidates:
        raise FileNotFoundError(f"No literature_collection*.json found in {checkpoint_dir}")
    coll_path = max(coll_candidates, key=_exp_num)

    state_candidates = sorted(checkpoint_dir.glob("run_state*.json"))
    if not state_candidates:
        raise FileNotFoundError(f"No run_state*.json found in {checkpoint_dir}")
    state_path = max(state_candidates, key=_exp_num)

    log.info("Loading checkpoint: %s + %s", coll_path.name, state_path.name)

    with open(coll_path, encoding="utf-8") as f:
        coll_data = _json.load(f)
    papers_loaded = 0
    for p_dict in coll_data.get("papers", []):
        try:
            rec = PaperRecord(**p_dict)
        except Exception as exc:
            log.warning(
                "Skipping malformed paper in checkpoint (%s): %s",
                p_dict.get("paper_id", "?")[:20], exc,
            )
            continue
        rec.expanded = True
        ctx.collection[rec.paper_id] = rec
        ctx.seen.add(rec.paper_id)
        ctx.expanded_forward.add(rec.paper_id)
        ctx.expanded_backward.add(rec.paper_id)
        if rec.source == "seed":
            ctx.seed_ids.add(rec.paper_id)
        papers_loaded += 1

    with open(state_path, encoding="utf-8") as f:
        state_data = _json.load(f)
    ctx.rejected.update(state_data.get("rejected_ids", []))
    ctx.seen.update(state_data.get("seen_ids", []))

    prior_iter = int(state_data.get("iteration", 1))
    ctx.iteration = prior_iter + 1
    ctx.prior_dir = checkpoint_dir

    missing_refs = [p for p in ctx.collection.values() if not p.references]
    if missing_refs:
        log.info("Repopulating references for %d loaded papers from cache/S2...", len(missing_refs))
        for p in missing_refs:
            try:
                p.references = ctx.s2.fetch_reference_ids(p.paper_id)
            except Exception as exc:
                log.debug("ref fetch failed for %s: %s", p.paper_id[:20], exc)

    phase_done(
        f"Checkpoint loaded: {papers_loaded} papers, "
        f"{len(ctx.rejected)} rejected, {len(ctx.seen)} seen "
        f"→ starting expansion #{ctx.iteration}"
    )
