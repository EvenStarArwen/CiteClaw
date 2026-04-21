"""``--continue-from`` checkpoint loader.

Reads the literature collection + run state from a previous run's
output directory and seeds a fresh :class:`citeclaw.context.Context`
so the next pipeline pass picks up where the prior one left off.

When the prior run produced multiple snapshots (e.g. via repeated
``--continue-from`` invocations), each successive iteration's output
file gets an ``.expN.`` infix (``literature_collection.exp2.json``,
``run_state.exp3.json``, etc — see
:func:`citeclaw.output.json_writer.with_iteration_suffix`). This loader
picks the highest-N pair so chained continuations resume from the
latest state rather than the original.

Loaded papers are stamped ``expanded=True`` and pre-populated into
``ctx.expanded_forward`` / ``ctx.expanded_backward`` so the next
expansion doesn't re-fetch citers / refs we already have. Missing
``references`` are repopulated from the S2 cache (or the network if
not cached) — DEBUG-logged on per-paper failure so the trail exists
without spamming WARNING in normal use.

Malformed paper entries in the JSON (e.g. schema drift between the
write-time and load-time PaperRecord shapes) are WARNING-logged + skipped
so one bad row doesn't abort the whole resume.
"""

from __future__ import annotations

import json as _json
import logging
import re
from pathlib import Path

from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.progress import phase_done

log = logging.getLogger("citeclaw.stages.checkpoint")

_EXP_RE = re.compile(r"\.exp(\d+)\.")


def _exp_num(p: Path) -> int:
    """Extract the ``.expN.`` iteration number from a filename, or 0 if absent.

    The original (un-suffixed) ``literature_collection.json`` /
    ``run_state.json`` are treated as iteration 0 so they sort below
    any ``.exp1.`` / ``.exp2.`` siblings.
    """
    m = _EXP_RE.search(p.name)
    return int(m.group(1)) if m else 0


def _pick_latest_checkpoint(
    checkpoint_dir: Path, pattern: str, label: str,
) -> Path:
    """Return the highest-``.expN.`` file matching ``pattern`` under ``checkpoint_dir``.

    Used for both ``literature_collection*.json`` and
    ``run_state*.json``. Raises :class:`FileNotFoundError` with a
    descriptive message when no match exists so the CLI surfaces a
    useful error rather than an opaque IndexError.
    """
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {label} found in {checkpoint_dir}")
    return max(candidates, key=_exp_num)


def load_checkpoint(ctx: Context, checkpoint_dir: Path) -> None:
    """Hydrate ``ctx`` from the latest ``.expN.`` snapshot in ``checkpoint_dir``.

    Six-step flow: (1) verify the directory exists; (2) pick the
    highest-N collection + state files; (3) deserialise each paper
    into ``ctx.collection`` / ``ctx.seen`` / ``ctx.expanded_*`` and
    re-stamp seed-source papers into ``ctx.seed_ids`` (per-row
    deserialise failures WARNING-log + skip); (4) merge the
    rejected/seen sets from the run-state; (5) bump
    ``ctx.iteration`` and stash the prior dir; (6) repopulate missing
    references from the S2 cache (DEBUG-log per-paper failures).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    coll_path = _pick_latest_checkpoint(
        checkpoint_dir, "literature_collection*.json", "literature_collection*.json",
    )
    state_path = _pick_latest_checkpoint(
        checkpoint_dir, "run_state*.json", "run_state*.json",
    )

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
