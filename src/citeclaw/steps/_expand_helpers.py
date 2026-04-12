"""Shared helpers for the ``ExpandBy*`` family.

The three ``ExpandBy*`` step files (``ExpandBySearch``,
``ExpandBySemantics``, ``ExpandByAuthor``) historically duplicated
the same screening pipeline boilerplate: idempotency-fingerprint
generation, the ``ctx.searched_signals`` short-circuit, hydration of
raw paper hits via ``ctx.s2.enrich_batch``, abstract enrichment,
``ctx.seen`` deduplication, source-less ``FilterContext`` construction,
``apply_block`` + ``record_rejections``, and ``ctx.collection`` commit.

The retriever (the *one* line that varies between the three) stays in
each step's ``run()`` method, but everything around it now lives here.

Two free helpers + one named-tuple result:

  - :func:`fingerprint_signal` — stable sha256 over the input signal
    plus arbitrary step-specific extras. Each step decides what
    parameters affect its idempotency key.

  - :func:`check_already_searched` — returns a noop ``StepResult`` if
    the fingerprint is already in ``ctx.searched_signals``, else
    ``None``. Caller does ``if (r := check_already_searched(...)):
    return r`` after computing the fingerprint.

  - :func:`screen_expand_candidates` — runs hydrate → enrich → dedup →
    screen → commit in one call. Takes the raw hits list (whatever the
    retriever returned), the source label, the screener block, and an
    optional ``post_hydrate_fn`` for steps that need to trim the
    hydrated list before screening (``ExpandBySearch`` uses this for
    ``apply_local_query``). Returns an :class:`ExpandScreenResult`
    carrying the per-stage counts so each step can fold them into its
    ``stats`` dict.

This is a pure refactor — no behaviour changes — but it removes ~80%
of the duplicated code across the three step files. The user-facing
YAML surface is unchanged: three step names, three retrievers, one
shared screening pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_helpers")


# ---------------------------------------------------------------------------
# Fingerprint + idempotency
# ---------------------------------------------------------------------------


def fingerprint_signal(step_name: str, signal: list[PaperRecord], **extra: Any) -> str:
    """Stable sha256 over (step name, sorted signal IDs, **extra).

    ``extra`` is the step-specific knob set that determines whether
    re-running on the same signal should re-do work or short-circuit.
    For example, ``ExpandBySearch`` includes the ``AgentConfig`` dict
    so changing ``max_iterations`` invalidates a prior fingerprint.
    """
    signal_ids = sorted(p.paper_id for p in signal if p.paper_id)
    payload: dict[str, Any] = {"step": step_name, "signal_ids": signal_ids}
    payload.update(extra)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def check_already_searched(
    step_name: str,
    fingerprint: str,
    ctx: Any,
    signal_len: int,
) -> StepResult | None:
    """Return a noop ``StepResult`` if ``fingerprint`` already in
    ``ctx.searched_signals``, else ``None``.

    Usage:
        fp = fingerprint_signal(self.name, signal, **knobs)
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip
    """
    if fingerprint in ctx.searched_signals:
        log.info(
            "%s: signal fingerprint already searched, skipping (no-op)", step_name,
        )
        return StepResult(
            signal=[],
            in_count=signal_len,
            stats={"reason": "already_searched", "fingerprint": fingerprint[:12]},
        )
    return None


# ---------------------------------------------------------------------------
# Shared screening pipeline
# ---------------------------------------------------------------------------


@dataclass
class ExpandScreenResult:
    """Per-stage counts returned by :func:`screen_expand_candidates`.

    Steps can fold ``base_stats`` into their own stats dict alongside
    retriever-specific keys (anchor_count, raw_hits, agent_iterations
    etc.) before returning a :class:`StepResult`.
    """

    hydrated: list[PaperRecord]
    novel: list[PaperRecord]
    passed: list[PaperRecord]
    rejected: list[Any]  # list[tuple[PaperRecord, FilterOutcome]]
    base_stats: dict[str, Any]


def screen_expand_candidates(
    *,
    raw_hits: list[Any] | None,
    source_label: str,
    screener: Any,
    ctx: Any,
    post_hydrate_fn: Callable[[list[PaperRecord]], list[PaperRecord]] | None = None,
) -> ExpandScreenResult:
    """Run the shared post-retrieval pipeline for any ``ExpandBy*`` step.

    Steps:
      1. Pull ``paperId`` from each raw hit (works for both S2 dicts
         and pre-built ``PaperRecord``-shaped dicts) and call
         ``ctx.s2.enrich_batch`` to hydrate them into PaperRecords.
      2. Fill missing abstracts via ``ctx.s2.enrich_with_abstracts``.
      3. Optionally apply ``post_hydrate_fn`` (used by ExpandBySearch
         for the ``apply_local_query`` post-fetch trim).
      4. Dedup against ``ctx.seen`` and stamp ``rec.source = source_label``
         on every novel record. Adds them to ``ctx.seen``.
      5. Build a **source-less** ``FilterContext`` (PC-05 invariant —
         every ExpandBy* step's screener cascade must tolerate
         ``fctx.source=None``).
      6. ``apply_block`` the screener.
      7. ``record_rejections`` for the rejected ones (writes to
         ``ctx.rejection_counts`` and ``ctx.rejection_ledger``).
      8. Stamp ``llm_verdict="accept"`` on the survivors and add them
         to ``ctx.collection``.

    The caller's ``run()`` method is then responsible only for the
    retriever (the one line that varies) and for marking the
    fingerprint in ``ctx.searched_signals``.
    """
    # 1. Pull paperIds from raw hits and hydrate.
    candidates: list[dict[str, Any]] = []
    for hit in raw_hits or []:
        if isinstance(hit, dict):
            pid = hit.get("paperId")
        else:
            pid = getattr(hit, "paper_id", None) or getattr(hit, "paperId", None)
        if isinstance(pid, str) and pid:
            candidates.append({"paper_id": pid})

    hydrated: list[PaperRecord] = (
        ctx.s2.enrich_batch(candidates) if candidates else []
    )

    # 2. Fill in abstracts so any title_abstract LLMFilter has text to read.
    if hydrated:
        try:
            ctx.s2.enrich_with_abstracts(hydrated)
        except Exception as exc:  # noqa: BLE001 — abstracts are best-effort
            log.info("enrich_with_abstracts failed: %s", exc)

    # 3. Optional post-hydrate trim (apply_local_query for ExpandBySearch).
    if post_hydrate_fn is not None and hydrated:
        try:
            hydrated = post_hydrate_fn(hydrated)
        except Exception as exc:  # noqa: BLE001 — fall back to untrimmed
            log.warning("post_hydrate_fn failed: %s; using untrimmed list", exc)

    # 4. Dedup against ctx.seen + stamp source.
    new_records: list[PaperRecord] = []
    for rec in hydrated:
        if not rec.paper_id:
            continue
        if rec.paper_id in ctx.seen:
            continue
        rec.source = source_label
        new_records.append(rec)
        ctx.seen.add(rec.paper_id)

    # 5-7. Screening (skip when no screener — accept all novel records).
    if screener is not None:
        fctx = FilterContext(
            ctx=ctx, source=None, source_refs=None, source_citers=None,
        )
        passed, rejected = apply_block(new_records, screener, fctx)
        record_rejections(rejected, fctx)
    else:
        passed = new_records
        rejected = []

    # 8. Survivors → collection (respect max_papers_total).
    max_total = ctx.config.max_papers_total
    committed: list[PaperRecord] = []
    for p in passed:
        if len(ctx.collection) >= max_total:
            break
        p.llm_verdict = "accept"
        ctx.collection[p.paper_id] = p
        committed.append(p)

    return ExpandScreenResult(
        hydrated=hydrated,
        novel=new_records,
        passed=committed,
        rejected=rejected,
        base_stats={
            "raw_hits": len(raw_hits or []),
            "hydrated": len(hydrated),
            "novel": len(new_records),
            "accepted": len(passed),
            "rejected": len(rejected),
        },
    )
