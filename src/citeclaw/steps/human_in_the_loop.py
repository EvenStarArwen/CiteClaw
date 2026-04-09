"""HumanInTheLoop — interactive screener-quality check (PD-02, v1).

Samples a balanced mix of accepted + LLM-rejected papers from the
current run, presents each one to the user via ``rich.prompt.Confirm``
("is this in scope?"), then computes per-LLM-filter agreement against
the user labels and writes a JSON report. If any filter's agreement
drops below 50% the step prompts the user for a continue/stop
decision.

The step is **non-destructive** by design: it never adds to or removes
from ``ctx.collection``. Its job is to surface a quality signal — the
user (or the next pipeline run) acts on the report. ``StepResult.signal``
is whatever came in.

v1 simplifications:

  * Rejected paper records are hydrated via ``ctx.s2.fetch_metadata``
    (cache-first). Hydration failures are silently dropped.
  * Per-filter agreement = `(filter decision == user label) / labelled`.
    A paper is "rejected" by a filter iff its rejection_ledger entry
    contains the filter's category (`llm_<filter_name>`); otherwise the
    filter is treated as having accepted it.
  * Timeout handling is best-effort: if ``rich.prompt.Confirm.ask``
    raises ``TimeoutError`` or ``KeyboardInterrupt`` the step logs a
    warning, skips that paper, and moves on. v1 does NOT install a
    SIGALRM handler — callers wanting strict cross-platform timeouts
    can wrap their own.
  * The continue/stop prompt at the end is also a Confirm call. If the
    user opts to stop the step **logs the request** but still returns
    a normal StepResult — pipeline-wide aborts will land in v2.
  * Sampling uses a seeded :class:`random.Random` so test fixtures get
    deterministic shuffles.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.human_in_the_loop")


class HumanInTheLoop:
    """Interactive screener-quality check via the rich CLI."""

    name = "HumanInTheLoop"

    def __init__(
        self,
        *,
        k: int = 10,
        timeout_sec: int = 120,
        include_accepted: bool = True,
        include_rejected: bool = True,
        balance_by_filter: bool = True,
        seed: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"HumanInTheLoop.k must be ≥ 1; got {k}")
        if not (include_accepted or include_rejected):
            raise ValueError(
                "HumanInTheLoop must include at least one of "
                "include_accepted / include_rejected"
            )
        self.k = k
        self.timeout_sec = timeout_sec
        self.include_accepted = include_accepted
        self.include_rejected = include_rejected
        self.balance_by_filter = balance_by_filter
        self._seed = seed

    # ------------------------------------------------------------------
    # Candidate-pool construction
    # ------------------------------------------------------------------

    def _build_rejected_pool(
        self, ctx,
    ) -> tuple[list[PaperRecord], dict[str, list[PaperRecord]]]:
        """Hydrate every rejection-ledger entry whose categories include
        an ``llm_*`` rejection. Returns a flat list AND a dict grouped
        by primary LLM rejection category (for ``balance_by_filter``)."""
        flat: list[PaperRecord] = []
        by_filter: dict[str, list[PaperRecord]] = {}
        for pid, categories in ctx.rejection_ledger.items():
            llm_cats = [c for c in categories if isinstance(c, str) and c.startswith("llm_")]
            if not llm_cats:
                continue
            try:
                rec = ctx.s2.fetch_metadata(pid)
            except Exception as exc:  # noqa: BLE001 — degrade silently
                log.debug(
                    "HITL: fetch_metadata(%s) failed, dropping from pool: %s",
                    pid, exc,
                )
                continue
            flat.append(rec)
            primary = llm_cats[0]
            by_filter.setdefault(primary, []).append(rec)
        return flat, by_filter

    def _sample_candidates(
        self, ctx, rng: random.Random,
    ) -> list[PaperRecord]:
        """Build the k-paper sample mixing accepted + LLM-rejected
        papers per the configured knobs."""
        accepted: list[PaperRecord] = (
            list(ctx.collection.values()) if self.include_accepted else []
        )
        rejected: list[PaperRecord] = []
        rejected_by_filter: dict[str, list[PaperRecord]] = {}
        if self.include_rejected:
            rejected, rejected_by_filter = self._build_rejected_pool(ctx)

        if self.include_accepted and self.include_rejected:
            half = self.k // 2
            other_half = self.k - half
        elif self.include_accepted:
            half = 0
            other_half = self.k
        else:
            half = self.k
            other_half = 0

        # ---- Sample rejected ----
        if half > 0 and rejected:
            if self.balance_by_filter and rejected_by_filter:
                n_filters = len(rejected_by_filter)
                per_filter = max(1, half // n_filters) if n_filters else half
                sampled_rejected: list[PaperRecord] = []
                for cat in sorted(rejected_by_filter):
                    group = list(rejected_by_filter[cat])
                    rng.shuffle(group)
                    sampled_rejected.extend(group[:per_filter])
                # If under-budget (e.g. small groups), top up from the
                # leftover pool.
                if len(sampled_rejected) < half:
                    chosen_ids = {p.paper_id for p in sampled_rejected}
                    leftovers = [p for p in rejected if p.paper_id not in chosen_ids]
                    rng.shuffle(leftovers)
                    sampled_rejected.extend(leftovers[: half - len(sampled_rejected)])
                sampled_rejected = sampled_rejected[:half]
            else:
                pool = list(rejected)
                rng.shuffle(pool)
                sampled_rejected = pool[:half]
        else:
            sampled_rejected = []

        # ---- Sample accepted ----
        if other_half > 0 and accepted:
            pool = list(accepted)
            rng.shuffle(pool)
            sampled_accepted = pool[:other_half]
        else:
            sampled_accepted = []

        candidates = sampled_accepted + sampled_rejected
        rng.shuffle(candidates)
        return candidates

    # ------------------------------------------------------------------
    # CLI presentation + label collection
    # ------------------------------------------------------------------

    def _format_prompt(
        self, paper: PaperRecord, i: int, n: int,
    ) -> str:
        title = paper.title or "<no title>"
        venue = paper.venue or "?"
        year = paper.year if paper.year is not None else "?"
        abstract = (paper.abstract or "")[:400]
        return (
            f"\n[{i}/{n}] {title}\n"
            f"  {venue} · {year}\n"
            f"  {abstract}\n"
            f"Is this paper in scope?"
        )

    def _collect_labels(
        self, candidates: list[PaperRecord],
    ) -> tuple[dict[str, bool], int]:
        """Walk the candidate list with rich.prompt.Confirm.ask. Returns
        ``(labels, timeouts)``. Late-imports rich so monkey-patched
        tests don't need rich installed at module-import time (rich is
        a real dep so this is just defensive)."""
        from rich.prompt import Confirm

        labels: dict[str, bool] = {}
        timeouts = 0
        for i, paper in enumerate(candidates, start=1):
            prompt_text = self._format_prompt(paper, i, len(candidates))
            try:
                verdict = Confirm.ask(prompt_text)
            except (TimeoutError, KeyboardInterrupt) as exc:
                log.warning(
                    "HITL: timeout/interrupt on paper %s — auto-continuing: %s",
                    paper.paper_id[:20], exc,
                )
                timeouts += 1
                continue
            labels[paper.paper_id] = bool(verdict)
        return labels, timeouts

    # ------------------------------------------------------------------
    # Per-filter agreement
    # ------------------------------------------------------------------

    def _compute_agreement(
        self,
        candidates: list[PaperRecord],
        labels: dict[str, bool],
        rejection_ledger: dict[str, list[str]],
    ) -> dict[str, float]:
        """Per-LLM-filter agreement on the labelled subset.

        For each LLM filter category present in the candidate pool,
        compute the fraction of labelled candidates where the filter's
        decision (kept = no rejection_ledger entry, rejected = entry
        present) matches the user's label.
        """
        filter_cats: set[str] = set()
        for paper in candidates:
            for cat in rejection_ledger.get(paper.paper_id, []):
                if isinstance(cat, str) and cat.startswith("llm_"):
                    filter_cats.add(cat)

        agreement: dict[str, float] = {}
        for cat in sorted(filter_cats):
            correct = 0
            total = 0
            for paper in candidates:
                if paper.paper_id not in labels:
                    continue
                user_keep = labels[paper.paper_id]
                filter_rejected = cat in rejection_ledger.get(paper.paper_id, [])
                filter_kept = not filter_rejected
                if filter_kept == user_keep:
                    correct += 1
                total += 1
            agreement[cat] = correct / total if total > 0 else 0.0
        return agreement

    # ------------------------------------------------------------------
    # Continue/stop prompt
    # ------------------------------------------------------------------

    def _maybe_prompt_continue(
        self, agreement: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Returns ``(should_continue, low_agreement_filters)``.

        v1 always returns ``should_continue=True`` even on user "stop"
        — the step is non-destructive and the spec leaves pipeline-wide
        aborts to a future version. The "stop" intent IS logged at
        WARN level so it shows up in shape_summary / log files.
        """
        low = sorted(cat for cat, score in agreement.items() if score < 0.5)
        if not low:
            return True, []
        from rich.prompt import Confirm

        prompt = (
            f"WARNING: filters with agreement < 0.5: {low}\n"
            f"Continue pipeline?"
        )
        try:
            wants_continue = bool(Confirm.ask(prompt))
        except (TimeoutError, KeyboardInterrupt) as exc:
            log.warning(
                "HITL: continue prompt timed out — defaulting to continue: %s", exc,
            )
            wants_continue = True
        if not wants_continue:
            log.warning(
                "HITL: user requested STOP after seeing low-agreement filters %s "
                "— v1 does not abort the pipeline; act on hitl_report.json instead.",
                low,
            )
        return True, low

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        rng = random.Random(self._seed)
        candidates = self._sample_candidates(ctx, rng)
        if not candidates:
            return StepResult(
                signal=signal,
                in_count=len(signal),
                stats={"reason": "no_candidates"},
            )

        labels, timeouts = self._collect_labels(candidates)
        agreement = self._compute_agreement(
            candidates, labels, ctx.rejection_ledger,
        )
        _, low_agreement = self._maybe_prompt_continue(agreement)

        report: dict[str, Any] = {
            "candidates": len(candidates),
            "labels_collected": len(labels),
            "timeouts": timeouts,
            "agreement_by_filter": agreement,
            "low_agreement_filters": low_agreement,
            "config": {
                "k": self.k,
                "timeout_sec": self.timeout_sec,
                "include_accepted": self.include_accepted,
                "include_rejected": self.include_rejected,
                "balance_by_filter": self.balance_by_filter,
            },
        }
        report_path: Path = ctx.config.data_dir / "hitl_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return StepResult(
            signal=signal,
            in_count=len(signal),
            stats={
                "candidates": len(candidates),
                "labels_collected": len(labels),
                "timeouts": timeouts,
                "low_agreement_filters": len(low_agreement),
                "report_path": str(report_path),
            },
        )
