"""HumanInTheLoop — interactive screener-quality check (PD-02, v2).

This step is the user-facing checkpoint of the pipeline: somewhere
late in the YAML, after the bulk of the LLM screening has accumulated
a meaningful rejection ledger, the user can opt in to a short
session where they get shown a balanced sample of accepted +
rejected papers and asked "is this in scope?". The step then
computes per-LLM-filter agreement against the user's labels and
writes ``hitl_report.json``. If the user clicks "stop" at the end,
the pipeline short-circuits to ``Finalize`` instead of continuing.

Behaviour vs. v1 (PD-02 first pass):

  * **Opt-in**: gated on ``enabled: true`` in YAML so an unattended
    run never blocks on input.
  * **Wait until N minutes elapsed**: the step has a ``min_delay_sec``
    knob (default 180s = 3 min). When the step is reached in the
    pipeline, if less than that time has passed since
    ``ctx.pipeline_started_at``, the step sleeps the remainder. This
    lets the user place HITL early in the YAML — the pipeline still
    runs uninterrupted for a meaningful warm-up period before it
    pauses for input.
  * **First-prompt deadline**: ``first_prompt_timeout_sec`` (default
    60s = 1 min). The first prompt is shown via a thread-backed input
    helper with a wallclock deadline. If the user doesn't engage at
    all within that window, the step bails with a 0-label report and
    the pipeline keeps running. Once the user responds to the first
    prompt, subsequent prompts use plain blocking input (no per-prompt
    deadline).
  * **Multi-bucket sampling**: when ``balance_by_filter`` is set, every
    LLM rejection category in a paper's ledger gets credit for that
    paper, not just the first one (A1 fix). So a paper rejected by
    both ``llm_title`` and ``llm_abstract`` is in both buckets and
    has a chance of being sampled under either.
  * **Sample-then-hydrate**: candidates are picked by paper id from
    the ledger first, and S2 metadata is only fetched for the chosen
    sample (A5 fix). On a corpus with 17K rejections this is the
    difference between 17K wasted cache lookups and ~10 useful ones.
  * **Accurate per-filter agreement**: a filter's agreement % is
    computed only over papers the filter actually saw (read from
    ``ctx.papers_screened_by_filter``), not over the whole sample
    (A2 fix). A filter that ran second in a Sequential and never saw
    papers the first filter rejected won't have its score inflated by
    "accidental kept" cases.
  * **LLM-accepted only**: when ``include_accepted=True``, the step
    only samples from papers some LLM filter actually accepted (read
    from ``ctx.papers_accepted_by_filter``), not from papers that
    survived only on cheap rules like ``YearFilter``. (A3 fix)
  * **Stop signal**: at the end of the labelling session the user is
    asked whether to continue. If they say no, the step returns
    ``StepResult(stop_pipeline=True)`` and the pipeline runner
    short-circuits to Finalize. (A6 fix)

The step is **non-destructive**: ``StepResult.signal`` is the input
signal verbatim regardless of user labels.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.human_in_the_loop")


def _ask_with_deadline(prompt: str, deadline_sec: float) -> bool | None:
    """Run ``rich.prompt.Confirm.ask`` in a daemon thread with a wallclock
    deadline. Returns ``True``/``False`` on user response, ``None`` on
    timeout. The reading thread is daemonised so the orphaned thread
    exits when the main process does.

    Catches ``KeyboardInterrupt`` / ``EOFError`` from the worker and
    returns ``None`` (treated as a "no engagement" timeout).
    """
    from rich.prompt import Confirm

    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            ans = Confirm.ask(prompt)
            result_q.put(("ok", ans))
        except (KeyboardInterrupt, EOFError):
            result_q.put(("timeout", None))
        except Exception as exc:  # noqa: BLE001 — thread mustn't crash main
            result_q.put(("error", exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        kind, value = result_q.get(timeout=deadline_sec)
    except queue.Empty:
        return None
    if kind == "ok":
        return bool(value)
    return None


class HumanInTheLoop:
    """Interactive screener-quality check via the rich CLI."""

    name = "HumanInTheLoop"

    def __init__(
        self,
        *,
        enabled: bool = False,
        min_delay_sec: int = 180,
        first_prompt_timeout_sec: int = 60,
        k: int = 10,
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
        if min_delay_sec < 0:
            raise ValueError(
                f"min_delay_sec must be ≥ 0; got {min_delay_sec}"
            )
        if first_prompt_timeout_sec < 1:
            raise ValueError(
                f"first_prompt_timeout_sec must be ≥ 1; got {first_prompt_timeout_sec}"
            )
        self.enabled = enabled
        self.min_delay_sec = min_delay_sec
        self.first_prompt_timeout_sec = first_prompt_timeout_sec
        self.k = k
        self.include_accepted = include_accepted
        self.include_rejected = include_rejected
        self.balance_by_filter = balance_by_filter
        self._seed = seed

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _build_rejected_index(
        self, ctx,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Walk ``ctx.rejection_ledger`` and return:

          * ``flat_ids`` — every paper id with at least one ``llm_*``
            rejection category, in deterministic ledger order.
          * ``by_filter`` — paper ids grouped by every ``llm_*``
            rejection category they belong to. A paper rejected by both
            ``llm_title_llm`` and ``llm_abstract_llm`` appears in BOTH
            buckets (A1 fix).

        No S2 metadata is fetched here — sampling happens against ids
        only and the chosen sample is hydrated afterwards (A5 fix).
        """
        flat_ids: list[str] = []
        seen_in_flat: set[str] = set()
        by_filter: dict[str, list[str]] = {}
        for pid, categories in ctx.rejection_ledger.items():
            llm_cats = [
                c for c in categories
                if isinstance(c, str) and c.startswith("llm_")
            ]
            if not llm_cats:
                continue
            if pid not in seen_in_flat:
                seen_in_flat.add(pid)
                flat_ids.append(pid)
            for cat in llm_cats:
                by_filter.setdefault(cat, []).append(pid)
        return flat_ids, by_filter

    def _llm_accepted_ids(self, ctx) -> list[str]:
        """Union of every paper id any LLM filter accepted (A3 fix).

        Restricts the accepted-pool sample to LLM-decided papers,
        excluding papers that survived only on hard rules like
        YearFilter / CitationFilter — the user wants to validate the
        LLM's judgement, not the cheap pre-filters.
        """
        out: set[str] = set()
        for pid_set in ctx.papers_accepted_by_filter.values():
            out.update(pid_set)
        # Restrict to papers that are still in the collection (ReScreen
        # may have removed some).
        return [pid for pid in out if pid in ctx.collection]

    def _sample_ids(
        self, ctx, rng: random.Random,
    ) -> list[str]:
        """Pick the k paper ids to label, mixing LLM-accepted +
        LLM-rejected per the configured knobs."""
        accepted_ids = (
            self._llm_accepted_ids(ctx) if self.include_accepted else []
        )
        rejected_ids: list[str] = []
        rejected_by_filter: dict[str, list[str]] = {}
        if self.include_rejected:
            rejected_ids, rejected_by_filter = self._build_rejected_index(ctx)

        if self.include_accepted and self.include_rejected:
            half = self.k // 2
            other_half = self.k - half
        elif self.include_accepted:
            half = 0
            other_half = self.k
        else:
            half = self.k
            other_half = 0

        # ---- Sample rejected (multi-bucket if balance_by_filter) ----
        sampled_rejected: list[str] = []
        if half > 0 and rejected_ids:
            if self.balance_by_filter and rejected_by_filter:
                n_filters = len(rejected_by_filter)
                per_filter = max(1, half // n_filters) if n_filters else half
                chosen: set[str] = set()
                for cat in sorted(rejected_by_filter):
                    group = list(rejected_by_filter[cat])
                    rng.shuffle(group)
                    for pid in group:
                        if pid in chosen:
                            continue
                        chosen.add(pid)
                        sampled_rejected.append(pid)
                        if len([p for p in sampled_rejected if p in rejected_by_filter[cat]]) >= per_filter:
                            break
                # Top up from the leftover pool if we're under-budget.
                if len(sampled_rejected) < half:
                    leftovers = [p for p in rejected_ids if p not in chosen]
                    rng.shuffle(leftovers)
                    sampled_rejected.extend(leftovers[: half - len(sampled_rejected)])
                sampled_rejected = sampled_rejected[:half]
            else:
                pool = list(rejected_ids)
                rng.shuffle(pool)
                sampled_rejected = pool[:half]

        # ---- Sample accepted ----
        sampled_accepted: list[str] = []
        if other_half > 0 and accepted_ids:
            pool = list(accepted_ids)
            rng.shuffle(pool)
            sampled_accepted = pool[:other_half]

        candidates = sampled_accepted + sampled_rejected
        rng.shuffle(candidates)
        return candidates

    def _hydrate_sample(
        self, paper_ids: list[str], ctx,
    ) -> list[PaperRecord]:
        """Fetch S2 metadata for ONLY the sampled paper ids (A5 fix).

        Cache-first via ``ctx.s2.fetch_metadata``. Hydration failures
        are silently dropped — the rest of the sample still gets
        labelled.
        """
        out: list[PaperRecord] = []
        for pid in paper_ids:
            existing = ctx.collection.get(pid)
            if existing is not None:
                out.append(existing)
                continue
            try:
                rec = ctx.s2.fetch_metadata(pid)
            except Exception as exc:  # noqa: BLE001 — degrade silently
                log.debug(
                    "HITL: hydrate(%s) failed, dropping: %s", pid[:20], exc,
                )
                continue
            out.append(rec)
        return out

    # ------------------------------------------------------------------
    # CLI presentation
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
    ) -> tuple[dict[str, bool], int, bool]:
        """Walk the candidate list, collecting bool labels.

        First prompt has a wallclock ``first_prompt_timeout_sec``
        deadline (A4 fix) — if the user doesn't engage at all in that
        window, the step bails with no labels. Subsequent prompts use
        regular blocking ``Confirm.ask`` (no per-prompt deadline) since
        the user has already shown they're at the keyboard.

        Returns ``(labels, timeouts, stop_requested)``. ``stop_requested``
        comes from a final continue/stop prompt at the end of the
        labelling session (A6 fix).
        """
        from rich.prompt import Confirm

        labels: dict[str, bool] = {}
        timeouts = 0
        stop_requested = False

        if not candidates:
            return labels, timeouts, stop_requested

        # First prompt — threaded with deadline.
        first = candidates[0]
        first_prompt = self._format_prompt(first, 1, len(candidates))
        first_ans = _ask_with_deadline(
            first_prompt, deadline_sec=float(self.first_prompt_timeout_sec),
        )
        if first_ans is None:
            log.info(
                "HITL: first prompt timed out (no response in %ds) — "
                "skipping HITL session entirely",
                self.first_prompt_timeout_sec,
            )
            return labels, 1, False
        labels[first.paper_id] = bool(first_ans)

        # Subsequent prompts — plain blocking input.
        for i, paper in enumerate(candidates[1:], start=2):
            prompt_text = self._format_prompt(paper, i, len(candidates))
            try:
                ans = Confirm.ask(prompt_text)
            except (KeyboardInterrupt, EOFError):
                log.info("HITL: user interrupted; ending label collection")
                break
            labels[paper.paper_id] = bool(ans)

        # Stop-pipeline prompt at end of session (A6 fix).
        try:
            stop = Confirm.ask(
                "Stop the pipeline after this step? (No will continue normally)",
                default=False,
            )
        except (KeyboardInterrupt, EOFError):
            stop = False
        stop_requested = bool(stop)

        return labels, timeouts, stop_requested

    # ------------------------------------------------------------------
    # Web-mode label collection (PE-09)
    # ------------------------------------------------------------------

    def _collect_labels_web(
        self,
        candidates: list[PaperRecord],
        ctx,
    ) -> tuple[dict[str, bool], int, bool]:
        """Emit an ``hitl_request`` event and block until the web backend
        writes labels into ``ctx.hitl_gate``.

        Returns ``(labels, timeouts, stop_requested)`` matching the same
        shape as ``_collect_labels`` so the rest of ``run()`` is agnostic
        to the input mode.
        """
        gate = ctx.hitl_gate

        # Build paper summaries for the frontend.
        paper_dicts = [
            {
                "paper_id": p.paper_id,
                "title": p.title or "<no title>",
                "venue": p.venue or "?",
                "year": p.year,
                "abstract": (p.abstract or "")[:400],
            }
            for p in candidates
        ]

        # Reset gate state for this request.
        gate.event.clear()
        gate.labels.clear()
        gate.stop_requested = False

        # Emit the request — the web backend forwards this to WebSocket
        # subscribers who render the HitlModal.
        run_id = getattr(ctx, "run_id", "unknown")
        ctx.event_sink.hitl_request(run_id, paper_dicts)

        log.info(
            "HITL web mode: emitted hitl_request with %d papers, "
            "waiting up to %.0fs for user labels",
            len(candidates), gate.timeout_sec,
        )

        # Block until the backend POST handler sets the event.
        responded = gate.event.wait(timeout=gate.timeout_sec)
        if not responded:
            log.warning(
                "HITL web mode: timed out after %.0fs with no response",
                gate.timeout_sec,
            )
            return {}, 1, False

        return dict(gate.labels), 0, gate.stop_requested

    # ------------------------------------------------------------------
    # Per-filter agreement (A2 fix)
    # ------------------------------------------------------------------

    def _compute_agreement(
        self,
        candidates: list[PaperRecord],
        labels: dict[str, bool],
        ctx,
    ) -> dict[str, float]:
        """Per-LLM-filter agreement on the labelled subset.

        For each LLM filter that ran on at least one labelled
        candidate, agreement = fraction of papers where the filter's
        decision (kept iff in ``ctx.papers_accepted_by_filter[cat]``)
        matches the user's label. Crucially, papers the filter never
        saw (because an upstream Sequential filter dropped them first)
        are NOT counted (A2 fix).
        """
        labelled_ids = set(labels.keys())
        if not labelled_ids:
            return {}
        agreement: dict[str, float] = {}
        for cat, screened_ids in ctx.papers_screened_by_filter.items():
            if not isinstance(cat, str) or not cat.startswith("llm_"):
                continue
            accepted_ids = ctx.papers_accepted_by_filter.get(cat, set())
            seen_labelled = labelled_ids & screened_ids
            if not seen_labelled:
                continue
            correct = 0
            for pid in seen_labelled:
                user_keep = labels[pid]
                filter_kept = pid in accepted_ids
                if filter_kept == user_keep:
                    correct += 1
            agreement[cat] = correct / len(seen_labelled)
        return agreement

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if not self.enabled:
            return StepResult(
                signal=signal,
                in_count=len(signal),
                stats={"reason": "disabled"},
            )

        # 1. Wait until min_delay_sec has elapsed since pipeline start.
        # If the previous steps already exceeded that, no wait.
        started = getattr(ctx, "pipeline_started_at", None)
        wait_for = 0.0
        if started is not None and self.min_delay_sec > 0:
            elapsed = time.monotonic() - started
            wait_for = max(0.0, self.min_delay_sec - elapsed)
        if wait_for > 0:
            log.info(
                "HITL: waiting %.0fs for pipeline state to accumulate "
                "(min_delay_sec=%d)",
                wait_for, self.min_delay_sec,
            )
            try:
                ctx.dashboard.warn(
                    f"HumanInTheLoop sleeping {int(wait_for)}s before sampling"
                )
            except Exception as exc:  # noqa: BLE001 — dashboard is best-effort
                # Audit "no silent failure" rule — dashboard write
                # failure isn't fatal but the trail belongs in DEBUG so
                # postmortem can see why the user banner didn't appear.
                log.debug("HITL: dashboard.warn failed: %s", exc)
            time.sleep(wait_for)

        # 2. Sample paper ids first (A5: no hydration before sampling).
        rng = random.Random(self._seed)
        sampled_ids = self._sample_ids(ctx, rng)
        if not sampled_ids:
            return StepResult(
                signal=signal,
                in_count=len(signal),
                stats={"reason": "no_candidates"},
            )

        # 3. Hydrate ONLY the chosen sample.
        candidates = self._hydrate_sample(sampled_ids, ctx)
        if not candidates:
            return StepResult(
                signal=signal,
                in_count=len(signal),
                stats={"reason": "hydration_failed"},
            )

        # 4. Collect labels — web mode or CLI mode.
        if ctx.hitl_gate is not None and ctx.event_sink is not None:
            labels, timeouts, stop_requested = self._collect_labels_web(
                candidates, ctx,
            )
        else:
            labels, timeouts, stop_requested = self._collect_labels(candidates)

        # 5. Compute per-filter agreement (A2 fix uses screened_by_filter).
        agreement = self._compute_agreement(candidates, labels, ctx)
        low_agreement = sorted(
            cat for cat, score in agreement.items() if score < 0.5
        )

        # 6. Write report.
        report: dict[str, Any] = {
            "candidates": len(candidates),
            "labels_collected": len(labels),
            "timeouts": timeouts,
            "stop_requested": stop_requested,
            "agreement_by_filter": agreement,
            "low_agreement_filters": low_agreement,
            "config": {
                "enabled": self.enabled,
                "min_delay_sec": self.min_delay_sec,
                "first_prompt_timeout_sec": self.first_prompt_timeout_sec,
                "k": self.k,
                "include_accepted": self.include_accepted,
                "include_rejected": self.include_rejected,
                "balance_by_filter": self.balance_by_filter,
            },
        }
        report_path: Path = ctx.config.data_dir / "hitl_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, default=str))
        except OSError as exc:
            log.warning("HITL: could not write %s: %s", report_path, exc)

        return StepResult(
            signal=signal,
            in_count=len(signal),
            stats={
                "candidates": len(candidates),
                "labels_collected": len(labels),
                "timeouts": timeouts,
                "low_agreement_filters": len(low_agreement),
                "stop_requested": stop_requested,
                "report_path": str(report_path),
            },
            stop_pipeline=stop_requested,
        )
