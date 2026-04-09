"""Pipeline orchestrator: build a Context, run the configured Step list."""

from __future__ import annotations

import logging
import os
import sys
import time

from citeclaw.cache import Cache
from citeclaw.clients.s2 import SemanticScholarClient
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.event_sink import EventSink, NullEventSink
from citeclaw.progress import (
    Dashboard,
    NullDashboard,
    console,
    reset_active_dashboard,
    set_active_dashboard,
)
from citeclaw.steps.finalize import Finalize
from citeclaw.steps.merge_duplicates import MergeDuplicates
from citeclaw.steps.parallel import Parallel
from citeclaw.steps.shape_log import ShapeLog

log = logging.getLogger("citeclaw.pipeline")


# Step name set used by :func:`_warn_similarity_in_sourceless_steps`. These
# are the steps whose ``FilterContext`` carries ``source=None`` (no upstream
# paper for similarity measures to compare against), so any
# ``SimilarityFilter`` in their screener block silently degrades to its
# ``on_no_data`` behaviour.
_SOURCELESS_STEP_NAMES = frozenset({
    "ExpandBySearch", "ExpandBySemantics", "ExpandByAuthor",
})


def build_context(config: Settings) -> tuple[Context, SemanticScholarClient, Cache]:
    budget = BudgetTracker()
    cache = Cache(config.data_dir / "cache.db")
    s2 = SemanticScholarClient(config, cache, budget)
    ctx = Context(config=config, s2=s2, cache=cache, budget=budget)
    return ctx, s2, cache


def _block_contains_similarity_filter(block) -> bool:
    """Recursive walk over a filter block looking for ``SimilarityFilter``.

    Knows how every block compositor stores its children:

      - ``Sequential`` / ``Any_`` use ``layers`` (plural list).
      - ``Not_`` uses ``layer`` (singular).
      - ``Route`` uses ``cases`` (a list of ``RouteCase`` with
        ``predicate`` + ``target`` fields).

    Atom blocks (the leaves) carry no nested filters; the only one we
    care about is ``SimilarityFilter`` itself.
    """
    from citeclaw.filters.blocks.similarity import SimilarityFilter

    if block is None:
        return False
    if isinstance(block, SimilarityFilter):
        return True
    layers = getattr(block, "layers", None)
    if isinstance(layers, list):
        return any(_block_contains_similarity_filter(layer) for layer in layers)
    inner = getattr(block, "layer", None)
    if inner is not None:
        return _block_contains_similarity_filter(inner)
    cases = getattr(block, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            target = getattr(case, "target", None)
            predicate = getattr(case, "predicate", None)
            if _block_contains_similarity_filter(target):
                return True
            if _block_contains_similarity_filter(predicate):
                return True
    return False


def _warn_similarity_in_sourceless_steps(pipeline: list) -> None:
    """Walk the pipeline and warn for every source-less ExpandBy* step
    whose screener contains a ``SimilarityFilter``.

    Source-less steps run the screener with ``fctx.source=None``, so
    every similarity measure (RefSim / CitSim / SemanticSim) returns
    ``None`` and the filter collapses to its ``on_no_data`` behaviour.
    A user reusing a screener block originally written for
    ``ExpandForward`` (with ``on_no_data: reject``) will silently see
    every candidate from every ExpandBy* step rejected. The startup
    warning makes that surprise visible the first time the pipeline runs.

    Recursively descends into ``Parallel`` branches.
    """
    def _walk(steps: list) -> None:
        for step in steps:
            if isinstance(step, Parallel):
                for branch in getattr(step, "branches", []) or []:
                    _walk(branch)
                continue
            if step.name not in _SOURCELESS_STEP_NAMES:
                continue
            screener = getattr(step, "screener", None)
            if screener is None:
                continue
            if _block_contains_similarity_filter(screener):
                log.warning(
                    "%s screener contains a SimilarityFilter. Source-less "
                    "expansion steps have no upstream paper, so RefSim / "
                    "CitSim / SemanticSim all return None and the filter "
                    "collapses to its on_no_data behaviour. If you wrote "
                    "this screener for ExpandForward / ExpandBackward, "
                    "verify the on_no_data setting (default 'pass') is "
                    "still what you want here.",
                    step.name,
                )

    _walk(pipeline)


def _ensure_merge_duplicates(pipeline: list) -> list:
    """Inject a :class:`MergeDuplicates` step before the first :class:`Finalize`
    if the user didn't add one explicitly.

    Deduplicating preprint↔published pairs is a quality bar we want every
    run to meet, not an opt-in. If the pipeline already contains a
    ``MergeDuplicates`` step (anywhere) we leave the order alone — the
    user is presumed to know what they're doing.
    """
    has_merge = any(isinstance(s, MergeDuplicates) for s in pipeline)
    if has_merge:
        return pipeline
    out: list = []
    injected = False
    for step in pipeline:
        if not injected and isinstance(step, Finalize):
            log.debug("auto-injecting MergeDuplicates before Finalize")
            out.append(MergeDuplicates())
            injected = True
        out.append(step)
    if not injected:
        # No Finalize in the pipeline — append MergeDuplicates at the end
        # so dedup still happens before the user reads the artifact dict.
        log.debug("auto-injecting MergeDuplicates at end of pipeline")
        out.append(MergeDuplicates())
    return out


def _build_dashboard(cfg: Settings, n_steps: int) -> Dashboard | NullDashboard:
    """Pick a real Dashboard if stderr is a TTY, else a NullDashboard.

    Set ``CITECLAW_FORCE_DASHBOARD=1`` to force the real Dashboard even when
    running in a non-interactive shell — useful for testing or for running
    via a logger that captures stderr.
    Set ``CITECLAW_NO_DASHBOARD=1`` to force NullDashboard even in a TTY.
    """
    if os.environ.get("CITECLAW_NO_DASHBOARD", "").lower() in ("1", "true", "yes"):
        return NullDashboard()
    force = os.environ.get("CITECLAW_FORCE_DASHBOARD", "").lower() in ("1", "true", "yes")
    if not force and not console.is_terminal:
        return NullDashboard()
    return Dashboard(
        model=cfg.screening_model,
        data_dir=str(cfg.data_dir),
        pipeline_length=n_steps,
        # Display cap; not enforced (real cap is max_llm_tokens / max_s2).
        budget_cap_usd=5.00,
    )


def _describe_step(step) -> str:
    """One-line description for the step header card.

    Pulls a few salient attributes off the step instance so the user sees
    "screen citers, max 30 each" instead of just "ExpandForward".
    """
    name = type(step).__name__
    if name in ("ExpandForward",):
        max_c = getattr(step, "max_citations", "?")
        return f"screen citers (max {max_c} per source) through the configured screener"
    if name in ("ExpandBackward",):
        return "screen references through the configured screener"
    if name in ("ReScreen",):
        return "re-apply screener block to the accumulated collection"
    if name in ("Cluster",):
        algo = getattr(step, "algorithm", None)
        algo_name = algo.get("type") if isinstance(algo, dict) else algo
        return f"cluster the signal · algorithm={algo_name}"
    if name in ("Rerank",):
        m = getattr(step, "metric", "citation")
        k = getattr(step, "k", "?")
        return f"rerank by {m} · top-{k}"
    if name in ("Finalize",):
        return "write JSON / BibTeX / GraphML / collab graph"
    if name in ("LoadSeeds",):
        return "load seed paper metadata from S2"
    if name in ("MergeDuplicates",):
        return "merge preprint↔published duplicates"
    if name in ("Parallel",):
        n = len(getattr(step, "branches", []))
        return f"broadcast signal to {n} branches"
    return ""


def run_pipeline(
    ctx: Context,
    *,
    event_sink: EventSink | None = None,
) -> dict:
    """Run the configured pipeline against ``ctx``.

    PE-03: an optional ``event_sink`` keyword consumes streaming run
    events (``step_start`` / ``paper_added`` / ``step_end`` /
    ``shape_table_update``). Defaults to :class:`NullEventSink` so the
    legacy CLI behavior is unchanged when no caller provides a sink.
    The web backend will pass in a fan-out sink that forwards events
    to WebSocket subscribers.
    """
    cfg = ctx.config
    pipeline = _ensure_merge_duplicates(cfg.pipeline_built or [])
    _warn_similarity_in_sourceless_steps(pipeline)
    sink: EventSink = event_sink if event_sink is not None else NullEventSink()

    # Wallclock anchor for steps that gate on "wait N minutes since
    # pipeline start" — currently only HumanInTheLoop. Set on the
    # context so the step can read it without re-plumbing pipeline state.
    ctx.pipeline_started_at = time.monotonic()

    dashboard = _build_dashboard(cfg, len(pipeline))
    ctx.dashboard = dashboard
    dashboard.attach(ctx)
    cv_token = set_active_dashboard(dashboard)

    # File log retains the legacy "CiteClaw pipeline ..." header even when
    # the dashboard is active, since the file handler stays installed.
    log.debug("=" * 60)
    log.debug("CiteClaw pipeline — model=%s data_dir=%s", cfg.screening_model, cfg.data_dir)
    log.debug("=" * 60)

    dashboard.begin_run()

    signal: list = []
    shape = ShapeLog()
    try:
        for idx, step in enumerate(pipeline, start=1):
            before_coll_ids = set(ctx.collection.keys())
            before_coll = len(before_coll_ids)
            desc = _describe_step(step)

            sink.step_start(idx=idx, name=step.name, description=desc)
            dashboard.begin_step(idx, step.name, desc)
            log.debug("[%s] in=%d", step.name, len(signal))
            try:
                result = step.run(signal, ctx)
            finally:
                dashboard.end_step()

            # Synthesise per-paper paper_added events from the
            # collection delta. Steps that want to emit paper_added in
            # real-time can call into ctx.event_sink directly in v2;
            # for v1 we just compare the keysets at step boundaries.
            after_coll_ids = set(ctx.collection.keys())
            new_ids = after_coll_ids - before_coll_ids
            for pid in sorted(new_ids):
                paper = ctx.collection.get(pid)
                source = getattr(paper, "source", "") if paper is not None else ""
                sink.paper_added(paper_id=pid, source=source)

            delta = len(after_coll_ids) - before_coll
            shape.record(step.name, result.in_count, len(result.signal), delta, result.stats)
            sink.step_end(
                idx=idx,
                name=step.name,
                in_count=result.in_count,
                out_count=len(result.signal),
                delta_collection=delta,
                stats=dict(result.stats),
            )
            log.debug(
                "[%s] in=%d out=%d Δcoll=%+d %s",
                step.name, result.in_count, len(result.signal), delta,
                " ".join(f"{k}={v}" for k, v in result.stats.items() if k != "branches"),
            )
            signal = result.signal
            if getattr(result, "stop_pipeline", False):
                dashboard.warn(
                    f"{step.name} requested pipeline stop — short-circuiting to Finalize"
                )
                if step.name != "Finalize":
                    dashboard.begin_step(idx + 1, "Finalize", _describe_step(Finalize()))
                    try:
                        finalize_result = Finalize().run(signal, ctx)
                    finally:
                        dashboard.end_step()
                    shape.record(
                        "Finalize",
                        finalize_result.in_count,
                        len(finalize_result.signal),
                        0,
                        finalize_result.stats,
                    )
                break
            if ctx.budget.is_exhausted(cfg) or len(ctx.collection) >= cfg.max_papers_total:
                dashboard.warn("Budget/cap reached — stopping early")
                if step.name != "Finalize":
                    dashboard.begin_step(idx + 1, "Finalize", _describe_step(Finalize()))
                    try:
                        finalize_result = Finalize().run(signal, ctx)
                    finally:
                        dashboard.end_step()
                    # Record the bypass-Finalize in the shape table so the
                    # printed summary doesn't truncate at the cap-breaking
                    # step. Without this, runs that hit max_papers_total
                    # mid-pipeline silently drop the Finalize row from the
                    # summary even though Finalize did run and write the
                    # output files.
                    shape.record(
                        "Finalize",
                        finalize_result.in_count,
                        len(finalize_result.signal),
                        0,  # Finalize never adds to the collection
                        finalize_result.stats,
                    )
                break

        rendered = shape.render()
        sink.shape_table_update(rendered_shape=rendered)
        log.debug("\nPipeline shape summary:\n%s", rendered)
        try:
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            (cfg.data_dir / "shape_summary.txt").write_text(rendered + "\n")
        except Exception as exc:
            dashboard.warn(f"could not write shape_summary.txt: {exc}")

        log.debug(
            "Done. accepted=%d rejected=%d seen=%d",
            len(ctx.collection), len(ctx.rejected), len(ctx.seen),
        )
        log.debug("\n%s", ctx.budget.detailed_summary())

        dashboard.finalize()
    finally:
        reset_active_dashboard(cv_token)

    return getattr(ctx, "result", {})


def finalize_partial(ctx: Context) -> None:
    Finalize().run([], ctx)
