"""CLI entry point — ``python -m citeclaw``, ``annotate``, and ``rebuild-graph``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from citeclaw.config import SeedPaper, load_settings
from citeclaw.logging_config import setup_logging
from citeclaw.models import BudgetExhaustedError, CiteClawError
from citeclaw.pipeline import build_context, finalize_partial, run_pipeline
from citeclaw.steps.checkpoint import load_checkpoint
from citeclaw.steps.finalize import write_graphs

log = logging.getLogger("citeclaw")


def _build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw")
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--topic", type=str, default=None)
    p.add_argument("--seed", type=str, nargs="+", default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--continue-from", type=Path, default=None, dest="continue_from")
    p.add_argument("-v", "--verbose", action="store_true")
    # Deprecated, accepted but ignored
    p.add_argument("--max-depth", type=int, default=None, help="(deprecated)")
    p.add_argument("--citation-beta", type=float, default=None, help="(deprecated)")
    return p


def _build_annotate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw annotate")
    p.add_argument("graph", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-i", "--instruction", type=str, default=None)
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p


def _build_rebuild_graph_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw rebuild-graph")
    p.add_argument("data_dir", type=Path, help="Data directory from a previous run")
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite the original citation_network.graphml / "
             "collaboration_network.graphml instead of writing .regen variants.",
    )
    return p


def _validate_config(config) -> None:
    errors: list[str] = []
    if not config.seed_papers:
        errors.append("At least one seed paper is required.")
    if not config.pipeline:
        errors.append("'pipeline' section is required.")
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        sys.exit(1)


def _run_snowball(argv: list[str]) -> None:
    parser = _build_run_parser()
    args = parser.parse_args(argv)

    if args.max_depth is not None:
        log.warning("--max-depth is deprecated and ignored")
    if args.citation_beta is not None:
        log.warning("--citation-beta is deprecated and ignored")

    overrides: dict = {}
    if args.topic:
        overrides["topic_description"] = args.topic
    if args.seed:
        overrides["seed_papers"] = [SeedPaper(paper_id=s) for s in args.seed]
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.max_papers is not None:
        overrides["max_papers_total"] = args.max_papers
    if args.model:
        overrides["screening_model"] = args.model

    config = load_settings(args.config, overrides)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=config.data_dir, level=log_level)
    _validate_config(config)

    ctx, s2, cache = build_context(config)

    if args.continue_from is not None:
        try:
            load_checkpoint(ctx, args.continue_from)
        except FileNotFoundError as exc:
            log.error("Checkpoint load failed: %s", exc)
            sys.exit(1)

    try:
        run_pipeline(ctx)
    except BudgetExhaustedError as exc:
        log.warning("Budget exhausted: %s", exc)
        finalize_partial(ctx)
    except CiteClawError as exc:
        log.error("CiteClaw error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial")
        finalize_partial(ctx)
    finally:
        s2.close()
        cache.close()


def _run_annotate(argv: list[str]) -> None:
    from citeclaw.annotate import annotate_graph

    parser = _build_annotate_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)
    output = args.output or args.graph.with_name(args.graph.stem + "_annotated.graphml")
    annotate_graph(
        graph_path=args.graph,
        output_path=output,
        instruction=args.instruction,
        config_path=args.config,
        api_key=args.api_key,
        model_override=args.model,
        limit=args.limit,
    )


def _run_rebuild_graph(argv: list[str]) -> None:
    """Rebuild the citation + collaboration graphs for an existing data dir.

    Useful when the original graphs have been modified in place during
    downstream analysis and the user wants a fresh copy regenerated from
    the same underlying run state (``literature_collection*.json`` +
    ``cache.db``) without re-running the whole pipeline.
    """
    parser = _build_rebuild_graph_parser()
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Load settings. If a config file is provided, honor it; otherwise fall
    # back to a minimal Settings with data_dir set so Cache/S2 point at the
    # right place.
    overrides = {"data_dir": data_dir}
    config = load_settings(args.config, overrides)

    setup_logging(log_dir=config.data_dir, level=logging.INFO)

    ctx, s2, cache = build_context(config)
    try:
        try:
            load_checkpoint(ctx, data_dir)
        except FileNotFoundError as exc:
            log.error(
                "Data directory %s does not contain a valid CiteClaw run: %s",
                data_dir, exc,
            )
            sys.exit(1)

        # load_checkpoint advances ``iteration`` to (prior+1) so continuation
        # runs don't clobber existing artifacts. For a rebuild we want the
        # *existing* iteration number — so rewind by one.
        ctx.iteration = max(1, ctx.iteration - 1)
        # Rebuild is a read-only regeneration: nothing was newly accepted,
        # so clear the continuation-only ``new_seed_ids`` trail.
        ctx.new_seed_ids = []

        suffix = "" if args.force else ".regen"
        write_graphs(ctx, suffix=suffix)
        log.info(
            "Rebuilt graphs in %s (suffix=%r, %d papers)",
            data_dir, suffix, len(ctx.collection),
        )
    finally:
        s2.close()
        cache.close()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "annotate":
        _run_annotate(argv[1:])
        return
    if argv and argv[0] == "rebuild-graph":
        _run_rebuild_graph(argv[1:])
        return
    _run_snowball(argv)


if __name__ == "__main__":
    main()
