"""CLI entry point — ``python -m citeclaw [subcommand] ...``.

Six subcommands, dispatched in :func:`main` by the first positional arg:

* (no arg or anything not below) → :func:`_run_snowball` — the default
  pipeline run that reads ``-c config.yaml`` + flags, validates the
  config, builds a Context, runs the pipeline, and finalises.
* ``annotate <graph>`` → :func:`_run_annotate` — the LLM-driven
  graph-node-labelling subcommand (see :mod:`citeclaw.annotate`).
* ``rebuild-graph <data_dir>`` → :func:`_run_rebuild_graph` —
  re-emit citation_network / collaboration_network GraphML from an
  existing run's literature_collection.json + cache.db (no S2 calls).
* ``fetch-pdfs <data_dir>`` → :func:`_run_fetch_pdfs` — the bulk PDF
  download CLI (see :mod:`citeclaw.fetch_pdfs`).
* ``mainpath <graph>`` → :func:`_run_mainpath` — extract the main path
  subnetwork from a citation GraphML (see :mod:`citeclaw.mainpath`).
* ``web`` → :func:`_run_web` — launch the FastAPI + React web UI
  (see :mod:`citeclaw.web_server`).

API keys are intentionally never read from YAML — :func:`_validate_config`
walks the configured pipeline + filter blocks, computes the set of
required env vars via :func:`citeclaw.preflight.find_missing_api_keys`,
and exits with a clear error before any LLM/S2 spend if any are unset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from citeclaw.config import SeedPaper, load_settings
from citeclaw.logging_config import setup_logging
from citeclaw.models import BudgetExhaustedError, CiteClawError, S2OutageError
from citeclaw.pipeline import build_context, finalize_partial, run_pipeline
from citeclaw.preflight import find_missing_api_keys
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


def _build_mainpath_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw mainpath")
    p.add_argument(
        "graph", type=Path,
        help="Input GraphML — a CiteClaw citation network "
             "(e.g. <data_dir>/citation_network.graphml).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output GraphML path (default: "
             "<stem>_mainpath_<search>_<weight>.graphml next to input). "
             "A sibling .json summary is always written too.",
    )
    p.add_argument(
        "-w", "--weight", default="spc",
        choices=["spc", "splc", "spnp"],
        help="Traversal weight (default: spc). "
             "spc follows Kirchhoff's conservation; splc treats every "
             "paper as a knowledge source; spnp treats every paper as "
             "both source and destination.",
    )
    p.add_argument(
        "-s", "--search", default="key-route",
        choices=["local-forward", "local-backward", "global",
                 "key-route", "multi-local"],
        help="Main-path extraction variant (default: key-route). "
             "key-route guarantees the highest-weighted arc is on the "
             "path; local-forward / local-backward are the classical "
             "priority-first search from sources / sinks; global is "
             "the critical (max-sum-weight) path; multi-local is "
             "local-forward with per-vertex tolerance relaxation.",
    )
    p.add_argument(
        "--cycle", default="shrink",
        choices=["shrink", "preprint"],
        help="How to handle strongly connected components "
             "(default: shrink). shrink collapses each cycle into its "
             "oldest representative paper (Liu, Lu & Ho 2019); "
             "preprint is Batagelj's preprint transform which "
             "preserves SCC members as individual vertices.",
    )
    p.add_argument(
        "--key-routes", type=int, default=1, dest="key_routes",
        help="Number of top-weighted arcs to seed as key routes "
             "when --search=key-route (default: 1).",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.2,
        help="Per-vertex tolerance when --search=multi-local: arcs "
             "with weight >= (1-tolerance) * per_vertex_max are "
             "included (default: 0.2, per Liu & Lu 2012's example).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _build_fetch_pdfs_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw fetch-pdfs")
    p.add_argument(
        "data_dir", type=Path,
        help="CiteClaw data directory containing literature_collection.json",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent download workers (default: 4)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and re-parse PDFs even if they already exist on disk.",
    )
    p.add_argument(
        "--no-refresh-cache", action="store_true",
        help="Skip the refresh of pdf_url from the local S2 cache.db. "
             "By default, papers missing pdf_url in the JSON get rechecked "
             "against cache.db's paper_metadata table.",
    )
    p.add_argument(
        "--no-update-cache", action="store_true",
        help="Skip writing parse outcomes back into cache.db's "
             "paper_full_text table.",
    )
    return p


def _validate_config(config) -> None:
    """Pre-flight check: seeds + pipeline + every required env var.

    Exits with status 1 + a structured error log if any check fails so
    the user sees actionable errors before any LLM / S2 spend. API keys
    are never read from YAML — :func:`citeclaw.preflight.find_missing_api_keys`
    walks the built pipeline + filter blocks to compute the env-var set
    that the configured providers will actually need at runtime.
    """
    errors: list[str] = []
    if not config.seed_papers:
        errors.append("At least one seed paper is required.")
    if not config.pipeline:
        errors.append("'pipeline' section is required.")
    errors.extend(find_missing_api_keys(config))
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        log.error(
            "Set the missing env vars and re-run. "
            "API keys are intentionally never read from YAML.",
        )
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
    except S2OutageError as exc:
        log.error(
            "S2 API appears to be down or rate-limiting hard — %s. "
            "Saving partial collection and exiting.", exc,
        )
        finalize_partial(ctx)
        sys.exit(1)
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


def _run_mainpath(argv: list[str]) -> None:
    """Run main path analysis on a CiteClaw citation GraphML.

    Thin CLI adapter around :func:`citeclaw.mainpath.run_mpa`.
    See :mod:`citeclaw.mainpath` for the algorithmic layer.
    """
    from citeclaw.mainpath import run_mpa

    parser = _build_mainpath_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_dir=None, level=log_level)

    if not args.graph.exists():
        log.error("Graph file not found: %s", args.graph)
        sys.exit(1)

    output = args.output or args.graph.with_name(
        f"{args.graph.stem}_mainpath_{args.search}_{args.weight}.graphml",
    )
    try:
        run_mpa(
            graph_path=args.graph,
            output_path=output,
            weight=args.weight,
            search=args.search,
            cycle=args.cycle,
            key_routes=args.key_routes,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        log.error("mainpath failed: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("mainpath failed: %s", exc)
        sys.exit(1)


def _run_fetch_pdfs(argv: list[str]) -> None:
    """Bulk-download open-access PDFs for a finished CiteClaw run.

    Loads ``literature_collection.json`` from the given data directory
    and writes ``<data_dir>/PDFs/<paper_id>.pdf`` (raw) and
    ``<paper_id>.txt`` (parsed body) for every accepted paper that has
    an ``openAccessPdf.url`` in S2. See :mod:`citeclaw.fetch_pdfs` for
    the implementation.
    """
    from citeclaw.fetch_pdfs import run_fetch_pdfs

    parser = _build_fetch_pdfs_parser()
    args = parser.parse_args(argv)

    setup_logging(log_dir=None, level=logging.INFO)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    try:
        run_fetch_pdfs(
            data_dir,
            max_workers=args.workers,
            overwrite=args.overwrite,
            refresh_from_cache=not args.no_refresh_cache,
            update_cache=not args.no_update_cache,
        )
    except FileNotFoundError as exc:
        log.error("fetch-pdfs failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)


def _build_web_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="citeclaw web")
    p.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port", type=int, default=9999,
        help="Port to listen on (default: 9999)",
    )
    return p


def _run_web(argv: list[str]) -> None:
    """Launch the CiteClaw web UI (FastAPI + React frontend)."""
    from citeclaw.web_server import serve

    parser = _build_web_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=None, level=logging.INFO)
    serve(host=args.host, port=args.port)


# Subcommand dispatch table. Order here only matters for docs / --help;
# ``main`` matches the first positional arg against the keys and falls
# through to ``_run_snowball`` (the default pipeline run) when nothing
# matches. Each handler parses the remaining argv tail itself.
_SUBCOMMANDS: dict[str, "Callable[[list[str]], None]"] = {
    "annotate": lambda argv: _run_annotate(argv),
    "rebuild-graph": lambda argv: _run_rebuild_graph(argv),
    "fetch-pdfs": lambda argv: _run_fetch_pdfs(argv),
    "mainpath": lambda argv: _run_mainpath(argv),
    "web": lambda argv: _run_web(argv),
}


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv[0]`` as a subcommand name and dispatch to its handler.

    When no match is found (or argv is empty), falls through to
    :func:`_run_snowball`, the default pipeline run.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _SUBCOMMANDS:
        _SUBCOMMANDS[argv[0]](argv[1:])
        return
    _run_snowball(argv)


if __name__ == "__main__":
    main()
