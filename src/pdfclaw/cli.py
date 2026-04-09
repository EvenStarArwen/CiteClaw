"""``python -m pdfclaw`` CLI.

Subcommands:

  list           — show DOI coverage of a checkpoint dir, by publisher
  login          — open Chromium with the dedicated profile so you can
                   sign in to your institution once
  fetch          — actually download + parse PDFs into <ckpt>/parsed/

All paths default to sensible values relative to the checkpoint dir,
so the typical workflow is:

  python -m pdfclaw login                                # one time
  python -m pdfclaw list  data_bio_checkpoint            # see what's there
  python -m pdfclaw fetch data_bio_checkpoint --max 5    # smoke test
  python -m pdfclaw fetch data_bio_checkpoint            # full run
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

from pdfclaw import __version__
from pdfclaw.collection import load_papers
from pdfclaw.fetcher import Fetcher, _stats_to_dict
from pdfclaw.publishers import build_default_registry, find_recipe
from pdfclaw.s2_enrich import enrich_dois_from_s2
from pdfclaw.title_search import enrich_dois_via_arxiv_title_search

log = logging.getLogger("pdfclaw")

DEFAULT_PROFILE = Path("~/.pdfclaw-chrome-profile").expanduser()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_and_enrich(
    checkpoint: Path, *, enrich: bool, title_search: bool = False,
) -> list:
    papers = load_papers(checkpoint)
    if enrich:
        papers = enrich_dois_from_s2(papers)
    if title_search:
        papers = enrich_dois_via_arxiv_title_search(papers)
    return papers


def cmd_list(args: argparse.Namespace) -> int:
    papers = _load_and_enrich(
        args.checkpoint,
        enrich=not args.no_enrich,
        title_search=args.title_search,
    )
    total = len(papers)
    with_doi = [p for p in papers if p.doi]
    no_doi = [p for p in papers if p.doi is None]

    by_prefix: dict[str, int] = collections.Counter()
    by_recipe: dict[str, int] = collections.Counter()
    no_recipe: list = []
    registry = build_default_registry()

    for p in with_doi:
        prefix = p.doi.split("/", 1)[0]
        by_prefix[prefix] += 1
        recipe = find_recipe(p.doi, registry)
        if recipe is None:
            no_recipe.append(p)
            by_recipe["(no recipe)"] += 1
        else:
            by_recipe[recipe.name] += 1

    print(f"\n=== {args.checkpoint} ===")
    print(f"Total accepted papers: {total}")
    print(f"  with DOI:    {len(with_doi)}")
    print(f"  without DOI: {len(no_doi)}")
    print()
    print("DOI prefix breakdown:")
    for pfx, n in sorted(by_prefix.items(), key=lambda kv: -kv[1])[:20]:
        recipe = find_recipe(f"{pfx}/x", registry)
        marker = recipe.name if recipe else "(no recipe)"
        print(f"  {pfx:15s}  {n:5d}  → {marker}")
    print()
    print("Recipe coverage:")
    for name, n in sorted(by_recipe.items(), key=lambda kv: -kv[1]):
        print(f"  {name:25s}  {n:5d}")
    print()
    if no_doi:
        print(f"⚠ {len(no_doi)} papers have no recoverable DOI in cache.db.")
        print("  These will be silently skipped during fetch.")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    from pdfclaw.browser import launch_for_login
    launch_for_login(args.profile)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    papers = _load_and_enrich(
        args.checkpoint,
        enrich=not args.no_enrich,
        title_search=args.title_search,
    )
    parsed_dir = args.parsed_dir or (args.checkpoint / "parsed")
    pdf_dir = args.pdf_dir or (args.checkpoint / "pdfs")

    fetcher = Fetcher(
        parsed_dir=parsed_dir,
        pdf_dir=pdf_dir,
        profile_path=args.profile,
        sleep_range=(args.sleep_min, args.sleep_max),
        headless=args.headless,
    )

    print(
        f"[pdfclaw] checkpoint:    {args.checkpoint}\n"
        f"[pdfclaw] parsed_dir:    {parsed_dir}\n"
        f"[pdfclaw] pdf_dir:       {pdf_dir}\n"
        f"[pdfclaw] profile:       {args.profile}\n"
        f"[pdfclaw] max_papers:    {args.max if args.max else '(no limit)'}\n"
        f"[pdfclaw] filter_recipe: {args.filter_recipe or '(none)'}\n"
        f"[pdfclaw] filter_doi:    {args.filter_doi or '(none)'}\n"
        f"[pdfclaw] headless:      {args.headless}\n"
    )

    stats = fetcher.run(
        papers,
        max_papers=args.max,
        filter_recipe=args.filter_recipe,
        filter_doi_prefix=args.filter_doi,
    )

    print()
    print("=== fetch summary ===")
    for k, v in _stats_to_dict(stats).items():
        print(f"  {k:20s}  {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pdfclaw",
        description="Institutional-auth PDF fetcher for CiteClaw checkpoints",
    )
    p.add_argument("--version", action="version", version=f"pdfclaw {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Chrome user-data-dir for the persistent SSO profile (default: {DEFAULT_PROFILE})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show DOI coverage of a checkpoint")
    p_list.add_argument("checkpoint", type=Path)
    p_list.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip S2 batch enrichment of missing DOIs (faster, less coverage)",
    )
    p_list.add_argument(
        "--title-search",
        action="store_true",
        help="After S2 enrichment, fall back to arXiv title search for "
             "still-missing DOIs (rate-limited, slow — recovers ML conference papers)",
    )
    p_list.set_defaults(func=cmd_list)

    p_login = sub.add_parser(
        "login",
        help="Open Chromium with the dedicated profile so you can sign in to your institution",
    )
    p_login.set_defaults(func=cmd_login)

    p_fetch = sub.add_parser("fetch", help="Download + parse PDFs from a checkpoint")
    p_fetch.add_argument("checkpoint", type=Path)
    p_fetch.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip S2 batch enrichment of missing DOIs (faster, less coverage)",
    )
    p_fetch.add_argument(
        "--title-search",
        action="store_true",
        help="After S2 enrichment, fall back to arXiv title search for "
             "still-missing DOIs (rate-limited, slow)",
    )
    p_fetch.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Where to write parsed JSON (default: <checkpoint>/parsed/)",
    )
    p_fetch.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Where to write raw PDFs (default: <checkpoint>/pdfs/)",
    )
    p_fetch.add_argument("--max", type=int, default=None, help="Stop after N papers")
    p_fetch.add_argument("--sleep-min", type=float, default=15.0)
    p_fetch.add_argument("--sleep-max", type=float, default=45.0)
    p_fetch.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless (NOT recommended — Cloudflare/Akamai sites detect this)",
    )
    p_fetch.add_argument(
        "--filter-recipe",
        type=str,
        default=None,
        metavar="NAME",
        help="Only run papers handled by this recipe name (e.g. nature_browser)",
    )
    p_fetch.add_argument(
        "--filter-doi",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Only run papers whose DOI starts with this prefix (e.g. 10.1038/)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)
