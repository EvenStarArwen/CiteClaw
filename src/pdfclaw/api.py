"""Public Python API for pdfclaw — DOI in, PDF out, no checkpoint required.

The CLI (:mod:`pdfclaw.cli`) is checkpoint-oriented (it reads
``literature_collection.json`` + ``cache.db`` from a CiteClaw run dir)
because that's the production use case.  But the underlying fetch logic
in :mod:`pdfclaw.fetcher` is fully DOI-keyed and can run for ad-hoc
DOI lists too.

This module exposes that capability as a small Python API consumed by:

  * The demo CLI subcommand ``python -m pdfclaw fetch-doi``.
  * CiteClaw pipeline steps that need a PDF for a single DOI rather
    than for an entire collection (e.g. interactive re-screening).
  * Future agentic systems that treat scientific PDFs as a primitive.

The API is two functions:

  * :func:`fetch_pdf` — single DOI → ``(Paper, FetchResult)``.
  * :func:`fetch_pdfs` — list of DOIs → list of ``(Paper, FetchResult)``,
    in input order.

Both write the same per-paper artefacts to disk that the checkpoint CLI
does (``out_dir/pdfs/<id>.pdf`` + ``out_dir/parsed/<id>.json``) so the
file shapes downstream consumers expect remain unchanged.  Nothing is
read from disk to drive the run — the DOI list comes in via arguments.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

from pdfclaw.collection import Paper
from pdfclaw.fetcher import Fetcher, FetchStats
from pdfclaw.publishers.base import FetchResult

DoiSpec = Union[str, dict]
"""Either a bare DOI string, or a dict with ``{doi, paper_id?, title?, year?, venue?, arxiv_id?, pmcid?}``."""

_DEFAULT_PROFILE = Path("~/.pdfclaw-chrome-profile").expanduser()


def _doi_to_paper_id(doi: str) -> str:
    """Stable 16-hex-char id for a DOI when the caller didn't supply one.

    Uses SHA1 of the lowercased DOI.  We only need uniqueness within
    the demo run; the resulting id is what file names on disk are keyed
    by, and it shows up in :class:`FetchResult.paper_id` so callers can
    correlate input DOIs to output files.
    """
    return hashlib.sha1(doi.strip().lower().encode("utf-8")).hexdigest()[:16]


def _coerce_paper(spec: DoiSpec) -> Paper:
    if isinstance(spec, str):
        doi = spec
        meta: dict = {}
    elif isinstance(spec, dict):
        doi = spec.get("doi") or ""
        meta = spec
    else:
        raise TypeError(
            f"DOI spec must be str or dict, got {type(spec).__name__}"
        )
    if not doi:
        raise ValueError("DOI spec missing 'doi' key (or empty string)")
    pid = meta.get("paper_id") or _doi_to_paper_id(doi)
    return Paper(
        paper_id=pid,
        title=meta.get("title", "") or "",
        year=meta.get("year"),
        venue=meta.get("venue"),
        doi=doi,
        doi_source=meta.get("doi_source", "user_supplied"),
        arxiv_id=meta.get("arxiv_id"),
        pmcid=meta.get("pmcid"),
    )


def fetch_pdfs(
    dois: list[DoiSpec],
    *,
    out_dir: str | Path,
    profile_path: str | Path | None = None,
    headless: bool = False,
    sleep_range: tuple[float, float] = (0.0, 0.0),
    parser: str = "pymupdf",
    parser_kwargs: dict | None = None,
) -> tuple[list[tuple[Paper, FetchResult]], FetchStats]:
    """Fetch PDFs for an arbitrary DOI list.

    Parameters
    ----------
    dois:
        Sequence of DOI strings or dicts.  When a dict is given,
        ``title`` / ``year`` / ``venue`` are passed through to recipes
        that consult them (most don't, but a few use ``title`` for
        cross-validation against the publisher page).
    out_dir:
        Directory to write outputs into.  Created on demand.  Two
        subdirs are populated: ``pdfs/`` (raw bytes) and ``parsed/``
        (JSON with ``body_text`` + metadata, mirroring what the
        checkpoint CLI emits).
    profile_path:
        Chrome user-data-dir for the persistent SSO profile.  Defaults
        to ``~/.pdfclaw-chrome-profile``.  Run ``python -m pdfclaw login``
        once to populate it.
    headless:
        Whether the Playwright browser should run headless.  Default
        ``False`` because Cloudflare / Akamai sites detect headless mode
        and block it; only set ``True`` for fully open-access publishers.
    sleep_range:
        ``(min, max)`` seconds to sleep between papers.  Defaults to
        ``(0, 0)`` for demo / interactive use; production runs should
        use ``(15, 45)`` to avoid rate-limit bans.

    Returns
    -------
    ``(results, stats)``:
        * ``results`` — list of ``(Paper, FetchResult)`` pairs in input
          order.  ``FetchResult.ok`` is ``True`` on success;
          ``FetchResult.pdf_bytes`` / ``body_text`` carry the payload
          (see :class:`pdfclaw.publishers.base.FetchResult`).
        * ``stats`` — aggregate :class:`FetchStats` (skipped, ok,
          failed, by_recipe).
    """
    out_dir = Path(out_dir).expanduser()
    parsed_dir = out_dir / "parsed"
    pdf_dir = out_dir / "pdfs"
    profile = Path(profile_path).expanduser() if profile_path else _DEFAULT_PROFILE

    papers = [_coerce_paper(d) for d in dois]

    collected: dict[str, FetchResult] = {}

    def _on_done(paper: Paper, result: FetchResult) -> None:
        collected[paper.paper_id] = result

    fetcher = Fetcher(
        parsed_dir=parsed_dir,
        pdf_dir=pdf_dir,
        profile_path=profile,
        sleep_range=sleep_range,
        headless=headless,
        parser=parser,
        parser_kwargs=parser_kwargs,
    )
    stats = fetcher.run(papers, on_paper_done=_on_done)

    # Build results in input order; papers that the planner skipped
    # (no DOI, already-cached, no recipe) won't have entries — emit a
    # placeholder FetchResult so the caller can still see one row per
    # input DOI.
    results: list[tuple[Paper, FetchResult]] = []
    for paper in papers:
        result = collected.get(paper.paper_id)
        if result is None:
            from pdfclaw.publishers.base import STATUS_NOT_FOUND
            result = FetchResult(
                paper_id=paper.paper_id,
                doi=paper.doi or "",
                status=STATUS_NOT_FOUND,
                fetched_via="(skipped by planner — already cached or no recipe)",
            )
        results.append((paper, result))
    return results, stats


def fetch_pdf(
    doi: str,
    *,
    out_dir: str | Path,
    paper_id: str | None = None,
    title: str = "",
    year: int | None = None,
    venue: str | None = None,
    arxiv_id: str | None = None,
    pmcid: str | None = None,
    profile_path: str | Path | None = None,
    headless: bool = False,
    parser: str = "pymupdf",
    parser_kwargs: dict | None = None,
) -> tuple[Paper, FetchResult]:
    """One-shot DOI → ``(Paper, FetchResult)``.

    Thin wrapper over :func:`fetch_pdfs` for the common case.  The
    optional metadata fields seed the :class:`Paper` so recipes that
    consult them (e.g. for HTML-page title sanity checks) can.
    """
    spec: dict = {"doi": doi}
    if paper_id:
        spec["paper_id"] = paper_id
    if title:
        spec["title"] = title
    if year is not None:
        spec["year"] = year
    if venue:
        spec["venue"] = venue
    if arxiv_id:
        spec["arxiv_id"] = arxiv_id
    if pmcid:
        spec["pmcid"] = pmcid
    results, _stats = fetch_pdfs(
        [spec],
        out_dir=out_dir,
        profile_path=profile_path,
        headless=headless,
        parser=parser,
        parser_kwargs=parser_kwargs,
    )
    return results[0]
