"""``citeclaw fetch-pdfs <data_dir>`` — bulk-download open-access PDFs.

Given a finished CiteClaw run directory (one with a
``literature_collection*.json`` and a ``cache.db``), this CLI subcommand:

  1. Loads the latest ``literature_collection*.json`` (latest ``.expN``
     iteration if continuation runs are present).
  2. Optionally refreshes each paper's ``pdf_url`` from the local
     ``cache.db`` ``paper_metadata`` table — useful when an older
     CiteClaw run wrote the JSON before openAccessPdf was populated.
  3. For every paper with an open-access URL, downloads the raw PDF to
     ``<data_dir>/PDFs/<paper_id>.pdf`` and writes the parsed body text
     alongside as ``<paper_id>.txt``.
  4. Optionally writes the parse outcome into the ``paper_full_text``
     cache table so subsequent CiteClaw runs that use the ``full_text``
     LLMFilter scope skip the redundant fetch.

The downloader is the same one ``PdfFetcher`` uses for full-text screening
(``download_pdf_bytes`` + ``parse_pdf_bytes`` from ``citeclaw.clients.pdf``)
so HTML/captcha sniffing, the 50 MB cap, the bioRxiv-friendly UA, and
the PyMuPDF-then-pypdf parser stack all come for free.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from citeclaw.cache import Cache
from citeclaw.clients.pdf import (
    DEFAULT_MAX_SIZE_MB,
    download_pdf_bytes,
    make_pdf_http_client,
    parse_pdf_bytes,
)
from citeclaw.models import PaperRecord
from citeclaw.progress import console

if TYPE_CHECKING:
    import httpx

log = logging.getLogger("citeclaw.fetch_pdfs")


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass
class _PaperFetchResult:
    """Outcome for one paper. ``status`` is the same vocabulary
    ``download_pdf_bytes`` / ``parse_pdf_bytes`` use plus two CLI-specific
    states (``downloaded`` / ``cached_disk``)."""

    paper_id: str
    title: str
    status: str
    pdf_path: Path | None = None
    text_path: Path | None = None
    text_chars: int = 0


# ---------------------------------------------------------------------------
# Loading + refresh helpers
# ---------------------------------------------------------------------------


def _find_latest_collection(data_dir: Path) -> Path:
    """Pick the highest-iteration ``literature_collection*.json`` in data_dir.

    Mirrors ``checkpoint.load_checkpoint``'s ``.expN`` selection so the
    CLI always sees the most recent continuation run.
    """
    candidates = sorted(data_dir.glob("literature_collection*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No literature_collection*.json found in {data_dir}"
        )

    def exp_num(p: Path) -> int:
        m = re.search(r"\.exp(\d+)\.", p.name)
        return int(m.group(1)) if m else 0

    return max(candidates, key=exp_num)


def _load_collection(coll_path: Path) -> list[PaperRecord]:
    with open(coll_path, encoding="utf-8") as f:
        data = json.load(f)
    papers: list[PaperRecord] = []
    for p_dict in data.get("papers", []):
        try:
            papers.append(PaperRecord(**p_dict))
        except Exception as exc:
            log.warning(
                "Skipping malformed paper in collection (%s): %s",
                p_dict.get("paper_id", "?")[:20], exc,
            )
    return papers


def _refresh_pdf_urls_from_cache(
    papers: list[PaperRecord], cache_path: Path,
) -> int:
    """For papers missing ``pdf_url``, look up the S2 cache directly.

    Mutates ``papers`` in place. Returns the number of papers that
    gained a ``pdf_url`` from this refresh. Useful when an older
    CiteClaw version wrote ``literature_collection.json`` before the
    JSON writer started persisting ``pdf_url`` (or before the S2 fields
    list included ``openAccessPdf``).

    Reads cache.db with raw sqlite — avoids dragging in the S2 client
    or a Settings object just to peek at one column.
    """
    if not cache_path.exists():
        return 0
    conn = sqlite3.connect(str(cache_path))
    gained = 0
    try:
        for p in papers:
            if p.pdf_url:
                continue
            cur = conn.execute(
                "SELECT data FROM paper_metadata WHERE paper_id = ?",
                (p.paper_id,),
            )
            row = cur.fetchone()
            if not row:
                continue
            try:
                data = json.loads(row[0])
            except Exception as exc:  # noqa: BLE001
                # Corrupted cache row — skip this paper. DEBUG-log so
                # postmortem can correlate missing PDF URLs with cache
                # entries that failed to deserialise, without spamming
                # WARNING on a per-paper loop.
                log.debug("fetch_pdfs: cache row JSON decode failed: %s", exc)
                continue
            blob = data.get("openAccessPdf") or {}
            url = blob.get("url") if isinstance(blob, dict) else None
            if url:
                p.pdf_url = url
                gained += 1
    finally:
        conn.close()
    return gained


# ---------------------------------------------------------------------------
# Per-paper fetch
# ---------------------------------------------------------------------------


def _safe_filename(paper_id: str) -> str:
    """Filesystem-safe filename derived from a paper id.

    S2 hex IDs are already safe; ``DOI:10.1/abc`` / ``ArXiv:...`` style
    IDs contain ``:`` and ``/`` which need replacing on Windows + macOS.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def _process_one(
    paper: PaperRecord,
    out_dir: Path,
    http: "httpx.Client",
    *,
    max_size_bytes: int,
    overwrite: bool,
) -> _PaperFetchResult:
    """Download + parse one paper. Returns a result struct (no exceptions)."""
    safe = _safe_filename(paper.paper_id)
    pdf_path = out_dir / f"{safe}.pdf"
    txt_path = out_dir / f"{safe}.txt"

    if not paper.pdf_url:
        return _PaperFetchResult(
            paper_id=paper.paper_id, title=paper.title, status="no_pdf",
        )

    # Both already exist — nothing to do unless --overwrite is set.
    if pdf_path.exists() and txt_path.exists() and not overwrite:
        try:
            chars = len(txt_path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            chars = 0
        return _PaperFetchResult(
            paper_id=paper.paper_id, title=paper.title, status="cached_disk",
            pdf_path=pdf_path, text_path=txt_path, text_chars=chars,
        )

    # PDF on disk but no parsed text yet — re-parse without re-downloading.
    if pdf_path.exists() and not overwrite:
        try:
            body = pdf_path.read_bytes()
        except OSError as exc:
            log.warning("Could not read existing PDF %s: %s", pdf_path, exc)
            return _PaperFetchResult(
                paper_id=paper.paper_id, title=paper.title,
                status="download_failed",
            )
    else:
        body, dl_err = download_pdf_bytes(
            http, paper.pdf_url, max_size_bytes=max_size_bytes,
        )
        if dl_err is not None:
            return _PaperFetchResult(
                paper_id=paper.paper_id, title=paper.title, status=dl_err,
            )
        try:
            pdf_path.write_bytes(body)
        except OSError as exc:
            log.warning("Could not write %s: %s", pdf_path, exc)
            return _PaperFetchResult(
                paper_id=paper.paper_id, title=paper.title,
                status="download_failed",
            )

    # Parse — no max_chars cap so the on-disk .txt holds the full body.
    text = parse_pdf_bytes(body, max_chars=None)
    if text is None:
        return _PaperFetchResult(
            paper_id=paper.paper_id, title=paper.title, status="parse_failed",
            pdf_path=pdf_path,
        )
    try:
        txt_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write %s: %s", txt_path, exc)
        return _PaperFetchResult(
            paper_id=paper.paper_id, title=paper.title, status="parse_failed",
            pdf_path=pdf_path,
        )
    return _PaperFetchResult(
        paper_id=paper.paper_id, title=paper.title, status="downloaded",
        pdf_path=pdf_path, text_path=txt_path, text_chars=len(text),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_fetch_pdfs(
    data_dir: Path,
    *,
    max_workers: int = 4,
    overwrite: bool = False,
    refresh_from_cache: bool = True,
    update_cache: bool = True,
) -> None:
    """Download + parse open-access PDFs for every accepted paper.

    Side effects:
      - Creates ``<data_dir>/PDFs/`` if missing.
      - Writes ``<data_dir>/PDFs/<paper_id>.pdf`` (raw bytes) and
        ``.txt`` (parsed body) for each successful paper.
      - If ``update_cache`` is true and ``cache.db`` exists, writes the
        parse outcome to the ``paper_full_text`` cache table so future
        full-text screening runs reuse this work.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {data_dir}")

    coll_path = _find_latest_collection(data_dir)
    console.print(f"[bold]Loading collection:[/] {coll_path.name}")
    papers = _load_collection(coll_path)
    n_total = len(papers)
    console.print(f"  {n_total} accepted papers")

    cache_path = data_dir / "cache.db"
    if refresh_from_cache:
        gained = _refresh_pdf_urls_from_cache(papers, cache_path)
        if gained:
            console.print(
                f"  [dim]refreshed pdf_url from cache for {gained} paper(s)[/]"
            )

    out_dir = data_dir / "PDFs"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_with_pdf = sum(1 for p in papers if p.pdf_url)
    n_without = n_total - n_with_pdf
    console.print(
        f"  [bright_green]●[/] {n_with_pdf} with open-access PDF   "
        f"[grey50]○[/] {n_without} without"
    )
    if n_with_pdf == 0:
        console.print("[yellow]Nothing to fetch — exiting.[/]")
        return

    # Optional cache writeback so future full-text scope runs reuse this.
    cache: Cache | None = None
    if update_cache and cache_path.exists():
        cache = Cache(cache_path)

    http = make_pdf_http_client()
    max_size_bytes = DEFAULT_MAX_SIZE_MB * 1024 * 1024
    results: list[_PaperFetchResult] = []

    targets = [p for p in papers if p.pdf_url]
    console.print(
        f"[bold]Downloading {len(targets)} PDFs → {out_dir}/[/] "
        f"[dim](workers={max_workers}, overwrite={overwrite})[/]"
    )
    console.print()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_one, p, out_dir, http,
                    max_size_bytes=max_size_bytes, overwrite=overwrite,
                ): p
                for p in targets
            }
            for fut in as_completed(futures):
                paper = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    log.warning(
                        "Unexpected error fetching %s: %s",
                        paper.paper_id, exc,
                    )
                    res = _PaperFetchResult(
                        paper_id=paper.paper_id, title=paper.title,
                        status="download_failed",
                    )
                results.append(res)
                _print_result_line(res)

                if cache is not None:
                    if res.status == "downloaded" and res.text_path is not None:
                        try:
                            cache.put_full_text(
                                res.paper_id,
                                text=res.text_path.read_text(encoding="utf-8"),
                                error=None,
                            )
                        except OSError:
                            pass
                    elif res.status == "cached_disk" and res.text_path is not None:
                        # Already on disk; only write to cache if it's empty
                        # there. Avoid clobbering successful prior parses.
                        existing = cache.get_full_text(res.paper_id)
                        if existing is None or existing.get("text") is None:
                            try:
                                cache.put_full_text(
                                    res.paper_id,
                                    text=res.text_path.read_text(encoding="utf-8"),
                                    error=None,
                                )
                            except OSError:
                                pass
                    elif res.status not in ("downloaded", "cached_disk"):
                        cache.put_full_text(
                            res.paper_id, text=None, error=res.status,
                        )
    finally:
        http.close()
        if cache is not None:
            cache.close()

    _write_failure_report(results, papers, out_dir)
    _print_summary(results, n_without)


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


_STATUS_RENDER = {
    "downloaded":      ("[bright_green]✓[/]", "downloaded"),
    "cached_disk":     ("[cyan]·[/]",         "on disk   "),
    "no_pdf":          ("[grey50]○[/]",       "no pdf    "),
    "download_failed": ("[red]✗[/]",          "dl-failed "),
    "parse_failed":    ("[yellow]?[/]",       "parse-fail"),
    "too_large":       ("[yellow]?[/]",       "too-large "),
    "not_pdf":         ("[yellow]?[/]",       "not-pdf   "),
}


def _print_result_line(res: _PaperFetchResult) -> None:
    icon, label = _STATUS_RENDER.get(res.status, ("?", res.status))
    title = (res.title or "?")[:55].ljust(55)
    chars = f"{res.text_chars:>7,} chars" if res.text_chars else " " * 13
    console.print(f"  {icon} [dim]{label}[/]  {title}  [dim]{chars}[/]")


def _write_failure_report(
    results: list[_PaperFetchResult],
    papers: list[PaperRecord],
    out_dir: Path,
) -> None:
    """Write a TSV at ``<out_dir>/_failed_downloads.tsv`` listing every
    paper we couldn't grab automatically, with title + reason + URL.

    Each row makes a manual download path explicit:
      - bioRxiv / Cloudflare-protected hosts: open in a browser
      - PMC (reCAPTCHA): open in a browser
      - Wiley / OUP paywalled: institutional access required

    The file is rewritten on every run so it always reflects the latest
    state — papers that succeed move out of the report automatically.
    """
    paper_by_id = {p.paper_id: p for p in papers}
    failures = [
        r for r in results
        if r.status not in ("downloaded", "cached_disk")
    ]
    report_path = out_dir / "_failed_downloads.tsv"
    if not failures:
        # Remove the report file when there's nothing to report — keeps the
        # PDFs/ folder tidy once a run finally bags everything.
        if report_path.exists():
            try:
                report_path.unlink()
            except OSError:
                pass
        return

    rows: list[str] = ["paper_id\tstatus\ttitle\tpdf_url"]
    for r in sorted(failures, key=lambda r: (r.status, r.title or "")):
        p = paper_by_id.get(r.paper_id)
        url = (p.pdf_url if p else "") or ""
        title = (r.title or "").replace("\t", " ").replace("\n", " ")
        rows.append(f"{r.paper_id}\t{r.status}\t{title}\t{url}")
    try:
        report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        console.print(
            f"  [dim]wrote failure report → {report_path}[/]"
        )
    except OSError as exc:
        log.warning("Could not write %s: %s", report_path, exc)


def _print_summary(
    results: list[_PaperFetchResult], n_without: int,
) -> None:
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    n_total_input = len(results) + n_without
    n_succeeded = (
        by_status.get("downloaded", 0) + by_status.get("cached_disk", 0)
    )

    console.print()
    console.print(f"[phase]{'═' * 65}[/]")
    console.print("  [bold bright_white]fetch-pdfs summary[/]")
    console.print(f"[phase]{'═' * 65}[/]")
    console.print(
        f"  total papers in collection : [bold]{n_total_input}[/]"
    )
    console.print(
        f"  with open-access pdf       : [bold]{len(results)}[/]"
    )
    console.print(
        f"  successfully on disk       : "
        f"[bold bright_green]{n_succeeded}[/]"
    )
    if by_status.get("downloaded"):
        console.print(
            f"    └ freshly downloaded     : "
            f"[bright_green]{by_status['downloaded']}[/]"
        )
    if by_status.get("cached_disk"):
        console.print(
            f"    └ already on disk        : "
            f"[cyan]{by_status['cached_disk']}[/]"
        )
    if by_status.get("download_failed"):
        console.print(
            f"  download failed            : "
            f"[red]{by_status['download_failed']}[/]"
        )
    if by_status.get("parse_failed"):
        console.print(
            f"  parse failed               : "
            f"[yellow]{by_status['parse_failed']}[/]"
        )
    if by_status.get("too_large"):
        console.print(
            f"  too large                  : "
            f"[yellow]{by_status['too_large']}[/]"
        )
    if by_status.get("not_pdf"):
        console.print(
            f"  not a pdf                  : "
            f"[yellow]{by_status['not_pdf']}[/]"
        )
    console.print(
        f"  no open-access (paywalled) : [grey50]{n_without}[/]"
    )
    console.print(f"[phase]{'═' * 65}[/]")
