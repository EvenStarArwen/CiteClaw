"""Top-level fetch orchestrator with per-paper recipe fallback chain.

For each paper:

  1. Find ALL matching recipes (e.g. PNAS papers match both
     ``HighwireOpenAccessRecipe`` HTTP and ``PNASRecipe`` browser).
  2. Try them in registry order — HTTP first, browser second.
  3. First success wins. If they all fail, the last error is reported.

The browser is opened lazily but only ONCE for the whole run, before
the loop starts. We open it if any paper in the plan has a recipe that
might need it.

Other design notes:

  * **Idempotent / resumable** — if ``parsed_dir/<paper_id>.json``
    already exists and looks valid, the paper is skipped. Ctrl-C and
    re-run any time without re-downloading.
  * **Per-publisher AUTH suppression** — if a recipe returns
    ``STATUS_AUTH``, we record the recipe name and skip subsequent
    papers that would only match recipes from the same auth-failed
    set. This avoids hammering Nature 100 times when the user just
    needs to re-run ``pdfclaw login``.
  * **Random backoff** — sleep ``sleep_range`` between fetches so we
    look like a human reading articles.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from pdfclaw.collection import Paper
from pdfclaw.parser import parse_pdf_bytes
from pdfclaw.publishers import Recipe, build_default_registry, find_recipes
from pdfclaw.publishers.base import (
    STATUS_AUTH,
    STATUS_OK,
    FetchResult,
)

log = logging.getLogger("pdfclaw.fetcher")


@dataclass
class FetchStats:
    total: int = 0
    skipped_existing: int = 0
    skipped_no_doi: int = 0
    skipped_no_recipe: int = 0
    ok: int = 0
    auth_required: int = 0
    failed: int = 0
    by_recipe: dict[str, int] = field(default_factory=dict)


class Fetcher:
    """Orchestrate the fetch of a list of Papers."""

    def __init__(
        self,
        *,
        parsed_dir: Path,
        pdf_dir: Path,
        profile_path: Path,
        recipes: list[Recipe] | None = None,
        sleep_range: tuple[float, float] = (15.0, 45.0),
        headless: bool = False,
        http_user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        ),
    ):
        self.parsed_dir = Path(parsed_dir)
        self.pdf_dir = Path(pdf_dir)
        self.profile_path = Path(profile_path).expanduser()
        self.recipes = recipes or build_default_registry()
        self.sleep_range = sleep_range
        self.headless = headless
        self.http_user_agent = http_user_agent

        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def parsed_path(self, paper_id: str) -> Path:
        return self.parsed_dir / f"{paper_id}.json"

    def pdf_path(self, paper_id: str) -> Path:
        return self.pdf_dir / f"{paper_id}.pdf"

    def already_done(self, paper_id: str) -> bool:
        p = self.parsed_path(paper_id)
        if not p.exists() or p.stat().st_size < 32:
            return False
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return bool(obj.get("body_text"))
        except Exception:  # noqa: BLE001
            return False

    def _plan(
        self,
        papers: list[Paper],
        *,
        filter_recipe: str | None = None,
        filter_doi_prefix: str | None = None,
    ) -> tuple[list[tuple[Paper, list[Recipe]]], FetchStats]:
        stats = FetchStats(total=len(papers))
        plan: list[tuple[Paper, list[Recipe]]] = []
        for paper in papers:
            if paper.doi is None:
                stats.skipped_no_doi += 1
                continue
            if filter_doi_prefix and not paper.doi.lower().startswith(filter_doi_prefix.lower()):
                continue
            if self.already_done(paper.paper_id):
                stats.skipped_existing += 1
                continue
            chain = find_recipes(paper.doi, self.recipes)
            if filter_recipe is not None:
                chain = [r for r in chain if r.name == filter_recipe]
            if not chain:
                stats.skipped_no_recipe += 1
                continue
            plan.append((paper, chain))
        return plan, stats

    def run(
        self,
        papers: list[Paper],
        *,
        max_papers: int | None = None,
        filter_recipe: str | None = None,
        filter_doi_prefix: str | None = None,
    ) -> FetchStats:
        plan, stats = self._plan(
            papers, filter_recipe=filter_recipe, filter_doi_prefix=filter_doi_prefix,
        )

        if max_papers is not None and max_papers > 0:
            plan = plan[:max_papers]

        # Will we need a browser at all?
        needs_browser = any(
            any(r.needs_browser for r in chain)
            for _, chain in plan
        )

        log.info(
            "fetch plan: %d papers (skipped: %d existing, %d no-DOI, %d no-recipe). "
            "Browser needed: %s",
            len(plan), stats.skipped_existing,
            stats.skipped_no_doi, stats.skipped_no_recipe, needs_browser,
        )

        if not plan:
            return stats

        # Tracks publishers that have already returned AUTH so we can
        # skip them quickly without re-launching the browser flow.
        auth_failed_recipes: set[str] = set()
        # Per-recipe consecutive failure count. After N failures in a
        # row from the same recipe, we suppress it for the rest of the
        # run — this catches CF/Akamai blocks (where every fetch fails
        # the same way) without grinding through all 57 bioRxiv papers.
        consecutive_failures: dict[str, int] = {}
        FAILURE_THRESHOLD = 3

        with httpx.Client(
            headers={"User-Agent": self.http_user_agent},
            timeout=120.0,
            follow_redirects=True,
        ) as http:
            page = None
            ctx_manager = None
            try:
                if needs_browser:
                    from pdfclaw.browser import open_browser_context
                    tmp_downloads = self.pdf_dir / ".tmp_downloads"
                    ctx_manager = open_browser_context(
                        self.profile_path,
                        headless=self.headless,
                        downloads_dir=tmp_downloads,
                    )
                    _ctx, page = ctx_manager.__enter__()

                for paper, chain in plan:
                    result = self._try_chain(
                        paper, chain, http, page, auth_failed_recipes,
                    )
                    self._handle_result(paper, result, stats)
                    # Track consecutive failures per recipe; after
                    # FAILURE_THRESHOLD, treat as auth-suppressed.
                    rname = result.fetched_via
                    if rname and rname != "(none)":
                        if result.ok:
                            consecutive_failures[rname] = 0
                        else:
                            consecutive_failures[rname] = consecutive_failures.get(rname, 0) + 1
                            if (
                                consecutive_failures[rname] >= FAILURE_THRESHOLD
                                and rname not in auth_failed_recipes
                            ):
                                auth_failed_recipes.add(rname)
                                log.warning(
                                    "Recipe %s hit %d consecutive failures; suppressing for the rest of the run "
                                    "(CF/Akamai/JS-rendering hostile in headless? or selectors broken?). "
                                    "See /tmp/pdfclaw_failures/ for snapshots.",
                                    rname, consecutive_failures[rname],
                                )
                    self._sleep_a_bit()
            finally:
                if ctx_manager is not None:
                    ctx_manager.__exit__(None, None, None)

        return stats

    def _try_chain(
        self,
        paper: Paper,
        chain: list[Recipe],
        http: httpx.Client,
        browser_page,
        auth_failed_recipes: set[str],
    ) -> FetchResult:
        """Walk the recipe chain until one returns OK or all fail."""
        last_result: FetchResult | None = None
        for recipe in chain:
            if recipe.needs_browser and browser_page is None:
                continue  # browser pass not running
            if recipe.name in auth_failed_recipes:
                # We already know this publisher needs a fresh login
                continue

            try:
                result = recipe.fetch(
                    paper.paper_id, paper.doi,
                    browser_page=browser_page if recipe.needs_browser else None,
                    http=http if not recipe.needs_browser else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Recipe %s raised on %s: %s",
                    recipe.name, paper.paper_id, exc,
                )
                continue

            if result.ok:
                return result
            if result.status == STATUS_AUTH:
                auth_failed_recipes.add(recipe.name)
                log.warning(
                    "AUTH required for recipe %s; will skip subsequent papers from this publisher",
                    recipe.name,
                )
            last_result = result

        if last_result is None:
            # All recipes were skipped (e.g. all browser recipes with no browser)
            from pdfclaw.publishers.base import STATUS_ERROR
            return FetchResult(
                paper_id=paper.paper_id, doi=paper.doi or "",
                status=STATUS_ERROR, fetched_via="(none)",
                error="No matching recipes were available to try.",
            )
        return last_result

    def _handle_result(
        self, paper: Paper, result: FetchResult, stats: FetchStats,
    ) -> None:
        # Track per-recipe counts
        if result.fetched_via:
            stats.by_recipe[result.fetched_via] = stats.by_recipe.get(result.fetched_via, 0) + 1

        if result.ok:
            self._save_pdf(paper, result)
            try:
                self._save_parsed(paper, result)
                stats.ok += 1
                log.info(
                    "OK %s (%s, %d bytes)",
                    paper.paper_id, result.fetched_via, len(result.pdf_bytes or b""),
                )
            except Exception as exc:  # noqa: BLE001
                stats.failed += 1
                log.warning("PARSE-FAIL %s: %s", paper.paper_id, exc)
        elif result.status == STATUS_AUTH:
            stats.auth_required += 1
            log.warning("AUTH %s [%s]: %s", paper.paper_id, result.fetched_via, result.error)
        else:
            stats.failed += 1
            log.warning(
                "FAIL %s [%s/%s]: %s",
                paper.paper_id, result.fetched_via, result.status, result.error,
            )

    def _save_pdf(self, paper: Paper, result: FetchResult) -> None:
        if not result.pdf_bytes:
            return
        path = self.pdf_path(paper.paper_id)
        path.write_bytes(result.pdf_bytes)

    def _save_parsed(self, paper: Paper, result: FetchResult) -> None:
        if result.pdf_bytes:
            parsed = parse_pdf_bytes(result.pdf_bytes)
            pdf_size = len(result.pdf_bytes)
            pdf_path_str = str(self.pdf_path(paper.paper_id))
        else:
            # Already-extracted text (BioC PMC, Elsevier TDM XML, etc.)
            text = result.body_text or ""
            parsed = {
                "n_pages": 0,
                "n_chars": len(text),
                "body_text": text,
                "meta": {},
            }
            pdf_size = 0
            pdf_path_str = ""
        record = {
            "paper_id": paper.paper_id,
            "doi": paper.doi,
            "doi_source": paper.doi_source,
            "title_from_collection": paper.title,
            "year": paper.year,
            "venue": paper.venue,
            "fetched_via": result.fetched_via,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "pdf_path": pdf_path_str,
            "pdf_size_bytes": pdf_size,
            **parsed,
            "fetch_extra": result.extra,
        }
        path = self.parsed_path(paper.paper_id)
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _sleep_a_bit(self) -> None:
        if self.sleep_range[1] <= 0:
            return
        delay = random.uniform(*self.sleep_range)
        log.debug("sleeping %.1fs", delay)
        time.sleep(delay)


def _stats_to_dict(stats: FetchStats) -> dict:
    return asdict(stats)
