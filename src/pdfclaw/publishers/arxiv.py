"""arXiv recipe — pure HTTP, no browser, no auth.

arXiv DOIs look like ``10.48550/arXiv.2306.15794`` (the canonical S2 form
since the migration to DOIs). The arXiv ID is the substring after
``arXiv.``. The corresponding PDF URL is the simplest URL pattern in
academic publishing: ``https://arxiv.org/pdf/<id>``. arXiv accepts
versionless requests (returning the latest version) and suffixed
``.pdf`` requests interchangeably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pdfclaw.publishers.base import (
    STATUS_ERROR,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    import httpx


class ArxivRecipe:
    name = "arxiv_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        d = doi.lower()
        # Canonical S2 form, plus a few legacy variants observed in older
        # caches that pre-date the arXiv → DOI migration.
        return d.startswith("10.48550/arxiv.") or d.startswith("10.48550/")

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page=None,  # noqa: ARG002
        http: "httpx.Client | None" = None,
    ) -> FetchResult:
        if http is None:
            raise ValueError("ArxivRecipe needs an httpx.Client")

        # Strip the prefix to recover the bare arXiv ID.
        # 10.48550/arXiv.2306.15794 -> 2306.15794
        # 10.48550/2306.15794 (rare) -> 2306.15794
        suffix = doi.split("/", 1)[1]
        if suffix.lower().startswith("arxiv."):
            suffix = suffix[len("arxiv."):]
        arxiv_id = suffix

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            resp = http.get(url, follow_redirects=True, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"GET {url} failed: {exc}",
            )

        if resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"GET {url} status {resp.status_code}",
            )

        body = resp.content
        if not body.startswith(b"%PDF-"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=f"Body doesn't look like PDF (first bytes: {body[:8]!r})",
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            pdf_bytes=body, fetched_via=self.name,
            extra={"pdf_url": url, "arxiv_id": arxiv_id, "n_bytes": len(body)},
        )
