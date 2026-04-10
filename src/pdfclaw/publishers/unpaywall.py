"""Unpaywall recipe — universal OA-copy lookup, matches ANY DOI.

Unpaywall is a free service that maintains a database of OA copies for
just about every DOI in CrossRef — preprints on arXiv / bioRxiv / SSRN,
deposited author manuscripts in university repositories, eLife / PLoS /
BMC fulltext, PubMed Central mirrors, and many more. Their public API
is free, no key required (just an email for politeness).

  https://api.unpaywall.org/v2/{doi}?email=YOUR_EMAIL

Returns a JSON blob with ``best_oa_location.url_for_pdf`` if any OA
copy exists. We GET that URL and verify it's a real PDF.

Coverage: ~50% of all DOIs have an OA copy in Unpaywall. Crucially,
this catches papers from publishers that DON'T expose Highwire-style
metadata (Wiley, Elsevier, ACS, RSC) when an author manuscript or
preprint copy is available — which it often is.

Email source: ``UNPAYWALL_EMAIL`` env var, falling back to a generic
research-tools placeholder. Unpaywall doesn't validate the email but
they ask you to use a real one to be a good citizen.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pdfclaw.publishers.base import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    import httpx

log = logging.getLogger("pdfclaw.unpaywall")

DEFAULT_EMAIL = "pdfclaw-research@example.com"


class UnpaywallRecipe:
    name = "unpaywall_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        # Matches every DOI — this is the catch-all OA fallback.
        return doi.lower().startswith("10.")

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page=None,  # noqa: ARG002
        http: "httpx.Client | None" = None,
    ) -> FetchResult:
        if http is None:
            raise ValueError(f"{self.name} needs an httpx.Client")

        email = os.environ.get("UNPAYWALL_EMAIL") or DEFAULT_EMAIL
        api_url = f"https://api.unpaywall.org/v2/{doi}"

        try:
            resp = http.get(api_url, params={"email": email}, timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"Unpaywall API request failed: {exc}",
            )

        if resp.status_code == 404:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error="Unpaywall has no record of this DOI",
            )

        if resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"Unpaywall API returned HTTP {resp.status_code}",
            )

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"Unpaywall JSON parse failed: {exc}",
            )

        if not data.get("is_oa"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error="Unpaywall: no OA copy known for this DOI",
            )

        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error="Unpaywall is_oa=true but no usable URL in best_oa_location",
            )

        # Fetch the PDF (or HTML landing — handle both)
        try:
            pdf_resp = http.get(pdf_url, follow_redirects=True, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"OA URL GET {pdf_url} failed: {exc}",
            )

        if pdf_resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"OA URL {pdf_url} returned HTTP {pdf_resp.status_code}",
            )

        body = pdf_resp.content
        if not body.startswith(b"%PDF-"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=(
                    f"OA URL {pdf_url} returned non-PDF content (first bytes: {body[:8]!r}). "
                    "Probably an HTML landing page that needs JS rendering."
                ),
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            pdf_bytes=body, fetched_via=self.name,
            extra={
                "pdf_url": pdf_url,
                "n_bytes": len(body),
                "host_type": best.get("host_type"),
                "evidence": best.get("evidence"),
            },
        )
