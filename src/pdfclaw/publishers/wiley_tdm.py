"""Wiley TDM API recipe — official institutional Text and Data Mining endpoint.

Wiley provides a free Text and Data Mining (TDM) API for academic
researchers. With a TDM token, you can fetch the full PDF for any
Wiley-published article without going through the web frontend or
needing institutional auth at the URL level.

  https://api.wiley.com/onlinelibrary/tdm/v1/articles/<url-encoded-doi>
  Headers: {"Wiley-TDM-Client-Token": "<token>"}

To get a token: register at https://onlinelibrary.wiley.com/library-info/resources/text-and-data-mining
The token is granted for non-commercial academic research and is
free. Set ``WILEY_TDM_API_TOKEN`` env var.

When the env var isn't set, this recipe is a no-op (returns
NOT_FOUND immediately so the chain falls through to wiley_browser).

Covers all 10.1002/ DOIs that Wiley publishes — including many
journals you might not associate with Wiley by name (Plant Biotech
Journal, etc.).

This recipe is also registered for ``10.15252/`` (EMBO Press) since
EMBO journals migrated to Wiley's hosting platform in 2023, so the
same TDM token works.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import quote

from pdfclaw.publishers.base import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    import httpx

log = logging.getLogger("pdfclaw.wiley_tdm")


class WileyTDMRecipe:
    name = "wiley_tdm_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        d = doi.lower()
        # Wiley plus EMBO (now hosted on Wiley)
        return d.startswith("10.1002/") or d.startswith("10.15252/")

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

        token = os.environ.get("WILEY_TDM_API_TOKEN")
        if not token:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error=(
                    "WILEY_TDM_API_TOKEN env var not set. Get a free TDM token at "
                    "https://onlinelibrary.wiley.com/library-info/resources/text-and-data-mining"
                ),
            )

        encoded_doi = quote(doi, safe="")
        api_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{encoded_doi}"

        try:
            resp = http.get(
                api_url,
                headers={"Wiley-TDM-Client-Token": token},
                follow_redirects=True,
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"Wiley TDM request failed: {exc}",
            )

        if resp.status_code == 404:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error=f"Wiley TDM 404 for {doi}",
            )

        if resp.status_code == 401:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error="Wiley TDM 401 — token invalid or expired",
            )

        if resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"Wiley TDM returned HTTP {resp.status_code}",
            )

        body = resp.content
        if not body.startswith(b"%PDF-"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=f"Wiley TDM returned non-PDF (first bytes: {body[:8]!r})",
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            pdf_bytes=body, fetched_via=self.name,
            extra={"n_bytes": len(body)},
        )
