"""Elsevier TDM API recipe — official institutional Text and Data Mining endpoint.

Elsevier provides a free Text and Data Mining (TDM) API for academic
researchers at subscribing institutions. With an API key, you can
fetch full article XML (and sometimes PDF) for any Elsevier-published
article — including ScienceDirect (Cell Press, Plant Cell, etc.).

  https://api.elsevier.com/content/article/doi/{doi}?apiKey={key}
  Headers: {"Accept": "application/pdf"} or {"Accept": "application/xml"}

To get an API key:
  1. Register at https://dev.elsevier.com/
  2. Create an "Academic" or "Research" application
  3. The API key is free for non-commercial research
  4. (Optional but recommended) Get an institutional token from your
     library — the API auto-detects subscription content based on
     IP, but the token works from anywhere

Set environment variables:
  ELSEVIER_TDM_API_KEY    — required
  ELSEVIER_INST_TOKEN     — optional, but recommended for off-campus

When ``ELSEVIER_TDM_API_KEY`` isn't set, this recipe returns NOT_FOUND
immediately so the chain falls through to elsevier_browser.
"""

from __future__ import annotations

import logging
import os
import re
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

log = logging.getLogger("pdfclaw.elsevier_tdm")

# Strip XML tags for body text extraction
XML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _xml_to_text(xml: str) -> str:
    text = XML_TAG_RE.sub(" ", xml)
    text = WS_RE.sub(" ", text)
    return text.strip()


class ElsevierTDMRecipe:
    name = "elsevier_tdm_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        return doi.lower().startswith("10.1016/")

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

        api_key = os.environ.get("ELSEVIER_TDM_API_KEY")
        if not api_key:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error=(
                    "ELSEVIER_TDM_API_KEY env var not set. Get a free academic key at "
                    "https://dev.elsevier.com/"
                ),
            )

        api_url = f"https://api.elsevier.com/content/article/doi/{doi}"
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/pdf",
        }
        inst_token = os.environ.get("ELSEVIER_INST_TOKEN")
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token

        # Try PDF first
        try:
            resp = http.get(
                api_url, headers=headers, follow_redirects=True, timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"Elsevier TDM request failed: {exc}",
            )

        if resp.status_code == 200:
            body = resp.content
            if body.startswith(b"%PDF-"):
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={"n_bytes": len(body), "format": "pdf"},
                )

        # Fall back to XML
        try:
            headers_xml = {**headers, "Accept": "application/xml"}
            resp_xml = http.get(api_url, headers=headers_xml, timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"Elsevier TDM XML request failed: {exc}",
            )

        if resp_xml.status_code == 401:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error="Elsevier TDM 401 — API key invalid or institutional access required",
            )

        if resp_xml.status_code == 404:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error=f"Elsevier TDM 404 for {doi}",
            )

        if resp_xml.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"Elsevier TDM XML returned HTTP {resp_xml.status_code}",
            )

        text = _xml_to_text(resp_xml.text or "")
        if len(text) < 500:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=f"Elsevier TDM XML returned only {len(text)} chars (probably empty/error)",
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            body_text=text, fetched_via=self.name,
            extra={"n_chars": len(text), "format": "elsevier_tdm_xml"},
        )
