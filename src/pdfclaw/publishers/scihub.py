"""Sci-Hub recipe — opt-in last-resort fallback for paywalled papers.

Sci-Hub is a shadow library that hosts most academic papers (papers
published before ~2021; their collection has been frozen since then).
It's a legal grey area: most countries treat it as copyright
infringement, while academic users widely tolerate / use it.

This recipe is **OFF BY DEFAULT** and only enabled when the
environment variable ``PDFCLAW_ENABLE_SCIHUB`` is set to a truthy
value. If you turn it on, you take responsibility for compliance
with your local law and your institution's policies.

How it works:
  1. POST to one of the Sci-Hub mirror domains with the DOI
  2. Parse the response HTML for the iframe ``src`` (or ``embed``)
     that points at the actual PDF
  3. GET that URL via httpx
  4. Verify it's a real PDF

We try a list of mirrors in order — they go up and down constantly
and the working ones rotate over time.
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

log = logging.getLogger("pdfclaw.scihub")

# Mirror list — rotated frequently. Add/remove as the network shifts.
MIRRORS = (
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.ren",
)

# Pull the actual PDF URL out of the Sci-Hub HTML response
PDF_SRC_RE = re.compile(
    r'(?:src|onclick)\s*=\s*[\'"]([^\'"]*\.pdf[^\'"]*?)[\'"]',
    re.IGNORECASE,
)
EMBED_RE = re.compile(
    r'(?:embed|iframe)[^>]+src\s*=\s*[\'"]([^\'"]+)[\'"]',
    re.IGNORECASE,
)


def _enabled() -> bool:
    return bool(os.environ.get("PDFCLAW_ENABLE_SCIHUB"))


class SciHubRecipe:
    name = "scihub_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        # Catch-all — but only when explicitly enabled
        return doi.lower().startswith("10.")

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page=None,  # noqa: ARG002
        http: "httpx.Client | None" = None,
    ) -> FetchResult:
        if not _enabled():
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error=(
                    "Sci-Hub fallback disabled. Set PDFCLAW_ENABLE_SCIHUB=1 "
                    "to enable. Note: Sci-Hub is a copyright grey area in many "
                    "jurisdictions; check your local law and institution policy."
                ),
            )

        if http is None:
            raise ValueError(f"{self.name} needs an httpx.Client")

        last_err = None
        for mirror in MIRRORS:
            url = f"{mirror}/{doi}"
            try:
                resp = http.get(
                    url, follow_redirects=True, timeout=30.0,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{mirror}: {exc}"
                continue

            if resp.status_code != 200:
                last_err = f"{mirror}: HTTP {resp.status_code}"
                continue

            # If the mirror just gave us the PDF directly, ship it
            if resp.content.startswith(b"%PDF-"):
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=resp.content, fetched_via=self.name,
                    extra={"mirror": mirror, "n_bytes": len(resp.content)},
                )

            # Otherwise it's HTML — find the embed/iframe URL
            html = resp.text
            pdf_url = None
            m = EMBED_RE.search(html)
            if m:
                pdf_url = m.group(1)
            else:
                m = PDF_SRC_RE.search(html)
                if m:
                    pdf_url = m.group(1)

            if not pdf_url:
                last_err = f"{mirror}: no embed/iframe URL found in response HTML"
                continue

            # Normalise the URL — Sci-Hub serves both relative and absolute
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif pdf_url.startswith("/"):
                pdf_url = mirror + pdf_url
            elif not pdf_url.startswith("http"):
                pdf_url = "https://" + pdf_url.lstrip("/")
            # Strip URL fragments
            pdf_url = pdf_url.split("#")[0]

            try:
                pdf_resp = http.get(pdf_url, follow_redirects=True, timeout=120.0)
            except Exception as exc:  # noqa: BLE001
                last_err = f"{mirror}: PDF download failed: {exc}"
                continue

            if pdf_resp.status_code != 200:
                last_err = f"{mirror}: PDF GET HTTP {pdf_resp.status_code}"
                continue

            body = pdf_resp.content
            if not body.startswith(b"%PDF-"):
                last_err = f"{mirror}: PDF body isn't PDF (first bytes: {body[:8]!r})"
                continue

            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_OK,
                pdf_bytes=body, fetched_via=self.name,
                extra={
                    "mirror": mirror,
                    "pdf_url": pdf_url,
                    "n_bytes": len(body),
                },
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
            fetched_via=self.name,
            error=f"All Sci-Hub mirrors failed; last error: {last_err}",
        )
