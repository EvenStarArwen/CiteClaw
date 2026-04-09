"""Generic open-access HTTP recipe — Highwire ``citation_pdf_url`` meta tag.

The Highwire Press metadata standard mandates a ``<meta name="citation_pdf_url">``
tag on every article landing page. Most academic publishers comply,
which means we can fetch their PDFs with three HTTP calls and zero
publisher-specific logic:

  1. GET https://doi.org/<doi> (follow redirects)
  2. Regex out the citation_pdf_url meta tag from the response HTML
  3. GET that URL

This recipe applies to a fixed allow-list of fully-open-access
publishers that:

  * Comply with the Highwire metadata standard (or close enough)
  * Are NOT behind Cloudflare bot protection
  * Don't gate the PDF download behind JavaScript

bioRxiv used to be in this list but Cloudflare ate it. PNAS is in the
list but only the OA-window papers will succeed; recent paywalled ones
will fail with a non-PDF response (the orchestrator can then route to
the paywalled-browser fallback if you wire that up).

Adding a new OA publisher: append its DOI prefix to ``OA_PREFIXES`` and
verify with ``python -m pdfclaw fetch <ckpt> --max 1`` filtered to that
prefix. If it works, you're done.
"""

from __future__ import annotations

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

META_PDF_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Some publishers swap the attr order
META_PDF_RE_REV = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    re.IGNORECASE,
)


class HighwireOpenAccessRecipe:
    name = "highwire_oa_http"
    needs_browser = False

    # Confirmed working via direct httpx + Highwire meta tag.
    # MDPI (10.3390) and eLife (10.7554) used to be here but they don't
    # expose citation_pdf_url in a fetchable way and have CDN gating —
    # they have dedicated browser recipes (MDPIRecipe, ELifeRecipe).
    OA_PREFIXES = (
        "10.1186/",  # BMC / Springer Open
        "10.3389/",  # Frontiers
        "10.1371/",  # PLoS
        "10.1073/",  # PNAS (OA window only — paywalled ones fall to PNASRecipe)
    )

    def matches(self, doi: str) -> bool:
        d = doi.lower()
        return any(d.startswith(p) for p in self.OA_PREFIXES)

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

        # Step 1: follow DOI redirect to article landing page
        try:
            landing = http.get(
                f"https://doi.org/{doi}",
                follow_redirects=True,
                timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"DOI redirect failed: {exc}",
            )

        if landing.status_code == 404:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error=f"404 for {doi}",
            )

        if landing.status_code >= 400:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"Landing page status {landing.status_code}",
            )

        ctype = landing.headers.get("content-type", "").lower()
        # Some publishers serve XML/JSON for the landing page (rare).
        if "html" not in ctype and "xml" not in ctype:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"Expected HTML landing, got content-type={ctype}",
            )

        html = landing.text
        m = META_PDF_RE.search(html) or META_PDF_RE_REV.search(html)
        if not m:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=(
                    "No citation_pdf_url meta tag on landing page. "
                    "Either the publisher doesn't use Highwire metadata, "
                    "or the page requires JavaScript rendering."
                ),
            )

        pdf_url = m.group(1)
        # Make relative URLs absolute against the landing URL
        if pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url
        elif pdf_url.startswith("/"):
            from urllib.parse import urlparse
            parts = urlparse(str(landing.url))
            pdf_url = f"{parts.scheme}://{parts.netloc}{pdf_url}"

        # Step 2: fetch the PDF
        try:
            pdf_resp = http.get(pdf_url, follow_redirects=True, timeout=180.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"PDF GET {pdf_url} failed: {exc}",
            )

        if pdf_resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"PDF GET {pdf_url} status {pdf_resp.status_code}",
            )

        body = pdf_resp.content
        if not body.startswith(b"%PDF-"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=(
                    f"Body doesn't look like PDF (first bytes: {body[:8]!r}). "
                    "Probably means this DOI is paywalled and the meta tag "
                    "pointed at a login wall."
                ),
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            pdf_bytes=body, fetched_via=self.name,
            extra={
                "pdf_url": pdf_url,
                "n_bytes": len(body),
                "landing_url": str(landing.url),
            },
        )
