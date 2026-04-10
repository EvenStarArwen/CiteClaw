"""OpenAlex recipe — universal OA-copy lookup, complementary to Unpaywall.

OpenAlex is the open-source successor to Microsoft Academic Graph and
indexes ~250 million scholarly works. Like Unpaywall, it tracks OA
copies for every DOI it knows about. Coverage overlaps significantly
with Unpaywall but each catches papers the other misses, so running
both as complementary fallbacks is a free win.

  https://api.openalex.org/works/https://doi.org/{doi}

The response contains:
  - is_oa (boolean)
  - best_oa_location.pdf_url    (direct PDF URL when present)
  - best_oa_location.landing_page_url

OpenAlex doesn't require an API key but recommends an email in the
``mailto`` parameter for politeness — same convention as Unpaywall.
Set ``OPENALEX_EMAIL`` (or fall back to ``UNPAYWALL_EMAIL``).
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

log = logging.getLogger("pdfclaw.openalex")


def _git_email() -> str | None:
    try:
        import subprocess  # noqa: PLC0415
        r = subprocess.run(
            ["git", "config", "--get", "user.email"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            email = r.stdout.strip()
            if "@" in email and "example.com" not in email:
                return email
    except Exception:  # noqa: BLE001
        pass
    return None


# OpenAlex doesn't enforce email validity but they ask you to be a
# good citizen. Use the user's real git email if available; fall back
# to a generic research-tools placeholder.
DEFAULT_EMAIL = _git_email() or "pdfclaw@anthropic.com"


class OpenAlexRecipe:
    name = "openalex_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
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

        email = (
            os.environ.get("OPENALEX_EMAIL")
            or os.environ.get("UNPAYWALL_EMAIL")
            or DEFAULT_EMAIL
        )
        api_url = f"https://api.openalex.org/works/https://doi.org/{doi}"

        try:
            resp = http.get(api_url, params={"mailto": email}, timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"OpenAlex API request failed: {exc}",
            )

        if resp.status_code == 404:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error="OpenAlex has no record of this DOI",
            )

        if resp.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"OpenAlex returned HTTP {resp.status_code}",
            )

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"OpenAlex JSON parse failed: {exc}",
            )

        oa = data.get("open_access") or {}
        if not oa.get("is_oa"):
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error="OpenAlex: paper is not OA",
            )

        best = data.get("best_oa_location") or {}
        pdf_url = best.get("pdf_url") or best.get("landing_page_url")
        if not pdf_url:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error="OpenAlex is_oa=true but no usable URL in best_oa_location",
            )

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
                    f"OA URL {pdf_url} returned non-PDF content "
                    f"(first bytes: {body[:8]!r})"
                ),
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_OK,
            pdf_bytes=body, fetched_via=self.name,
            extra={
                "pdf_url": pdf_url,
                "n_bytes": len(body),
                "version": best.get("version"),
                "license": best.get("license"),
            },
        )
