"""eLife XML recipe — fetch JATS XML directly from eLife's GitHub mirror.

eLife (10.7554/...) is fully open access AND publishes its entire
corpus as JATS XML on GitHub at:

  https://github.com/elifesciences/elife-article-xml/tree/master/articles

The naming convention is ``elife-<id>-v<version>.xml``. We:
  1. Extract the eLife ID from the DOI (10.7554/eLife.<id>)
  2. List the articles directory once and find the matching file (cached)
  3. Fetch the raw XML and extract body text

No auth, no API key, no Cloudflare. Always works for eLife papers
that have been published (which is all of them post-2018).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pdfclaw.publishers.base import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    import httpx

log = logging.getLogger("pdfclaw.elife_xml")

ELIFE_DOI_RE = re.compile(r"10\.7554/[Ee][Ll]ife\.(\d+)")
XML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _xml_to_text(xml: str) -> str:
    text = XML_TAG_RE.sub(" ", xml)
    text = WS_RE.sub(" ", text)
    return text.strip()


class ELifeXMLRecipe:
    name = "elife_xml_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        return doi.lower().startswith("10.7554/")

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

        m = ELIFE_DOI_RE.search(doi)
        if not m:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error=f"DOI {doi} doesn't match eLife format",
            )
        elife_id = m.group(1).zfill(5)

        # Try v1, v2, v3 in order until we find one
        for version in ("v3", "v2", "v1"):
            xml_url = (
                f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/"
                f"master/articles/elife-{elife_id}-{version}.xml"
            )
            try:
                resp = http.get(xml_url, follow_redirects=True, timeout=60.0)
            except Exception:  # noqa: BLE001
                continue
            if resp.status_code != 200:
                continue
            text = _xml_to_text(resp.text or "")
            if len(text) > 500:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    body_text=text, fetched_via=self.name,
                    extra={
                        "elife_id": elife_id, "version": version,
                        "format": "jats_xml", "n_chars": len(text),
                    },
                )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
            fetched_via=self.name,
            error=f"eLife XML not found for {doi} (tried v1/v2/v3)",
        )
