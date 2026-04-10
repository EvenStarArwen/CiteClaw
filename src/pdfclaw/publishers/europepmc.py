"""Europe PMC fulltext recipe — universal biomedical fulltext via REST API.

Europe PMC is the European mirror of NCBI's PubMed Central. It hosts
fulltext for ~7 million biomedical / life-science papers, including
many Nature / Science / Cell papers after their 6-12 month embargo
expires. Critically, it has a clean REST API that returns JATS XML
or PDF directly — no browser, no auth, no Cloudflare.

  https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:<doi>&format=json
  https://www.ebi.ac.uk/europepmc/webservices/rest/<source>/<id>/fullTextXML
  https://europepmc.org/articles/PMC<id>?pdf=render

Coverage in this collection: every biomedical paper deposited in PMC,
which is mandatory for NIH-funded work and increasingly common for
EU-funded work. Catches a meaningful fraction of paywalled biomed
papers (Nature Methods / Cell / etc.) that we'd otherwise miss.

This recipe MATCHES ANY DOI — it's a fallback that runs after
publisher-specific recipes have had their first chance.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger("pdfclaw.europepmc")

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Strip JATS XML tags for body text extraction
JATS_TAG_RE = re.compile(r"<[^>]+>")
JATS_WS_RE = re.compile(r"\s+")


def _xml_to_text(xml: str) -> str:
    text = JATS_TAG_RE.sub(" ", xml)
    text = JATS_WS_RE.sub(" ", text)
    return text.strip()


class EuropePMCRecipe:
    name = "europepmc_http"
    needs_browser = False

    def matches(self, doi: str) -> bool:
        # Catch-all biomedical fulltext fallback
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

        # Step 1: search for the paper by DOI to get its PMC ID and source
        try:
            search = http.get(
                EPMC_SEARCH,
                params={
                    "query": f"DOI:{doi}",
                    "format": "json",
                    "resultType": "lite",
                },
                timeout=20.0,
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"EPMC search failed: {exc}",
            )

        if search.status_code != 200:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name,
                error=f"EPMC search returned HTTP {search.status_code}",
            )

        try:
            data = search.json()
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error=f"EPMC JSON parse failed: {exc}",
            )

        results = (data.get("resultList") or {}).get("result") or []
        if not results:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name, error="EPMC has no record of this DOI",
            )

        first = results[0]
        source = first.get("source")          # PMC | MED | PPR | ...
        article_id = first.get("id")
        pmcid = first.get("pmcid")
        has_full_text = first.get("hasFullTextXML") in ("Y", "y")
        in_open_access = first.get("isOpenAccess") in ("Y", "y")

        # Try direct PDF download from EPMC for OA papers with PMCID
        if pmcid and in_open_access:
            pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            try:
                pdf_resp = http.get(pdf_url, follow_redirects=True, timeout=120.0)
                if pdf_resp.status_code == 200 and pdf_resp.content.startswith(b"%PDF-"):
                    return FetchResult(
                        paper_id=paper_id, doi=doi, status=STATUS_OK,
                        pdf_bytes=pdf_resp.content, fetched_via=self.name,
                        extra={
                            "pmcid": pmcid, "source": source,
                            "n_bytes": len(pdf_resp.content),
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug("EPMC PDF render failed for %s: %s", pmcid, exc)

        # Fall back to fulltext XML
        if has_full_text and source and article_id:
            xml_url = (
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
                f"{source}/{article_id}/fullTextXML"
            )
            try:
                xml_resp = http.get(xml_url, timeout=60.0)
                if xml_resp.status_code == 200 and xml_resp.text:
                    text = _xml_to_text(xml_resp.text)
                    if len(text) > 500:  # plausibility check
                        return FetchResult(
                            paper_id=paper_id, doi=doi, status=STATUS_OK,
                            body_text=text, fetched_via=self.name,
                            extra={
                                "pmcid": pmcid or "", "source": source,
                                "format": "jats_xml",
                                "n_chars": len(text),
                            },
                        )
            except Exception as exc:  # noqa: BLE001
                log.debug("EPMC XML fetch failed: %s", exc)

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
            fetched_via=self.name,
            error=(
                f"EPMC has the paper (source={source}, pmcid={pmcid}) "
                f"but no fulltext available (hasFullTextXML={first.get('hasFullTextXML')}, "
                f"isOpenAccess={first.get('isOpenAccess')})"
            ),
        )
