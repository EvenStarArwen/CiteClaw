"""Enrich Paper objects with DOIs missing from the local cache.db.

CiteClaw's ``cache.db`` contains a S2 metadata blob for every paper it
fetched, but for many open-access papers S2's ``externalIds`` field is
``None`` and the ``openAccessPdf.disclaimer`` (which we use to recover
DOIs from closed-access entries) is empty. The DOIs are still in S2
itself though — they're just not in the local cache.

This module hits S2's ``/v1/paper/batch`` endpoint with the missing
``paper_id`` list and fills in DOIs in-place. **It does not write back
to cache.db** — pdfclaw deliberately treats CiteClaw's cache as read-only
to avoid step-on-toes problems if both projects are running concurrently.

Rate limiting: S2's anonymous limit is ~1 request per second; with an
API key (``S2_API_KEY`` / ``SEMANTIC_SCHOLAR_API_KEY`` env var) it's
much higher. The batch endpoint accepts up to 500 IDs per request, so
527 papers takes 1-2 calls regardless.

This module is a hard requirement for pdfclaw to be useful on real
CiteClaw checkpoints — the cache-only DOI recovery covers only 10-15%
of accepted papers in practice.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from pdfclaw.collection import Paper

log = logging.getLogger("pdfclaw.s2_enrich")

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 500
FIELDS = "externalIds,openAccessPdf,title,year,venue"


def enrich_dois_from_s2(papers: list[Paper], *, timeout: float = 60.0) -> list[Paper]:
    """Return a new list of Papers with DOIs filled in from S2 where missing.

    Network failures are NOT fatal — papers stay as-is if the API call
    or parse fails, so the rest of the pipeline can keep going.
    """
    missing = [p for p in papers if p.doi is None]
    if not missing:
        return papers

    log.info("Enriching %d papers with missing DOIs via S2 batch API", len(missing))

    api_key = (
        os.environ.get("S2_API_KEY")
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or ""
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    enriched: dict[str, str] = {}  # paper_id -> doi

    with httpx.Client(headers=headers, timeout=timeout) as http:
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            ids = [p.paper_id for p in batch]
            params = {"fields": FIELDS}

            for attempt in range(3):
                try:
                    resp = http.post(
                        S2_BATCH_URL, params=params, json={"ids": ids},
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("S2 batch network error (attempt %d): %s", attempt + 1, exc)
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 429:
                    wait = 5 + attempt * 5
                    log.warning("S2 rate limited; sleeping %ds", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    log.warning(
                        "S2 batch returned HTTP %d: %s",
                        resp.status_code, resp.text[:200],
                    )
                    break  # don't retry on non-rate-limit failures

                try:
                    rows = resp.json()
                except Exception as exc:  # noqa: BLE001
                    log.warning("S2 batch JSON parse failed: %s", exc)
                    break

                for paper_id, row in zip(ids, rows or [], strict=False):
                    if not isinstance(row, dict):
                        continue
                    ext = row.get("externalIds") or {}
                    doi = ext.get("DOI") or ext.get("doi")
                    if doi:
                        enriched[paper_id] = doi.strip()
                break  # success

            if not api_key:
                # Anonymous rate limit safety
                time.sleep(1.1)

    if not enriched:
        log.warning("S2 enrichment found no new DOIs")
        return papers

    log.info("S2 enrichment recovered %d additional DOIs", len(enriched))

    out: list[Paper] = []
    for p in papers:
        if p.doi is None and p.paper_id in enriched:
            out.append(
                Paper(
                    paper_id=p.paper_id,
                    title=p.title,
                    year=p.year,
                    venue=p.venue,
                    doi=enriched[p.paper_id],
                    doi_source="s2_batch",
                )
            )
        else:
            out.append(p)
    return out
