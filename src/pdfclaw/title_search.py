"""Last-resort enrichment: search arXiv by paper title.

For papers with neither a DOI nor an ArXiv ID in the S2 cache (these
are mostly ML conference papers — ICLR, NeurIPS, ICML — that have an
arXiv preprint but no formal DOI), we can fall back to arXiv's
public Atom search API and find the matching preprint by title.

The arXiv API is:

  http://export.arxiv.org/api/query?search_query=ti:%22<title>%22&max_results=3

It returns Atom XML; each ``<entry>`` has an ``<id>`` of the form
``http://arxiv.org/abs/<id>v<n>`` from which we extract the bare
arXiv ID. We accept the match if the candidate title matches the
target title via simple normalisation (case + whitespace + trimming).

The arXiv API is rate-limited to 1 request every ~3 seconds. We
sleep accordingly between calls.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from pdfclaw.collection import Paper

log = logging.getLogger("pdfclaw.title_search")

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^v\s]+)(?:v\d+)?", re.IGNORECASE)


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip(" .,;:!?")


def _query_arxiv_by_title(http: httpx.Client, title: str) -> str | None:
    """Return canonical arXiv DOI form (10.48550/arXiv.<id>) or None."""
    if not title or len(title) < 10:
        return None

    # Strip aggressive — arxiv search treats most punctuation as noise
    cleaned = re.sub(r"[^\w\s\-]", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    params = {
        "search_query": f'ti:"{cleaned}"',
        "max_results": "3",
        "sortBy": "relevance",
    }
    try:
        resp = http.get(ARXIV_API, params=params, timeout=20.0)
    except Exception as exc:  # noqa: BLE001
        log.debug("arXiv title search failed for %r: %s", title[:50], exc)
        return None

    if resp.status_code != 200:
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    target_norm = _norm_title(title)
    for entry in root.findall("a:entry", ATOM_NS):
        candidate_title_el = entry.find("a:title", ATOM_NS)
        if candidate_title_el is None or not candidate_title_el.text:
            continue
        cand_norm = _norm_title(candidate_title_el.text)
        # Accept exact match or one being a prefix of the other
        # (handles minor punctuation differences)
        if (
            cand_norm == target_norm
            or cand_norm.startswith(target_norm)
            or target_norm.startswith(cand_norm)
        ):
            id_el = entry.find("a:id", ATOM_NS)
            if id_el is not None and id_el.text:
                m = ARXIV_ID_RE.search(id_el.text)
                if m:
                    return f"10.48550/arXiv.{m.group(1)}"
    return None


def enrich_dois_via_arxiv_title_search(
    papers: list[Paper], *, timeout: float = 30.0,
) -> list[Paper]:
    """Mutate papers list: fill in DOIs for any paper still missing one
    by searching arXiv for the paper's title.

    Idempotent and rate-limited (one request per ~3.5 seconds, per
    arXiv's published API limits).
    """
    missing = [p for p in papers if p.doi is None and p.title]
    if not missing:
        return papers

    log.info(
        "Searching arXiv by title for %d papers (rate-limited, ~3s per request)",
        len(missing),
    )

    rescued: dict[str, str] = {}
    with httpx.Client(timeout=timeout) as http:
        for p in missing:
            doi = _query_arxiv_by_title(http, p.title)
            if doi:
                rescued[p.paper_id] = doi
                log.info("  rescued %s via arxiv search", p.paper_id)
            time.sleep(3.5)  # arXiv API rate limit

    if not rescued:
        log.info("Title search rescued no additional DOIs")
        return papers

    log.info("Title search rescued %d papers", len(rescued))
    out: list[Paper] = []
    for p in papers:
        if p.paper_id in rescued:
            out.append(
                Paper(
                    paper_id=p.paper_id, title=p.title, year=p.year, venue=p.venue,
                    doi=rescued[p.paper_id], doi_source="arxiv_title_search",
                )
            )
        else:
            out.append(p)
    return out
