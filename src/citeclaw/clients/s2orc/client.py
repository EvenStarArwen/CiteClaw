"""HTTP client for the S2ORC full-text mirror (``/s2orc/v1``).

Corpusid resolution order for a CiteClaw paper (whose primary key is the
S2 ``paperId`` sha, which S2ORC has no index for):

1. ``external_ids["CorpusId"]`` — present on S2-hydrated records; direct.
2. ``external_ids["DOI"]`` / ``["ArXiv"]`` / ``["PubMed"]`` / ``["PubMedCentral"]``
   — the mirror resolves these via its idmap.

The best available id is handed to the mirror as ``CorpusId:`` / ``DOI:``
/ ``ARXIV:`` / ``PMID:`` / ``PMCID:`` and the mirror does the lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
class S2orcResult:
    text: str
    source: str                     # "s2orc" (fresh from mirror) | "cache"
    corpusid: int | None = None
    license: str | None = None
    status: str | None = None
    open_access_url: str | None = None
    annotations: dict | None = None

    @property
    def chars(self) -> int:
        return len(self.text or "")


def _normalize_base(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if u and not u.endswith("/s2orc/v1"):
        u = u + "/s2orc/v1"
    return u


# external_ids key -> mirror id prefix, in resolution-priority order.
_ID_PRIORITY = (
    ("corpusid", "CorpusId"),
    ("doi", "DOI"),
    ("arxiv", "ARXIV"),
    ("pubmed", "PMID"),
    ("pubmedcentral", "PMCID"),
)


def mirror_id_for(paper) -> str | None:
    """Best mirror lookup id for a paper, or None if it carries no usable id."""
    ext = {str(k).lower(): v for k, v in (getattr(paper, "external_ids", None) or {}).items()}
    for low_key, prefix in _ID_PRIORITY:
        val = ext.get(low_key)
        if val:
            return f"{prefix}:{str(val).strip()}"
    return None


class S2orcClient:
    def __init__(self, base_url: str, api_key: str = "", *,
                 timeout: float = 60.0, transport=None) -> None:
        self.base = _normalize_base(base_url)
        headers = {"x-api-key": api_key} if api_key else {}
        self._http = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    # ---- single paper ----------------------------------------------------

    def fetch_full_text(self, paper, *, cache=None,
                        include_annotations: bool = False) -> S2orcResult | None:
        """Full text for one accepted OA paper, or ``None`` if not in S2ORC.

        Cache-first: an existing ``paper_full_text.text`` row (from S2ORC
        *or* a prior PDF parse) is returned as ``source="cache"``. On a
        fresh S2ORC hit the body is written through to the cache. A miss
        writes nothing — leaving the PDF path free to try later.
        """
        pid = getattr(paper, "paper_id", None)
        if cache is not None and pid:
            try:
                hit = cache.get_full_text(pid)
            except Exception:
                hit = None
            if hit and hit.get("text"):
                return S2orcResult(text=hit["text"], source="cache")

        mid = mirror_id_for(paper)
        if not mid:
            return None
        rec = self._get_paper(mid, include_annotations=include_annotations)
        if rec is None:
            return None
        text = rec.get("text") or ""
        if cache is not None and pid and text:
            try:
                cache.put_full_text(pid, text=text)
            except Exception:  # cache write is best-effort
                log.debug("s2orc: cache write failed for %s", pid, exc_info=True)
        return S2orcResult(
            text=text, source="s2orc", corpusid=rec.get("corpusid"),
            license=rec.get("license"), status=rec.get("status"),
            open_access_url=rec.get("openAccessUrl"),
            annotations=rec.get("annotations"),
        )

    def _get_paper(self, mirror_id: str, *, include_annotations: bool) -> dict | None:
        include = "text,annotations" if include_annotations else "text"
        try:
            r = self._http.get(f"{self.base}/paper/{mirror_id}", params={"include": include})
        except httpx.HTTPError as exc:
            log.warning("s2orc: request failed for %s: %s", mirror_id, exc)
            return None
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            log.warning("s2orc: %s -> HTTP %s", mirror_id, r.status_code)
            return None
        try:
            return r.json()
        except Exception:
            return None

    # ---- batch membership ------------------------------------------------

    def availability(self, papers) -> dict[str, bool]:
        """``{paper_id: in_s2orc}`` for a list of papers via one batch/meta call
        per 1000 ids (cheap — the mirror skips the body blob for ``meta``)."""
        out: dict[str, bool] = {getattr(p, "paper_id", None): False for p in papers}
        pids: list[str] = []
        ids: list[str] = []
        for p in papers:
            mid = mirror_id_for(p)
            pid = getattr(p, "paper_id", None)
            if mid and pid:
                pids.append(pid)
                ids.append(mid)
        for i in range(0, len(ids), 1000):
            chunk_ids = ids[i: i + 1000]
            chunk_pids = pids[i: i + 1000]
            try:
                r = self._http.post(f"{self.base}/paper/batch",
                                    params={"include": "meta"}, json={"ids": chunk_ids})
                data = r.json() if r.status_code == 200 else []
            except (httpx.HTTPError, ValueError):
                data = []
            for pid, entry in zip(chunk_pids, data):
                out[pid] = entry is not None
        return out


def build_s2orc_client(settings) -> S2orcClient | None:
    """A client from Settings, or ``None`` when no S2ORC mirror is configured."""
    url = (getattr(settings, "s2orc_mirror_url", "") or "").strip()
    if not url:
        return None
    key = (getattr(settings, "s2orc_mirror_key", "") or "").strip()
    return S2orcClient(url, key)
