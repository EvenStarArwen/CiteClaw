"""Abstract fallback for seed papers.

Semantic Scholar has no abstract for a sizable minority of records —
conference proceedings especially (NeurIPS / ICML / ICLR), where the S2
entry carries only a title. OpenAlex has near-complete abstract coverage
(stored as an inverted index), is free and keyless, so we use it as a
fallback: look the paper up by DOI, else by a guarded title search, and
reconstruct the abstract text.

Called lazily — only when the user opens a paper whose S2 abstract is
empty — so the seed search itself stays fast.
"""

from __future__ import annotations

import os
import re
import subprocess

import httpx

_OA_WORKS = "https://api.openalex.org/works"
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with",
         "using", "via", "by", "from", "at", "as"}


def _email() -> str:
    for var in ("OPENALEX_EMAIL", "UNPAYWALL_EMAIL"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    try:
        out = subprocess.run(["git", "config", "user.email"],
                             capture_output=True, text=True, timeout=3)
        if out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "citeclaw@example.org"


_EMAIL = _email()


def _toks(s: str) -> set[str]:
    return {w for w in _WORD.findall((s or "").lower()) if w not in _STOP and len(w) > 1}


def _title_overlap(a: str, b: str) -> float:
    """Overlap coefficient on content-word sets — tolerant of reordering and
    subtitle differences (preprint vs published), strict enough to reject a
    different paper."""
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _reconstruct(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    if not pos:
        return ""
    return " ".join(pos[i] for i in sorted(pos))


def _oa_by_doi(client: httpx.Client, doi: str) -> dict | None:
    doi = doi.strip().replace("https://doi.org/", "")
    try:
        r = client.get(f"https://api.openalex.org/works/doi:{doi}", params={"mailto": _EMAIL})
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return None


def _oa_by_title(client: httpx.Client, title: str) -> dict | None:
    try:
        r = client.get(_OA_WORKS, params={
            "search": title, "per-page": 5, "mailto": _EMAIL,
            "select": "display_name,abstract_inverted_index,publication_year,doi",
        })
        r.raise_for_status()
        for w in r.json().get("results", []):
            if not w.get("abstract_inverted_index"):
                continue
            if _title_overlap(title, w.get("display_name", "")) >= 0.75:
                return w
    except httpx.HTTPError:
        pass
    return None


def fetch_abstract(paper_id: str, external_ids: dict | None,
                   title: str, year=None) -> dict:
    """Return {"abstract": str, "source": str|None}. Empty abstract => not found."""
    external_ids = external_ids or {}
    with httpx.Client(timeout=20.0, headers={"User-Agent": f"CiteClaw ({_EMAIL})"}) as client:
        work = None
        doi = external_ids.get("DOI")
        if doi:
            work = _oa_by_doi(client, doi)
        if not (work and work.get("abstract_inverted_index")) and title:
            work = _oa_by_title(client, title)

    ab = _reconstruct((work or {}).get("abstract_inverted_index"))
    if ab:
        return {"abstract": ab.strip(), "source": "OpenAlex"}
    return {"abstract": "", "source": None}
