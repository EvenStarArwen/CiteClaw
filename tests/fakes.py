"""In-memory fake of :class:`citeclaw.clients.s2.SemanticScholarClient`.

The real S2 client talks to ``api.semanticscholar.org`` (rate-limited at
roughly 1 req/s on the public tier), so driving the full test suite through
it would be prohibitively slow and flaky. ``FakeS2Client`` mimics the same
public surface as ``SemanticScholarClient`` and serves everything from a
deterministic, hand-built corpus — so tests can exercise the pipeline at
realistic scale without touching the network or burning LLM tokens.

The fake is intentionally *not* a subclass of the real client: a duck-typed
stand-in is the simplest way to keep type surface minimal while still
satisfying every call the pipeline makes at runtime.
"""

from __future__ import annotations

import copy
from typing import Any

from citeclaw.clients.s2.converters import paper_to_record
from citeclaw.models import PaperRecord


def make_paper(
    paper_id: str,
    *,
    title: str = "",
    abstract: str | None = None,
    year: int | None = 2022,
    venue: str | None = "Nature",
    citation_count: int | None = 10,
    influential_citation_count: int | None = 1,
    references: list[str] | None = None,
    authors: list[dict] | None = None,
    embedding: list[float] | None = None,
    pdf_url: str | None = None,
    external_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one fake paper record in the S2 JSON shape the client expects.

    Note the ``_ref_ids`` private key: this is the denormalized list of
    reference paper ids that ``FakeS2Client`` indexes for forward/backward
    lookups. The public ``references`` field uses S2's nested
    ``[{"citedPaper": {"paperId": ...}}, ...]`` shape so that
    :func:`citeclaw.clients.s2.converters.paper_to_record` can parse it
    without patching.
    """
    ref_ids = references or []
    return {
        "paperId": paper_id,
        "title": title or f"Title of {paper_id}",
        "abstract": abstract if abstract is not None else f"Abstract of {paper_id}.",
        "year": year,
        "venue": venue,
        "citationCount": citation_count,
        "influentialCitationCount": influential_citation_count,
        "references": [{"citedPaper": {"paperId": r}} for r in ref_ids],
        "_ref_ids": ref_ids,
        "authors": authors
        or [
            {"authorId": f"{paper_id}_au1", "name": f"{paper_id} Author 1"},
            {"authorId": f"{paper_id}_au2", "name": f"{paper_id} Author 2"},
        ],
        "embedding": embedding,
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
        "externalIds": external_ids or {},
    }


class FakeS2Client:
    """Offline stand-in for :class:`SemanticScholarClient`.

    Holds a dict of fake papers. Each fake paper may declare ``references``
    (outbound edges) — forward citations (incoming) are derived on the fly.
    All methods match the subset of the real client's surface that the
    pipeline uses. No method ever touches the network.
    """

    def __init__(self, corpus: dict[str, dict[str, Any]] | None = None) -> None:
        self._papers: dict[str, dict[str, Any]] = dict(corpus or {})
        self._embeddings: dict[str, list[float] | None] = {}
        self._authors: dict[str, dict[str, Any]] = {}
        # Phase A search surface: query-keyed canned responses, each
        # registered explicitly via the ``register_*`` helpers below.
        # Unregistered keys fall through to empty results so tests that
        # don't care about a particular surface aren't forced to set it up.
        self._search_bulk_results: dict[str, list[dict[str, Any]]] = {}
        self._search_match_results: dict[str, dict[str, Any]] = {}
        self._recommendation_results: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        self._author_papers_results: dict[str, list[dict[str, Any]]] = {}
        # Bookkeeping so tests can assert call counts
        self.calls: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def add(self, paper: dict[str, Any]) -> None:
        self._papers[paper["paperId"]] = paper
        if paper.get("embedding") is not None:
            self._embeddings[paper["paperId"]] = list(paper["embedding"])

    def add_author(self, author_id: str, data: dict[str, Any]) -> None:
        self._authors[author_id] = data

    def _record_call(self, key: str) -> None:
        self.calls[key] = self.calls.get(key, 0) + 1

    def _citers_of(self, paper_id: str) -> list[str]:
        """Return ids of papers whose ``_ref_ids`` include ``paper_id``."""
        return [pid for pid, d in self._papers.items() if paper_id in (d.get("_ref_ids") or [])]

    def _get_or_stub(self, paper_id: str) -> dict[str, Any]:
        """Return the paper dict, creating a minimal stub if unknown."""
        if paper_id not in self._papers:
            self._papers[paper_id] = {
                "paperId": paper_id,
                "title": "",
                "abstract": None,
                "year": None,
                "venue": None,
                "citationCount": None,
                "influentialCitationCount": None,
                "references": [],
                "_ref_ids": [],
                "authors": [],
                "openAccessPdf": None,
            }
        return self._papers[paper_id]

    # ------------------------------------------------------------------
    # Single-paper metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperRecord:
        self._record_call("fetch_metadata")
        if paper_id not in self._papers:
            raise KeyError(f"FakeS2Client: unknown paper {paper_id}")
        rec = paper_to_record(self._papers[paper_id])
        if rec is None:
            raise ValueError(f"FakeS2Client: could not convert {paper_id}")
        return rec

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def fetch_references(
        self, paper_id: str, *, progress_cb: Any = None,
    ) -> list[PaperRecord]:
        self._record_call("fetch_references")
        data = self._get_or_stub(paper_id)
        refs = data.get("_ref_ids") or []
        out: list[PaperRecord] = []
        for rid in refs:
            r = self._get_or_stub(rid)
            rec = paper_to_record(r)
            if rec is not None:
                out.append(rec)
        return out

    def fetch_reference_ids(
        self, paper_id: str, *, progress_cb: Any = None,
    ) -> list[str]:
        self._record_call("fetch_reference_ids")
        return list(self._get_or_stub(paper_id).get("_ref_ids") or [])

    def fetch_reference_edges(self, paper_id: str) -> list[dict[str, Any]]:
        self._record_call("fetch_reference_edges")
        refs = self._get_or_stub(paper_id).get("_ref_ids") or []
        return [
            {
                "target_id": rid,
                "contexts": [f"ctx-for-{rid}"],
                "intents": ["background"],
                "is_influential": False,
            }
            for rid in refs
        ]

    def has_cached_references(self, paper_id: str) -> bool:
        return paper_id in self._papers

    # ------------------------------------------------------------------
    # Citations (forward edges)
    # ------------------------------------------------------------------

    def fetch_citation_ids_and_counts(
        self, paper_id: str, *, max_items: int | None = None,
        progress_cb: Any = None,
    ) -> list[dict[str, Any]]:
        self._record_call("fetch_citation_ids_and_counts")
        citers = self._citers_of(paper_id)
        rows: list[dict[str, Any]] = []
        for cid in citers:
            d = self._papers[cid]
            rows.append(
                {
                    "paper_id": cid,
                    "citation_count": d.get("citationCount"),
                    "year": d.get("year"),
                }
            )
        if max_items is not None:
            rows = rows[:max_items]
        return rows

    def fetch_citation_edges(self, paper_id: str) -> list[dict[str, Any]]:
        self._record_call("fetch_citation_edges")
        return [
            {"source_id": cid, "contexts": [], "intents": [], "is_influential": False}
            for cid in self._citers_of(paper_id)
        ]

    def has_cached_citations(self, paper_id: str) -> bool:
        return paper_id in self._papers

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich_with_abstracts(self, records: list[PaperRecord]) -> list[PaperRecord]:
        self._record_call("enrich_with_abstracts")
        for rec in records:
            data = self._papers.get(rec.paper_id)
            if not data:
                continue
            if not rec.abstract and data.get("abstract"):
                rec.abstract = data["abstract"]
            if not rec.venue and data.get("venue"):
                rec.venue = data["venue"]
            if rec.citation_count is None and data.get("citationCount") is not None:
                rec.citation_count = data["citationCount"]
            if rec.year is None and data.get("year") is not None:
                rec.year = data["year"]
        return records

    def enrich_batch(self, candidates: list[dict[str, Any]]) -> list[PaperRecord]:
        self._record_call("enrich_batch")
        out: list[PaperRecord] = []
        for c in candidates:
            pid = c.get("paper_id")
            if not pid:
                continue
            data = self._papers.get(pid)
            if not data:
                continue
            rec = paper_to_record(data)
            if rec is not None:
                out.append(rec)
        return out

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def fetch_embedding(self, paper_id: str) -> list[float] | None:
        self._record_call("fetch_embedding")
        return copy.copy(self._embeddings.get(paper_id))

    def fetch_embeddings_batch(
        self, paper_ids: list[str],
    ) -> dict[str, list[float] | None]:
        self._record_call("fetch_embeddings_batch")
        return {pid: copy.copy(self._embeddings.get(pid)) for pid in paper_ids if pid}

    def has_cached_embedding(self, paper_id: str) -> bool:
        return paper_id in self._embeddings

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    def fetch_authors_batch(
        self, author_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        self._record_call("fetch_authors_batch")
        out: dict[str, dict[str, Any]] = {}
        for aid in author_ids:
            if aid in self._authors:
                out[aid] = self._authors[aid]
            else:
                out[aid] = {}
        return out

    # ------------------------------------------------------------------
    # Search surface (Phase A: PA-01 / PA-02 / PA-03)
    # ------------------------------------------------------------------
    #
    # The real ``SemanticScholarClient`` exposes four new search-flavored
    # methods after PA-01..PA-05. Phase B's meta-LLM agent and Phase C's
    # ``ExpandBy*`` step family will exercise them through this fake. To
    # keep tests deterministic, callers register canned responses keyed
    # by the relevant primary input (query text, anchor ids, author id);
    # unregistered keys return empty payloads in the same shape the real
    # client uses for empty results, so dependent code is exercised on
    # both populated and empty paths without surprising None checks.

    def register_search_bulk(
        self, query: str, papers: list[dict[str, Any]],
    ) -> None:
        """Register a canned ``search_bulk`` response for ``query``."""
        self._search_bulk_results[query] = list(papers)

    def search_bulk(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        sort: str | None = None,
        token: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Canned bulk search response keyed by ``query`` text.

        Mirrors the real client's payload shape (``data`` / ``total`` /
        ``token``). ``filters`` / ``sort`` / ``token`` are accepted for
        signature compatibility but ignored — Phase A tests don't need
        the filter dimension and we'd rather have one canned answer per
        query than a combinatorial registration table.
        """
        self._record_call("search_bulk")
        papers = self._search_bulk_results.get(query, [])
        sliced = [copy.deepcopy(p) for p in papers[:limit]]
        return {"data": sliced, "total": len(papers), "token": None}

    def register_search_match(
        self, title: str, paper: dict[str, Any] | None,
    ) -> None:
        """Register a canned ``search_match`` response for ``title``.

        Pass ``paper=None`` to clear an earlier registration.
        """
        if paper is None:
            self._search_match_results.pop(title, None)
        else:
            self._search_match_results[title] = dict(paper)

    def search_match(self, title: str) -> dict[str, Any] | None:
        """Canned best-match title lookup. Returns ``None`` when the
        title hasn't been registered, mirroring the real client's
        behavior on a 404 / empty data array."""
        self._record_call("search_match")
        hit = self._search_match_results.get(title)
        return copy.deepcopy(hit) if hit is not None else None

    def register_recommendations(
        self, positive_ids: list[str], papers: list[dict[str, Any]],
    ) -> None:
        """Register canned recommendations for the given anchor set.

        The key is order-independent: ``["a", "b"]`` and ``["b", "a"]``
        share one canned entry. Pass exactly the anchor ids the test
        will later send through ``fetch_recommendations``.
        """
        key = tuple(sorted(positive_ids))
        self._recommendation_results[key] = list(papers)

    def fetch_recommendations(
        self,
        positive_ids: list[str],
        *,
        negative_ids: list[str] | None = None,
        limit: int = 100,
        fields: str = "paperId,title",
    ) -> list[dict[str, Any]]:
        """Canned multi-anchor SPECTER2 kNN response. Looks up canned
        results by the sorted tuple of ``positive_ids``; returns ``[]``
        when no entry is registered. ``negative_ids`` / ``fields`` are
        accepted for signature compatibility but ignored.
        """
        self._record_call("fetch_recommendations")
        key = tuple(sorted(positive_ids))
        papers = self._recommendation_results.get(key, [])
        return [copy.deepcopy(p) for p in papers[:limit]]

    def register_author_papers(
        self, author_id: str, papers: list[dict[str, Any]],
    ) -> None:
        """Register a canned paper list for ``author_id``."""
        self._author_papers_results[author_id] = list(papers)

    def fetch_author_papers(
        self,
        author_id: str,
        *,
        limit: int = 100,
        fields: str = "paperId,title,year,venue,citationCount",
    ) -> list[dict[str, Any]]:
        """Canned author-papers lookup. Returns the registered list
        truncated to ``limit``, or ``[]`` when ``author_id`` has not
        been registered. ``fields`` is accepted for signature
        compatibility but ignored.
        """
        self._record_call("fetch_author_papers")
        papers = self._author_papers_results.get(author_id, [])
        return [copy.deepcopy(p) for p in papers[:limit]]

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Convenience corpus builders
# ---------------------------------------------------------------------------


def build_chain_corpus() -> FakeS2Client:
    """A tiny chain ``SEED`` — cited-by — ``CITER1`` — cited-by — ``CITER2``.

    Plus a couple of references off each of those, so both forward and
    backward expansion have something to chew on.

    Structure:
        REF1  ← SEED ← CITER1 ← CITER2
        REF2  ← SEED
                       REF3 ← CITER1
    """
    client = FakeS2Client()
    client.add(
        make_paper(
            "REF1",
            title="Foundational Reference 1",
            year=2015,
            citation_count=500,
            references=[],
        )
    )
    client.add(
        make_paper(
            "REF2",
            title="Foundational Reference 2",
            year=2016,
            citation_count=300,
            references=[],
        )
    )
    client.add(
        make_paper(
            "REF3",
            title="Reference for Citer 1",
            year=2017,
            citation_count=120,
            references=["REF1"],
        )
    )
    client.add(
        make_paper(
            "SEED",
            title="The Seed Paper On Transformers",
            year=2018,
            citation_count=1000,
            references=["REF1", "REF2"],
            embedding=[1.0, 0.0, 0.0],
        )
    )
    client.add(
        make_paper(
            "CITER1",
            title="New Transformer Architecture",
            year=2020,
            citation_count=250,
            references=["SEED", "REF3"],
            embedding=[0.9, 0.1, 0.0],
        )
    )
    client.add(
        make_paper(
            "CITER2",
            title="Another Follow-up Study",
            year=2022,
            citation_count=80,
            references=["SEED", "CITER1"],
            venue="arXiv",
            embedding=[0.7, 0.3, 0.0],
        )
    )
    # A low-quality paper that should fail citation / year thresholds
    client.add(
        make_paper(
            "WEAK",
            title="Obscure Paper",
            year=2010,
            citation_count=1,
            references=["REF1"],
            embedding=[0.0, 0.0, 1.0],
        )
    )
    # Author metadata for a couple of authors
    client.add_author(
        "SEED_au1",
        {"name": "SEED Author 1", "citationCount": 5000, "hIndex": 20,
         "paperCount": 60, "affiliations": ["Exeter"]},
    )
    client.add_author(
        "CITER1_au1",
        {"name": "CITER1 Author 1", "citationCount": 800, "hIndex": 10,
         "paperCount": 25, "affiliations": ["JIC"]},
    )
    return client
