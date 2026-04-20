"""SemanticScholarClient — public methods composing http + cache + converters."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx

from citeclaw.cache import Cache
from citeclaw.clients.s2.cache_layer import S2CacheLayer
from citeclaw.clients.s2.converters import edge_to_record, paper_to_record
from citeclaw.clients.s2.http import BATCH_URL, S2Http
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.models import PaperRecord, SemanticScholarAPIError

log = logging.getLogger("citeclaw.s2")

_MAX_BATCH = 500

PAPER_FIELDS = ",".join([
    "paperId", "title", "abstract", "venue", "year", "citationCount", "referenceCount",
    "influentialCitationCount", "openAccessPdf", "externalIds",
    "authors.authorId", "authors.name",
    # Subject-area + publication-type signals for the local query
    # engine and filters. ``fieldsOfStudy`` is the legacy flat list and
    # ``s2FieldsOfStudy`` is S2's richer per-source list — converters
    # merges both into PaperRecord.fields_of_study.
    "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes",
])
EDGE_LIGHT = ",".join(["paperId", "title", "year", "venue", "citationCount"])
EDGE_IDS_AND_COUNTS = ",".join(["paperId", "year", "citationCount"])
# Edge metadata is requested at the wrapper level (e.g. ``contexts,intents,isInfluential``)
# rather than the inner-paper level — these belong to the citation edge itself.
EDGE_META_FIELDS = "contexts,intents,isInfluential"
EMBEDDING_FIELDS = "paperId,embedding.specter_v2"
AUTHOR_FIELDS = "name,citationCount,hIndex,paperCount,affiliations"
AUTHOR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/author/batch"
_MAX_AUTHOR_BATCH = 1000

# Recommendations API lives outside ``/graph/v1`` so it must be addressed
# by full URL rather than the BASE_URL-relative path used by ``S2Http.get``.
RECOMMENDATIONS_BATCH_URL = "https://api.semanticscholar.org/recommendations/v1/papers"


class SemanticScholarClient:
    """Stateful S2 client with caching, rate-limiting, and budget tracking."""

    def __init__(self, config: Settings, cache: Cache, budget: BudgetTracker) -> None:
        self._config = config
        self._budget = budget
        self._http = S2Http(config, budget)
        self._cache = S2CacheLayer(cache, budget)

    # ------------------------------------------------------------------
    # Single-paper metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, paper_id: str) -> PaperRecord:
        cached = self._cache.get_metadata(paper_id)
        if cached is not None:
            rec = paper_to_record(cached)
            if rec:
                return rec

        data = self._http.get(f"/paper/{paper_id}", {"fields": PAPER_FIELDS}, req_type="metadata")
        canonical = data.get("paperId") or paper_id
        self._cache.put_metadata(canonical, data)
        if paper_id != canonical:
            self._cache.put_metadata(paper_id, data)
        rec = paper_to_record(data)
        if not rec:
            raise SemanticScholarAPIError(f"S2 returned no paperId for {paper_id}")
        return rec

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    # Whitelist of S2-recognized filter parameters that ``search_bulk``
    # forwards from its ``filters`` dict. Anything not in this set is
    # silently dropped — keeps callers from accidentally smuggling
    # unsupported keys into the request.
    _SEARCH_BULK_FILTER_KEYS = (
        "year",
        "venue",
        "fieldsOfStudy",
        "minCitationCount",
        "publicationTypes",
        "publicationDateOrYear",
        "openAccessPdf",
    )

    # Per S2 docs, ``/paper/search/bulk`` only sorts by these keys; the
    # default-sorted endpoint is ``/paper/search`` (relevance-ranked).
    # Anything else returns 400 Bad Request. We canonicalise to the bare
    # field name (allowing optional ``:asc`` / ``:desc`` suffixes) and drop
    # any unrecognised value rather than letting the agent crash the
    # request — invalid sort = no sort = stable result by paperId.
    _SEARCH_BULK_SORT_FIELDS = frozenset({"paperId", "publicationDate", "citationCount"})

    def _sanitize_bulk_sort(self, sort: str | None) -> str | None:
        """Return a valid ``/paper/search/bulk`` sort string or ``None``.

        Accepts ``"<field>"`` or ``"<field>:asc|desc"``. Strips invalid
        values silently rather than 400ing the whole search call. Logs
        the drop at INFO so a curious operator can spot bad agent
        suggestions in their logs.
        """
        if not sort:
            return None
        s = sort.strip()
        if not s:
            return None
        field, _, direction = s.partition(":")
        if field not in self._SEARCH_BULK_SORT_FIELDS:
            log.info("search_bulk: dropping invalid sort=%r (allowed: %s)",
                     sort, sorted(self._SEARCH_BULK_SORT_FIELDS))
            return None
        if direction and direction.lower() not in ("asc", "desc"):
            log.info("search_bulk: dropping invalid sort direction in %r", sort)
            return field  # keep the field, drop the direction
        return s if direction else field

    def search_bulk(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        sort: str | None = None,
        token: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Bulk paper search via ``GET /paper/search/bulk``.

        Returns the raw JSON response dict from S2 (typically containing
        ``data``, ``total``, and ``token`` keys). Only ``paperId,title``
        are requested — call ``enrich_batch`` afterwards for full
        metadata. Filters are forwarded only when present in
        :attr:`_SEARCH_BULK_FILTER_KEYS`.

        Cached: keyed by sha256 over ``{q, filters, sort, token}``
        (``limit`` is intentionally excluded so a wider call can serve
        a narrower follower from cache). Hits bump
        ``BudgetTracker._s2_cache["search"]``; misses go to the network
        and persist the response.
        """
        cache_key_payload = {"q": query, "filters": filters, "sort": sort, "token": token}
        query_hash = hashlib.sha256(
            json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = self._cache.get_search_results(query_hash)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "query": query,
            "fields": "paperId,title",
            "limit": limit,
        }
        if filters:
            for key in self._SEARCH_BULK_FILTER_KEYS:
                value = filters.get(key)
                if value is not None:
                    params[key] = value
        clean_sort = self._sanitize_bulk_sort(sort)
        if clean_sort is not None:
            params["sort"] = clean_sort
        if token is not None:
            params["token"] = token
        result = self._http.get("/paper/search/bulk", params, req_type="search")
        self._cache.put_search_results(query_hash, cache_key_payload, result)
        return result

    def search_match(self, title: str) -> dict[str, Any] | None:
        """Best-match paper lookup by title via ``GET /paper/search/match``.

        Returns the matched paper dict, or ``None`` when S2 has no
        match. Used by ``ResolveSeeds`` to convert fuzzy title seed
        entries into canonical paper ids. Does NOT cache — title
        resolutions are rare and freshness matters more than savings.
        """
        try:
            result = self._http.get(
                "/paper/search/match",
                {"query": title, "fields": PAPER_FIELDS},
                req_type="search_match",
            )
        except httpx.HTTPStatusError as exc:
            # S2 returns 404 when no paper matches the title — treat
            # that as a clean "no match" rather than an error.
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        data = result.get("data") if isinstance(result, dict) else None
        if isinstance(data, list) and data:
            return data[0]
        return None

    # ------------------------------------------------------------------
    # Recommendations — powers ExpandBySemantics; no caching, freshness wins.
    # ------------------------------------------------------------------

    def fetch_recommendations(
        self,
        positive_ids: list[str],
        *,
        negative_ids: list[str] | None = None,
        limit: int = 100,
        fields: str = "paperId,title",
    ) -> list[dict[str, Any]]:
        """Multi-anchor SPECTER2 kNN via ``POST /recommendations/v1/papers``.

        Posts ``{"positivePaperIds": [...], "negativePaperIds": [...]}`` and
        unwraps the ``recommendedPapers`` envelope so callers always
        receive a flat list. Use this for ``ExpandBySemantics`` where
        the input signal supplies multiple anchors.

        Does NOT cache — recommendations are non-deterministic and
        cheap relative to caching overhead.
        """
        body: dict[str, Any] = {"positivePaperIds": list(positive_ids)}
        if negative_ids:
            body["negativePaperIds"] = list(negative_ids)
        result = self._http.post(
            RECOMMENDATIONS_BATCH_URL,
            params={"fields": fields, "limit": limit},
            json_body=body,
            req_type="recommendations",
        )
        if isinstance(result, dict):
            recs = result.get("recommendedPapers")
            if isinstance(recs, list):
                return recs
        return []

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def fetch_references(self, paper_id: str) -> list[PaperRecord]:
        cached = self._cache.get_references(paper_id)
        if cached is not None:
            recs = [r for r in (edge_to_record(e, "citedPaper") for e in cached) if r]
            log.debug("fetch_references(%s): %d refs (cached)", paper_id[:20], len(recs))
            return recs

        raw = self._http.paginate(
            paper_id, "references",
            fields=f"citedPaper.{EDGE_LIGHT},{EDGE_META_FIELDS}",
        )
        self._cache.put_references(paper_id, raw)
        recs = [r for r in (edge_to_record(e, "citedPaper") for e in raw) if r]
        log.debug("fetch_references(%s): %d refs (api, %d raw edges)", paper_id[:20], len(recs), len(raw))
        return recs

    def fetch_reference_ids(self, paper_id: str) -> list[str]:
        cached = self._cache.get_references(paper_id)
        if cached is None:
            cached = self._http.paginate(
                paper_id, "references",
                fields=f"citedPaper.{EDGE_LIGHT},{EDGE_META_FIELDS}",
            )
            self._cache.put_references(paper_id, cached)
        return [
            e["citedPaper"]["paperId"]
            for e in cached
            if e.get("citedPaper") and e["citedPaper"].get("paperId")
        ]

    def fetch_reference_edges(self, paper_id: str) -> list[dict[str, Any]]:
        """Return cached reference edge metadata for ``paper_id``.

        Reads from the existing ``paper_references`` cache. Returns
        ``[{target_id, contexts, intents, is_influential}, ...]``. When the
        cached entry pre-dates the edge-metadata field expansion, the rich
        fields come back empty / False — we never re-fetch.
        """
        cached = self._cache.get_references(paper_id)
        if cached is None:
            return []
        out: list[dict[str, Any]] = []
        for e in cached:
            inner = e.get("citedPaper") or {}
            tid = inner.get("paperId")
            if not tid:
                continue
            out.append({
                "target_id": tid,
                "contexts": e.get("contexts") or [],
                "intents": e.get("intents") or [],
                "is_influential": bool(e.get("isInfluential", False)),
            })
        return out

    def has_cached_references(self, paper_id: str) -> bool:
        return self._cache.has_references(paper_id)

    def cached_reference_ids(self, paper_id: str) -> list[str] | None:
        """Return cached reference IDs without ever hitting the S2 API.

        Returns ``None`` if the paper's references aren't cached. Used by
        the saturation metric so it never triggers surprise S2 calls.
        """
        cached = self._cache.get_references(paper_id)
        if cached is None:
            return None
        return [
            e["citedPaper"]["paperId"]
            for e in cached
            if e.get("citedPaper") and e["citedPaper"].get("paperId")
        ]

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    def fetch_citation_ids_and_counts(
        self, paper_id: str, *, max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cache.get_citations(paper_id)
        if cached is None:
            cached = self._http.paginate(
                paper_id, "citations",
                fields=f"citingPaper.{EDGE_IDS_AND_COUNTS},{EDGE_META_FIELDS}",
                max_items=max_items,
            )
            self._cache.put_citations(paper_id, cached)
            log.debug("fetch_citations(%s): %d citers (api)", paper_id[:20], len(cached))
        else:
            log.debug("fetch_citations(%s): %d citers (cached)", paper_id[:20], len(cached))
        subset = cached[:max_items] if max_items else cached
        return [
            {
                "paper_id": e["citingPaper"]["paperId"],
                "citation_count": e["citingPaper"].get("citationCount"),
                "year": e["citingPaper"].get("year"),
            }
            for e in subset
            if e.get("citingPaper") and e["citingPaper"].get("paperId")
        ]

    def fetch_citation_edges(self, paper_id: str) -> list[dict[str, Any]]:
        """Return cached forward-citation edge metadata for ``paper_id``.

        Reads from the existing ``paper_citations`` cache. Returns
        ``[{source_id, contexts, intents, is_influential}, ...]`` where each
        source_id is the citing paper. When the cached entry pre-dates the
        edge-metadata field expansion, the rich fields come back empty /
        False — we never re-fetch.
        """
        cached = self._cache.get_citations(paper_id)
        if cached is None:
            return []
        out: list[dict[str, Any]] = []
        for e in cached:
            inner = e.get("citingPaper") or {}
            sid = inner.get("paperId")
            if not sid:
                continue
            out.append({
                "source_id": sid,
                "contexts": e.get("contexts") or [],
                "intents": e.get("intents") or [],
                "is_influential": bool(e.get("isInfluential", False)),
            })
        return out

    def has_cached_citations(self, paper_id: str) -> bool:
        return self._cache.has_citations(paper_id)

    # ------------------------------------------------------------------
    # Batch / enrichment
    # ------------------------------------------------------------------

    def enrich_with_abstracts(self, records: list[PaperRecord]) -> list[PaperRecord]:
        needs = [r for r in records if not r.abstract]
        if not needs:
            return records

        # 1) try local metadata cache first
        still_missing: list[PaperRecord] = []
        for rec in needs:
            cached = self._cache.get_metadata(rec.paper_id)
            if cached and cached.get("abstract"):
                rec.abstract = cached["abstract"]
                if not rec.venue and cached.get("venue"):
                    rec.venue = cached["venue"]
                if rec.citation_count is None and cached.get("citationCount") is not None:
                    rec.citation_count = cached["citationCount"]
                if rec.year is None and cached.get("year") is not None:
                    rec.year = cached["year"]
            else:
                still_missing.append(rec)

        if not still_missing:
            return records

        # 2) batch-fetch from S2 and persist whatever the API returns
        ids = [r.paper_id for r in still_missing]
        fetched = self._batch_fetch(ids, fields=PAPER_FIELDS)
        id_to_data = {d["paperId"]: d for d in fetched if d and d.get("paperId")}
        for rec in still_missing:
            data = id_to_data.get(rec.paper_id)
            if not data:
                continue
            self._cache.put_metadata(rec.paper_id, data)
            if data.get("abstract"):
                rec.abstract = data["abstract"]
            if not rec.venue and data.get("venue"):
                rec.venue = data["venue"]
            if rec.citation_count is None and data.get("citationCount") is not None:
                rec.citation_count = data["citationCount"]
            if rec.year is None and data.get("year") is not None:
                rec.year = data["year"]

        # 3) OpenAlex fallback for records still missing an abstract. We
        # only attempt this for papers that carry a DOI in external_ids
        # — OpenAlex's lookup path is DOI-keyed, and trying arbitrary
        # titles would waste rate budget on false matches. Records
        # without a DOI stay un-enriched (the pipeline tolerates None
        # abstracts throughout). Failures log at info and never propagate.
        oa_needs = [
            r for r in still_missing
            if not r.abstract and r.external_ids and r.external_ids.get("DOI")
        ]
        if oa_needs:
            self._openalex_abstract_fallback(oa_needs)

        return records

    def _openalex_abstract_fallback(self, records: list[PaperRecord]) -> None:
        """Best-effort fallback: query OpenAlex for abstracts we couldn't
        get from S2. Imports are deferred so configs without OpenAlex
        credentials don't pay the import cost.
        """
        try:
            from citeclaw.clients.openalex import OpenAlexClient
        except ImportError as exc:  # pragma: no cover - defensive
            log.debug("OpenAlex client unavailable: %s", exc)
            return
        client = OpenAlexClient(self._config)
        try:
            for rec in records:
                doi = rec.external_ids.get("DOI")
                if not doi:
                    continue
                try:
                    abstract = client.fetch_abstract_by_doi(doi)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    log.debug("OpenAlex abstract fetch failed for %s: %s",
                              rec.paper_id[:20], exc)
                    continue
                if abstract:
                    rec.abstract = abstract
                    # Also write-through to the metadata cache so rerun
                    # doesn't re-hit OpenAlex.
                    try:
                        cached = self._cache.get_metadata(rec.paper_id) or {}
                        cached["abstract"] = abstract
                        self._cache.put_metadata(rec.paper_id, cached)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            client.close()

    def enrich_batch(self, candidates: list[dict[str, Any]]) -> list[PaperRecord]:
        """Hydrate ``candidates`` (list of ``{paper_id: ...}`` dicts) into
        :class:`PaperRecord` objects, serving from ``paper_metadata``
        cache where possible.

        Cache-first matters here because ExpandForward / ExpandBackward
        call this on every citer / reference; a second run with the
        same seeds would otherwise re-fetch the whole candidate set
        against the 1 rps S2 cap.
        """
        ids = [c.get("paper_id", "") for c in candidates if c.get("paper_id")]
        if not ids:
            return []

        records: list[PaperRecord] = []
        missing_ids: list[str] = []

        # 1) Try local metadata cache for every id first.
        for pid in ids:
            cached = self._cache.get_metadata(pid)
            if cached and cached.get("paperId"):
                rec = paper_to_record(cached)
                if rec:
                    records.append(rec)
                    continue
            missing_ids.append(pid)

        if not missing_ids:
            return records

        # 2) Batch-fetch only the misses, then persist + materialise.
        for data in self._batch_fetch(missing_ids, fields=PAPER_FIELDS):
            if not data or not data.get("paperId"):
                continue
            rec = paper_to_record(data)
            if rec:
                self._cache.put_metadata(rec.paper_id, data)
                records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Embeddings (SPECTER2 via S2)
    # ------------------------------------------------------------------

    def fetch_embedding(self, paper_id: str) -> list[float] | None:
        """Return S2's SPECTER2 embedding for this paper, or None if unavailable.

        Uses the same persistent cache table as everything else; a confirmed
        'no embedding' is stored as ``[]`` so we never re-request it.
        """
        if self._cache.has_embedding(paper_id):
            return self._cache.get_embedding(paper_id)
        result = self.fetch_embeddings_batch([paper_id])
        return result.get(paper_id)

    def fetch_embeddings_batch(self, paper_ids: list[str]) -> dict[str, list[float] | None]:
        """Batch-fetch SPECTER2 embeddings. Returns {paper_id: vector | None}.

        Papers already in the cache are served from it. Uncached papers are
        fetched via the S2 batch endpoint (up to _MAX_BATCH at a time); the
        results, including 'no embedding' sentinels, are persisted.
        """
        out: dict[str, list[float] | None] = {}
        missing: list[str] = []
        for pid in paper_ids:
            if not pid:
                continue
            if self._cache.has_embedding(pid):
                out[pid] = self._cache.get_embedding(pid)
            else:
                missing.append(pid)
        if not missing:
            return out

        fetched = self._batch_fetch(missing, fields=EMBEDDING_FIELDS)
        returned_ids: set[str] = set()
        for data in fetched:
            if not data:
                continue
            pid = data.get("paperId")
            if not pid:
                continue
            returned_ids.add(pid)
            emb = data.get("embedding")
            vec = emb.get("vector") if isinstance(emb, dict) else None
            if isinstance(vec, list) and vec:
                self._cache.put_embedding(pid, vec)
                out[pid] = vec
            else:
                # Confirmed no embedding — store sentinel
                self._cache.put_embedding(pid, [])
                out[pid] = None

        # Papers that S2 didn't return at all — also record as 'no embedding'
        # so we don't pester the API again.
        for pid in missing:
            if pid not in returned_ids:
                self._cache.put_embedding(pid, [])
                out.setdefault(pid, None)

        return out

    def has_cached_embedding(self, paper_id: str) -> bool:
        return self._cache.has_embedding(paper_id)

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    def fetch_authors_batch(self, author_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-fetch S2 author metadata. Returns ``{author_id: data}``.

        Cache-first: each author is served from the persistent
        ``author_metadata`` table when available; uncached authors are
        POSTed to ``/author/batch`` in chunks of 1000.
        """
        out: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        seen: set[str] = set()
        for aid in author_ids:
            if not aid or aid in seen:
                continue
            seen.add(aid)
            cached = self._cache.get_author_metadata(aid)
            if cached is not None:
                out[aid] = cached
            else:
                missing.append(aid)

        if not missing:
            return out

        for i in range(0, len(missing), _MAX_AUTHOR_BATCH):
            chunk = missing[i: i + _MAX_AUTHOR_BATCH]
            try:
                batch_result = self._http.post(
                    AUTHOR_BATCH_URL,
                    params={"fields": AUTHOR_FIELDS},
                    json_body={"ids": chunk},
                    req_type="author_metadata",
                )
            except Exception as exc:
                log.warning("S2 author batch failed for %d ids: %s", len(chunk), exc)
                continue
            if not isinstance(batch_result, list):
                log.warning("S2 author batch returned non-list: %s", type(batch_result))
                continue
            # Response order matches request order; entries may be None for
            # unknown ids.
            for aid, data in zip(chunk, batch_result):
                if not data:
                    self._cache.put_author_metadata(aid, {})
                    continue
                self._cache.put_author_metadata(aid, data)
                out[aid] = data
        return out

    # ------------------------------------------------------------------
    # Author papers — powers ExpandByAuthor.
    # ------------------------------------------------------------------

    # S2's max page size for the author/papers endpoint. Mirrors
    # PAGE_SIZE in http.py.
    _AUTHOR_PAPERS_PAGE_SIZE = 100

    def fetch_author_papers(
        self,
        author_id: str,
        *,
        limit: int = 100,
        fields: str = "paperId,title,year,venue,citationCount",
    ) -> list[dict[str, Any]]:
        """Fetch papers authored by ``author_id`` via
        ``GET /author/{author_id}/papers``.

        Cache-first: when a per-author entry exists in the
        ``author_papers`` table, the cached list is returned (sliced
        to ``limit``) with no S2 traffic. Otherwise the endpoint is
        paginated via S2's ``offset``/``limit``, capped at ``limit``
        so very prolific authors don't blow up the budget. The
        collected result is then persisted under ``author_id``.

        ``limit`` is the upper bound on the *returned* list — pagination
        stops as soon as ``limit`` items are accumulated. ``fields``
        defaults to a lightweight projection suitable for ranking; pass a
        wider field list when you need abstracts.
        """
        cached = self._cache.get_author_papers(author_id)
        if cached is not None:
            return cached[:limit] if limit and limit > 0 else cached

        results: list[dict[str, Any]] = []
        offset = 0
        page_size = self._AUTHOR_PAPERS_PAGE_SIZE
        while True:
            data = self._http.get(
                f"/author/{author_id}/papers",
                {"fields": fields, "limit": page_size, "offset": offset},
                req_type="author_papers",
            )
            batch = data.get("data", []) if isinstance(data, dict) else []
            if not batch:
                break
            results.extend(batch)
            if limit and len(results) >= limit:
                results = results[:limit]
                break
            if len(batch) < page_size:
                break
            offset += len(batch)

        self._cache.put_author_papers(author_id, results)
        return results

    def _batch_fetch(self, paper_ids: list[str], *, fields: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for i in range(0, len(paper_ids), _MAX_BATCH):
            chunk = paper_ids[i: i + _MAX_BATCH]
            try:
                batch_result = self._http.post(
                    BATCH_URL, params={"fields": fields}, json_body={"ids": chunk},
                )
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    log.warning("S2 batch returned non-list: %s", type(batch_result))
            except Exception as exc:
                log.warning("S2 batch fetch failed for %d papers: %s", len(chunk), exc)
                for pid in chunk:
                    try:
                        results.append(
                            self._http.get(f"/paper/{pid}", {"fields": fields}, req_type="metadata")
                        )
                    except Exception as inner_exc:
                        log.debug("S2 individual fetch failed for %s: %s", pid, inner_exc)
        return results

    def close(self) -> None:
        self._http.close()
