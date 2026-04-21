"""Read-through cache helpers shared by :class:`SemanticScholarClient`.

Wraps :class:`citeclaw.cache.Cache` so every cache hit gets booked to
the same per-bucket :class:`BudgetTracker` slot as a network call
would — just with ``cached=True`` so it counts toward call frequency
without spending the API quota. The seven buckets (``metadata``,
``references``, ``citations``, ``embeddings``, ``author_metadata``,
``author_papers``, ``search``) are pinned by ``tests/test_s2_cache_layer.py``
and consumed verbatim by :func:`BudgetTracker.detailed_summary`, so
the strings here are load-bearing — never rename them silently.

``put_*`` / ``has_*`` methods are thin pass-throughs to the underlying
:class:`Cache`; they exist on this layer so callers only ever talk to
``S2CacheLayer`` and never reach across to :class:`Cache` directly.
"""

from __future__ import annotations

from typing import Any, TypeVar

from citeclaw.cache import Cache
from citeclaw.budget import BudgetTracker

T = TypeVar("T")


class S2CacheLayer:
    """Thin wrapper around :class:`Cache` that records cache hits in budget."""

    def __init__(self, cache: Cache, budget: BudgetTracker) -> None:
        self._cache = cache
        self._budget = budget

    def _record_hit(self, value: T | None, bucket: str) -> T | None:
        """Book a free cache hit on ``bucket`` when ``value`` is not None.

        Used by every read-through path except :meth:`get_embedding`,
        which has a "confirmed missing" sentinel of ``[]`` and so must
        gate the booking on :meth:`Cache.has_embedding` instead.
        """
        if value is not None:
            self._budget.record_s2(bucket, cached=True)
        return value

    # ----- metadata -----
    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        return self._record_hit(self._cache.get_metadata(paper_id), "metadata")

    def put_metadata(self, paper_id: str, data: dict[str, Any]) -> None:
        self._cache.put_metadata(paper_id, data)

    # ----- references -----
    def get_references(self, paper_id: str) -> list[dict[str, Any]] | None:
        return self._record_hit(self._cache.get_references(paper_id), "references")

    def put_references(self, paper_id: str, edges: list[dict[str, Any]]) -> None:
        self._cache.put_references(paper_id, edges)

    def has_references(self, paper_id: str) -> bool:
        return self._cache.has_references(paper_id)

    # ----- citations -----
    def get_citations(self, paper_id: str) -> list[dict[str, Any]] | None:
        return self._record_hit(self._cache.get_citations(paper_id), "citations")

    def put_citations(self, paper_id: str, edges: list[dict[str, Any]]) -> None:
        self._cache.put_citations(paper_id, edges)

    def has_citations(self, paper_id: str) -> bool:
        return self._cache.has_citations(paper_id)

    # ----- embeddings -----
    def get_embedding(self, paper_id: str) -> list[float] | None:
        """Return the cached embedding (``None`` when uncached, ``[]`` when
        confirmed-missing) and book a hit if either sentinel applies.

        Special-case relative to :meth:`_record_hit`: the empty-list
        sentinel from :meth:`Cache.put_embedding([])` ALSO counts as a
        billed hit (we don't want to re-fetch a paper S2 has already
        told us has no SPECTER2 vector). :meth:`Cache.has_embedding`
        is the right truthiness — it returns True for both real
        vectors and the empty-list sentinel.
        """
        cached = self._cache.get_embedding(paper_id)
        if self._cache.has_embedding(paper_id):
            self._budget.record_s2("embeddings", cached=True)
        return cached

    def has_embedding(self, paper_id: str) -> bool:
        return self._cache.has_embedding(paper_id)

    def put_embedding(self, paper_id: str, vector: list[float]) -> None:
        self._cache.put_embedding(paper_id, vector)

    # ----- author metadata -----
    def get_author_metadata(self, author_id: str) -> dict[str, Any] | None:
        return self._record_hit(
            self._cache.get_author_metadata(author_id), "author_metadata",
        )

    def put_author_metadata(self, author_id: str, data: dict[str, Any]) -> None:
        self._cache.put_author_metadata(author_id, data)

    def has_author_metadata(self, author_id: str) -> bool:
        return self._cache.has_author_metadata(author_id)

    # ----- author papers -----
    def get_author_papers(self, author_id: str) -> list[dict[str, Any]] | None:
        return self._record_hit(
            self._cache.get_author_papers(author_id), "author_papers",
        )

    def put_author_papers(
        self, author_id: str, papers: list[dict[str, Any]],
    ) -> None:
        self._cache.put_author_papers(author_id, papers)

    # ----- search results -----
    def get_search_results(self, query_hash: str) -> dict[str, Any] | None:
        return self._record_hit(
            self._cache.get_search_results(query_hash), "search",
        )

    def put_search_results(
        self, query_hash: str, query: dict[str, Any], result: dict[str, Any],
    ) -> None:
        self._cache.put_search_results(query_hash, query, result)
