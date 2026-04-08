"""Read-through cache helpers shared by the S2 API."""

from __future__ import annotations

from typing import Any

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker


class S2CacheLayer:
    """Thin wrapper around :class:`Cache` that records cache hits in budget."""

    def __init__(self, cache: Cache, budget: BudgetTracker) -> None:
        self._cache = cache
        self._budget = budget

    # ----- metadata -----
    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        cached = self._cache.get_metadata(paper_id)
        if cached is not None:
            self._budget.record_s2("metadata", cached=True)
        return cached

    def put_metadata(self, paper_id: str, data: dict[str, Any]) -> None:
        self._cache.put_metadata(paper_id, data)

    # ----- references -----
    def get_references(self, paper_id: str) -> list[dict[str, Any]] | None:
        cached = self._cache.get_references(paper_id)
        if cached is not None:
            self._budget.record_s2("references", cached=True)
        return cached

    def put_references(self, paper_id: str, edges: list[dict[str, Any]]) -> None:
        self._cache.put_references(paper_id, edges)

    def has_references(self, paper_id: str) -> bool:
        return self._cache.has_references(paper_id)

    # ----- citations -----
    def get_citations(self, paper_id: str) -> list[dict[str, Any]] | None:
        cached = self._cache.get_citations(paper_id)
        if cached is not None:
            self._budget.record_s2("citations", cached=True)
        return cached

    def put_citations(self, paper_id: str, edges: list[dict[str, Any]]) -> None:
        self._cache.put_citations(paper_id, edges)

    def has_citations(self, paper_id: str) -> bool:
        return self._cache.has_citations(paper_id)

    # ----- embeddings -----
    def get_embedding(self, paper_id: str) -> list[float] | None:
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
        cached = self._cache.get_author_metadata(author_id)
        if cached is not None:
            self._budget.record_s2("author_metadata", cached=True)
        return cached

    def put_author_metadata(self, author_id: str, data: dict[str, Any]) -> None:
        self._cache.put_author_metadata(author_id, data)

    def has_author_metadata(self, author_id: str) -> bool:
        return self._cache.has_author_metadata(author_id)

    # ----- author papers (PA-03) -----
    def get_author_papers(self, author_id: str) -> list[dict[str, Any]] | None:
        cached = self._cache.get_author_papers(author_id)
        if cached is not None:
            self._budget.record_s2("author_papers", cached=True)
        return cached

    def put_author_papers(
        self, author_id: str, papers: list[dict[str, Any]],
    ) -> None:
        self._cache.put_author_papers(author_id, papers)

    # ----- search results (PA-05) -----
    def get_search_results(self, query_hash: str) -> dict[str, Any] | None:
        cached = self._cache.get_search_results(query_hash)
        if cached is not None:
            self._budget.record_s2("search", cached=True)
        return cached

    def put_search_results(
        self, query_hash: str, query: dict[str, Any], result: dict[str, Any],
    ) -> None:
        self._cache.put_search_results(query_hash, query, result)
