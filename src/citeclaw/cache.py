"""SQLite cache for API responses (Semantic Scholar / OpenAlex)."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

log = logging.getLogger("citeclaw.cache")

# Default freshness window for cached search results (PA-04). PA-05 will
# reuse this when wiring search_bulk through the cache layer.
_SEARCH_TTL_DAYS_DEFAULT = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_metadata (
    paper_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_references (
    paper_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_citations (
    paper_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS author_metadata (
    author_id  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_queries (
    query_hash TEXT PRIMARY KEY,
    query_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS author_papers (
    author_id  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


class Cache:
    """Thread-safe SQLite cache with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.info("Cache opened: %s", db_path)

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # --- generic helpers ---

    def _get(self, table: str, paper_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute(f"SELECT data FROM {table} WHERE paper_id = ?", (paper_id,))
            row = cur.fetchone()
        if row:
            log.debug("Cache HIT [%s] %s", table, paper_id)
            return json.loads(row[0])
        log.debug("Cache MISS [%s] %s", table, paper_id)
        return None

    def _put(self, table: str, paper_id: str, data: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                f"INSERT OR REPLACE INTO {table} (paper_id, data, fetched_at) VALUES (?, ?, ?)",
                (paper_id, json.dumps(data), now),
            )
        log.debug("Cache STORE [%s] %s", table, paper_id)

    # --- public API ---

    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        return self._get("paper_metadata", paper_id)

    def put_metadata(self, paper_id: str, data: dict[str, Any]) -> None:
        self._put("paper_metadata", paper_id, data)

    def get_references(self, paper_id: str) -> list[dict[str, Any]] | None:
        return self._get("paper_references", paper_id)

    def put_references(self, paper_id: str, data: list[dict[str, Any]]) -> None:
        self._put("paper_references", paper_id, data)

    def get_citations(self, paper_id: str) -> list[dict[str, Any]] | None:
        return self._get("paper_citations", paper_id)

    def put_citations(self, paper_id: str, data: list[dict[str, Any]]) -> None:
        self._put("paper_citations", paper_id, data)

    def has_references(self, paper_id: str) -> bool:
        """Check if references exist in cache without reading them."""
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM paper_references WHERE paper_id = ? LIMIT 1", (paper_id,))
            return cur.fetchone() is not None

    def has_citations(self, paper_id: str) -> bool:
        """Check if citations exist in cache without reading them."""
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM paper_citations WHERE paper_id = ? LIMIT 1", (paper_id,))
            return cur.fetchone() is not None

    def get_embedding(self, paper_id: str) -> list[float] | None:
        """Return cached embedding, or None if not cached.

        A stored empty list ``[]`` is the sentinel for 'confirmed no embedding
        available from S2' — still returned as None so callers treat it
        uniformly, but persists so we don't re-fetch.
        """
        data = self._get("paper_embeddings", paper_id)
        if data is None:
            return None
        if isinstance(data, list) and data:
            return data
        return None  # sentinel (empty list) → no embedding available

    def has_embedding(self, paper_id: str) -> bool:
        """True if we've recorded a lookup (hit or confirmed miss)."""
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM paper_embeddings WHERE paper_id = ? LIMIT 1", (paper_id,))
            return cur.fetchone() is not None

    def put_embedding(self, paper_id: str, vector: list[float]) -> None:
        """Store an embedding. Pass ``[]`` to record 'confirmed no embedding'."""
        self._put("paper_embeddings", paper_id, vector)

    # --- author metadata (keyed by author_id, not paper_id) ---

    def get_author_metadata(self, author_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT data FROM author_metadata WHERE author_id = ?", (author_id,))
            row = cur.fetchone()
        if row:
            log.debug("Cache HIT [author_metadata] %s", author_id)
            return json.loads(row[0])
        log.debug("Cache MISS [author_metadata] %s", author_id)
        return None

    def put_author_metadata(self, author_id: str, data: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO author_metadata (author_id, data, fetched_at) VALUES (?, ?, ?)",
                (author_id, json.dumps(data), now),
            )
        log.debug("Cache STORE [author_metadata] %s", author_id)

    def has_author_metadata(self, author_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM author_metadata WHERE author_id = ? LIMIT 1", (author_id,))
            return cur.fetchone() is not None

    # --- search query results (PA-04) — keyed by hash, TTL-aware ---

    def _is_fresh(self, fetched_at_iso: str, ttl_days: int) -> bool:
        """True iff ``fetched_at_iso`` is within ``ttl_days`` of now."""
        try:
            fetched = datetime.fromisoformat(fetched_at_iso)
        except ValueError:
            return False
        # Be tolerant of naive timestamps in older rows.
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched <= timedelta(days=ttl_days)

    def get_search_results(
        self, query_hash: str, ttl_days: int = _SEARCH_TTL_DAYS_DEFAULT,
    ) -> dict[str, Any] | None:
        """Return cached search response for ``query_hash``, or None on miss
        / expired entry. The TTL knob exists so callers can override the
        default freshness window when policy demands it (e.g. PA-05)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT result_json, fetched_at FROM search_queries WHERE query_hash = ?",
                (query_hash,),
            )
            row = cur.fetchone()
        if row is None:
            log.debug("Cache MISS [search_queries] %s", query_hash[:12])
            return None
        result_json, fetched_at = row
        if not self._is_fresh(fetched_at, ttl_days):
            log.debug("Cache STALE [search_queries] %s (ttl=%d)", query_hash[:12], ttl_days)
            return None
        log.debug("Cache HIT [search_queries] %s", query_hash[:12])
        return json.loads(result_json)

    def put_search_results(
        self, query_hash: str, query: dict[str, Any], result: dict[str, Any],
    ) -> None:
        """Persist a search response keyed by its query hash. ``query`` is
        the original query dict (stored verbatim for debuggability) and
        ``result`` is the JSON-serialisable response payload."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO search_queries "
                "(query_hash, query_json, result_json, fetched_at) VALUES (?, ?, ?, ?)",
                (query_hash, json.dumps(query), json.dumps(result), now),
            )
        log.debug("Cache STORE [search_queries] %s", query_hash[:12])

    def has_search_results(
        self, query_hash: str, ttl_days: int = _SEARCH_TTL_DAYS_DEFAULT,
    ) -> bool:
        """True iff a non-expired result exists for ``query_hash``."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT fetched_at FROM search_queries WHERE query_hash = ? LIMIT 1",
                (query_hash,),
            )
            row = cur.fetchone()
        if row is None:
            return False
        return self._is_fresh(row[0], ttl_days)

    # --- author papers (PA-04) — full paper list per author ---

    def get_author_papers(self, author_id: str) -> list[dict[str, Any]] | None:
        """Return cached paper list for ``author_id``, or None on miss."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT data FROM author_papers WHERE author_id = ?", (author_id,),
            )
            row = cur.fetchone()
        if row is None:
            log.debug("Cache MISS [author_papers] %s", author_id)
            return None
        log.debug("Cache HIT [author_papers] %s", author_id)
        return json.loads(row[0])

    def put_author_papers(
        self, author_id: str, data: list[dict[str, Any]],
    ) -> None:
        """Persist the full S2 paper list for ``author_id``."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO author_papers (author_id, data, fetched_at) "
                "VALUES (?, ?, ?)",
                (author_id, json.dumps(data), now),
            )
        log.debug("Cache STORE [author_papers] %s", author_id)

    def close(self) -> None:
        self._conn.close()
