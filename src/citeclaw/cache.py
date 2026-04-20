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

# Default freshness window for cached search results.
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
CREATE TABLE IF NOT EXISTS paper_full_text (
    -- Cached full-text body for open-access PDFs. ``text`` is NULL when
    -- fetch/parse failed; ``error`` records why ("no_pdf",
    -- "download_failed", "parse_failed", "too_large") so a second pass
    -- doesn't redo a known-failing fetch.
    paper_id   TEXT PRIMARY KEY,
    text       TEXT,
    error      TEXT,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_response_cache (
    -- Content-addressable LLM prompt cache. ``cache_key`` is a sha256
    -- hash over (model, reasoning_effort, system, user, response_schema,
    -- with_logprobs) computed by ``citeclaw.clients.llm.caching.make_cache_key``.
    -- Repeat calls with the same prompt skip the LLM entirely and serve
    -- the response from this table — the chief savings on iterative
    -- runs and on multi-pass screening.
    cache_key         TEXT PRIMARY KEY,
    model             TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    reasoning_content TEXT,
    logprob_tokens    TEXT,
    fetched_at        TEXT NOT NULL
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

    def _has(self, table: str, paper_id: str, *, key_col: str = "paper_id") -> bool:
        """SELECT 1 existence check; cheaper than _get for has_*-style lookups."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE {key_col} = ? LIMIT 1", (paper_id,)
            )
            return cur.fetchone() is not None

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
        return self._has("paper_references", paper_id)

    def has_citations(self, paper_id: str) -> bool:
        """Check if citations exist in cache without reading them."""
        return self._has("paper_citations", paper_id)

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
        return self._has("paper_embeddings", paper_id)

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
        return self._has("author_metadata", author_id, key_col="author_id")

    # --- search query results — keyed by hash, TTL-aware ---

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
        """Return cached search response for ``query_hash``, or None on
        miss / expired entry. The TTL knob lets callers override the
        default freshness window."""
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

    # --- author papers — full paper list per author ---

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

    # --- full-text PDF body ---

    def get_full_text(self, paper_id: str) -> dict[str, Any] | None:
        """Return ``{"text": str | None, "error": str | None}`` for the
        cached PDF parse, or ``None`` if we have never tried this paper.

        A row exists for both successful and failed fetches: success
        stores the parsed body in ``text`` (with ``error=None``);
        failure stores the failure category in ``error`` (with
        ``text=None``). The caller treats the row's presence as
        "we already know the answer for this paper, don't refetch".
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT text, error FROM paper_full_text WHERE paper_id = ?",
                (paper_id,),
            )
            row = cur.fetchone()
        if row is None:
            log.debug("Cache MISS [paper_full_text] %s", paper_id)
            return None
        log.debug("Cache HIT [paper_full_text] %s", paper_id)
        return {"text": row[0], "error": row[1]}

    def put_full_text(
        self,
        paper_id: str,
        *,
        text: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist a full-text fetch outcome.

        Pass ``text`` for a successful parse; pass ``error`` (one of
        ``"no_pdf"``, ``"download_failed"``, ``"parse_failed"``,
        ``"too_large"``) for a failure. Exactly one of the two should be
        non-None — the caller decides which based on the fetch outcome.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO paper_full_text "
                "(paper_id, text, error, fetched_at) VALUES (?, ?, ?, ?)",
                (paper_id, text, error, now),
            )
        log.debug(
            "Cache STORE [paper_full_text] %s (text=%d chars, error=%r)",
            paper_id, len(text or ""), error,
        )

    # --- LLM prompt cache ---

    def get_llm_response(self, cache_key: str) -> dict[str, Any] | None:
        """Return the cached LLM response for ``cache_key`` or ``None``.

        Returns a dict with keys ``text`` (str), ``reasoning_content``
        (str, may be empty), and ``logprob_tokens`` (list, may be
        empty). The caller reconstructs an :class:`LLMResponse` from it.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT response_text, reasoning_content, logprob_tokens "
                "FROM llm_response_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        if row is None:
            log.debug("Cache MISS [llm_response_cache] %s", cache_key[:12])
            return None
        log.debug("Cache HIT [llm_response_cache] %s", cache_key[:12])
        try:
            logprobs = json.loads(row[2]) if row[2] else []
        except (json.JSONDecodeError, TypeError):
            logprobs = []
        return {
            "text": row[0],
            "reasoning_content": row[1] or "",
            "logprob_tokens": logprobs,
        }

    def put_llm_response(
        self,
        cache_key: str,
        *,
        model: str,
        text: str,
        reasoning_content: str = "",
        logprob_tokens: list | None = None,
    ) -> None:
        """Persist a fresh LLM response under ``cache_key``.

        ``logprob_tokens`` is JSON-serialised when present; pass ``None``
        (or an empty list) when the call didn't request logprobs.
        """
        now = datetime.now(timezone.utc).isoformat()
        logprob_blob = (
            json.dumps(logprob_tokens) if logprob_tokens else None
        )
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO llm_response_cache "
                "(cache_key, model, response_text, reasoning_content, "
                "logprob_tokens, fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cache_key, model, text,
                    reasoning_content or None,
                    logprob_blob, now,
                ),
            )
        log.debug(
            "Cache STORE [llm_response_cache] %s (%d chars)",
            cache_key[:12], len(text or ""),
        )

    def close(self) -> None:
        self._conn.close()
