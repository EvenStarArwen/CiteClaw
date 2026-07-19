"""Read-only access layer over the built shard DBs.

One ``sqlite3`` connection per shard file, opened lazily with
``mode=ro&immutable=1`` (the store is never written after a build;
immutable mode skips all locking, which also makes cross-thread sharing
safe — Python's sqlite3 serializes calls per connection).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from s2mirror import jsonio, schema
from s2mirror.reducer import ADJ_DTYPE

_ADJ_CACHE_MAX = 2048


class MirrorStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta: dict = {}
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
        # shard name -> (connection | None, lock). A sqlite3 connection is
        # NOT safe for concurrent statement execution across threads (it
        # raises "bad parameter or other API misuse" under load), so every
        # query runs under its shard's lock with the cursor fully consumed
        # before release. Different shards still query in parallel.
        self._conns: dict[str, tuple[sqlite3.Connection | None, threading.Lock | None]] = {}
        self._conn_lock = threading.Lock()
        self._adj_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._adj_lock = threading.Lock()
        # shared fan-out pool: creating a ThreadPoolExecutor per get_papers
        # call costs ~1ms + thread churn on the hottest path in the server.
        self._pool = ThreadPoolExecutor(max_workers=48, thread_name_prefix="shardfan")

    # ---- connections -----------------------------------------------------

    def _q(self, name: str, sql: str, params=()) -> list[tuple]:
        pair = self._conns.get(name)
        if pair is None:
            with self._conn_lock:
                pair = self._conns.get(name)
                if pair is None:
                    path = self.root / f"{name}.db"
                    if path.exists():
                        conn = sqlite3.connect(
                            f"file:{path}?mode=ro&immutable=1",
                            uri=True, check_same_thread=False,
                        )
                        pair = (conn, threading.Lock())
                    else:
                        pair = (None, None)  # negative-cache missing shards
                    self._conns[name] = pair
        conn, lock = pair
        if conn is None:
            return []
        with lock:
            return conn.execute(sql, params).fetchall()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        with self._conn_lock:
            for conn, _lock in self._conns.values():
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._conns.clear()

    # ---- id resolution ---------------------------------------------------

    def resolve(self, raw_id: str) -> int | None:
        parsed = schema.parse_paper_id(raw_id)
        if parsed is None:
            return None
        kind, val = parsed
        if kind == "corpus":
            return int(val)
        rows = self._q(f"ids_{schema.id_shard(str(val)):02d}",
                       "SELECT corpusid FROM idmap WHERE k = ?", (val,))
        return rows[0][0] if rows else None

    # ---- papers ----------------------------------------------------------

    def get_paper(self, corpusid: int) -> dict | None:
        rows = self._q(f"papers_{schema.paper_shard(corpusid):02d}",
                       "SELECT js FROM papers WHERE corpusid = ?", (corpusid,))
        if not rows:
            return None
        return jsonio.loads(zlib.decompress(rows[0][0]))

    def get_papers(self, corpusids: list[int]) -> dict[int, dict]:
        """Bulk fetch, grouped per shard, shards queried concurrently.

        The fan-out matters on a FUSE-mounted volume: a 1000-row page
        touches ~all 64 shards, and cold page reads are network round
        trips — parallel shards turn 64 sequential seek-chains into ~8
        concurrent ones.
        """
        by_shard: dict[int, list[int]] = {}
        for cid in corpusids:
            by_shard.setdefault(schema.paper_shard(cid), []).append(cid)
        out: dict[int, dict] = {}

        def fetch(shard: int, cids: list[int]) -> list[tuple[int, bytes]]:
            rows: list[tuple[int, bytes]] = []
            for i in range(0, len(cids), 400):
                chunk = cids[i: i + 400]
                marks = ",".join("?" * len(chunk))
                rows.extend(self._q(
                    f"papers_{shard:02d}",
                    f"SELECT corpusid, js FROM papers WHERE corpusid IN ({marks})", chunk,
                ))
            return rows

        if len(by_shard) <= 2:
            results = [fetch(s, c) for s, c in by_shard.items()]
        else:
            results = list(self._pool.map(lambda sc: fetch(*sc), by_shard.items()))
        for rows in results:
            for cid, js in rows:
                out[cid] = jsonio.loads(zlib.decompress(js))
        return out

    def has_paper(self, corpusid: int) -> bool:
        return bool(self._q(f"papers_{schema.paper_shard(corpusid):02d}",
                            "SELECT 1 FROM papers WHERE corpusid = ?", (corpusid,)))

    # ---- adjacency -------------------------------------------------------

    @staticmethod
    def _dedupe(arr: np.ndarray) -> np.ndarray:
        """Order-preserving first-occurrence dedupe on ``other``.

        The S2AG citations dump materializes most edges in two dump
        files (two shard families), so blobs built before the reducer
        grew its own dedupe carry ~2x rows. Cheap belt-and-suspenders
        at read time keeps every store version correct.
        """
        if len(arr) < 2:
            return arr
        _, first_idx = np.unique(arr["other"], return_index=True)
        if len(first_idx) == len(arr):
            return arr
        mask = np.zeros(len(arr), dtype=bool)
        mask[first_idx] = True
        return arr[mask]

    def adjacency(self, table: str, corpusid: int) -> np.ndarray:
        """Deduped (other, flags) array for refs/citers; empty when none."""
        cache_key = (table, corpusid)
        with self._adj_lock:
            hit = self._adj_cache.get(cache_key)
            if hit is not None:
                self._adj_cache.move_to_end(cache_key)
                return hit
        rows = self._q(f"graph_{schema.paper_shard(corpusid):02d}",
                       f"SELECT adj FROM {table} WHERE corpusid = ?", (corpusid,))
        arr = np.empty(0, dtype=ADJ_DTYPE)
        if rows:
            arr = self._dedupe(np.frombuffer(rows[0][0], dtype=ADJ_DTYPE))
        with self._adj_lock:
            self._adj_cache[cache_key] = arr
            while len(self._adj_cache) > _ADJ_CACHE_MAX:
                self._adj_cache.popitem(last=False)
        return arr

    # ---- authors ---------------------------------------------------------

    def get_author(self, authorid: int) -> dict | None:
        rows = self._q(f"authors_{schema.author_shard(authorid):02d}",
                       "SELECT js FROM authors WHERE authorid = ?", (authorid,))
        return jsonio.loads(zlib.decompress(rows[0][0])) if rows else None

    def author_paper_ids(self, authorid: int) -> np.ndarray:
        rows = self._q(f"authors_{schema.author_shard(authorid):02d}",
                       "SELECT corpusids FROM author_papers WHERE authorid = ?",
                       (authorid,))
        if not rows:
            return np.empty(0, dtype="<i8")
        arr = np.frombuffer(rows[0][0], dtype="<i8")
        _, first_idx = np.unique(arr, return_index=True)
        return arr[np.sort(first_idx)] if len(first_idx) < len(arr) else arr


class CurrentStore:
    """Resolves ``<base>/CURRENT`` (a text file holding a version dir name)
    to a :class:`MirrorStore`, re-checking every ``ttl`` seconds so a
    weekly rebuild can flip versions under a running server."""

    def __init__(self, base: str | Path, ttl: float = 30.0,
                 reload_hook=None) -> None:
        self.base = Path(base)
        self.ttl = ttl
        self.reload_hook = reload_hook  # e.g. modal Volume.reload for warm containers
        self._store: MirrorStore | None = None
        self._version = ""
        self._checked = 0.0
        self._lock = threading.Lock()

    def get(self) -> MirrorStore | None:
        now = time.time()
        if self._store is not None and now - self._checked < self.ttl:
            return self._store
        with self._lock:
            if self._store is not None and now - self._checked < self.ttl:
                return self._store
            self._checked = now
            if self.reload_hook is not None:
                try:
                    self.reload_hook()
                except Exception:
                    pass
            pointer = self.base / "CURRENT"
            if not pointer.exists():
                return self._store
            version = pointer.read_text().strip()
            if version and version != self._version:
                root = self.base / version
                if root.is_dir():
                    old = self._store
                    self._store = MirrorStore(root)
                    self._version = version
                    if old is not None:
                        old.close()
            return self._store

    @property
    def version(self) -> str:
        return self._version
