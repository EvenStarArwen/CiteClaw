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
        self._conns: dict[str, sqlite3.Connection] = {}
        self._conn_lock = threading.Lock()
        self._adj_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._adj_lock = threading.Lock()

    # ---- connections -----------------------------------------------------

    def _conn(self, name: str) -> sqlite3.Connection | None:
        conn = self._conns.get(name)
        if conn is not None:
            return conn
        with self._conn_lock:
            conn = self._conns.get(name)
            if conn is not None:
                return conn
            path = self.root / f"{name}.db"
            if not path.exists():
                return None
            conn = sqlite3.connect(
                f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False,
            )
            self._conns[name] = conn
            return conn

    def close(self) -> None:
        with self._conn_lock:
            for c in self._conns.values():
                try:
                    c.close()
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
        conn = self._conn(f"ids_{schema.id_shard(str(val)):02d}")
        if conn is None:
            return None
        row = conn.execute(
            "SELECT corpusid FROM idmap WHERE k = ?", (val,)
        ).fetchone()
        return row[0] if row else None

    # ---- papers ----------------------------------------------------------

    def get_paper(self, corpusid: int) -> dict | None:
        conn = self._conn(f"papers_{schema.paper_shard(corpusid):02d}")
        if conn is None:
            return None
        row = conn.execute(
            "SELECT js FROM papers WHERE corpusid = ?", (corpusid,)
        ).fetchone()
        if not row:
            return None
        return jsonio.loads(zlib.decompress(row[0]))

    def get_papers(self, corpusids: list[int]) -> dict[int, dict]:
        """Bulk fetch, grouped per shard. Missing ids are simply absent."""
        by_shard: dict[int, list[int]] = {}
        for cid in corpusids:
            by_shard.setdefault(schema.paper_shard(cid), []).append(cid)
        out: dict[int, dict] = {}
        for shard, cids in by_shard.items():
            conn = self._conn(f"papers_{shard:02d}")
            if conn is None:
                continue
            for i in range(0, len(cids), 400):
                chunk = cids[i: i + 400]
                marks = ",".join("?" * len(chunk))
                for cid, js in conn.execute(
                    f"SELECT corpusid, js FROM papers WHERE corpusid IN ({marks})", chunk,
                ):
                    out[cid] = jsonio.loads(zlib.decompress(js))
        return out

    def has_paper(self, corpusid: int) -> bool:
        conn = self._conn(f"papers_{schema.paper_shard(corpusid):02d}")
        if conn is None:
            return False
        return conn.execute(
            "SELECT 1 FROM papers WHERE corpusid = ?", (corpusid,)
        ).fetchone() is not None

    # ---- adjacency -------------------------------------------------------

    def adjacency(self, table: str, corpusid: int) -> np.ndarray:
        """Full (other, flags) array for refs/citers; empty when none."""
        cache_key = (table, corpusid)
        with self._adj_lock:
            hit = self._adj_cache.get(cache_key)
            if hit is not None:
                self._adj_cache.move_to_end(cache_key)
                return hit
        conn = self._conn(f"graph_{schema.paper_shard(corpusid):02d}")
        arr = np.empty(0, dtype=ADJ_DTYPE)
        if conn is not None:
            row = conn.execute(
                f"SELECT adj FROM {table} WHERE corpusid = ?", (corpusid,)
            ).fetchone()
            if row:
                arr = np.frombuffer(row[0], dtype=ADJ_DTYPE)
        with self._adj_lock:
            self._adj_cache[cache_key] = arr
            while len(self._adj_cache) > _ADJ_CACHE_MAX:
                self._adj_cache.popitem(last=False)
        return arr

    # ---- authors ---------------------------------------------------------

    def get_author(self, authorid: int) -> dict | None:
        conn = self._conn(f"authors_{schema.author_shard(authorid):02d}")
        if conn is None:
            return None
        row = conn.execute(
            "SELECT js FROM authors WHERE authorid = ?", (authorid,)
        ).fetchone()
        return jsonio.loads(zlib.decompress(row[0])) if row else None

    def author_paper_ids(self, authorid: int) -> np.ndarray:
        conn = self._conn(f"authors_{schema.author_shard(authorid):02d}")
        if conn is None:
            return np.empty(0, dtype="<i8")
        row = conn.execute(
            "SELECT corpusids FROM author_papers WHERE authorid = ?", (authorid,)
        ).fetchone()
        return np.frombuffer(row[0], dtype="<i8") if row else np.empty(0, dtype="<i8")


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
