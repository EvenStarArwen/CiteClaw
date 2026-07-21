"""Read-only access layer over the built shard DBs.

One ``sqlite3`` connection per shard file, opened lazily with
``mode=ro&immutable=1`` (the store is never written after a build;
immutable mode skips all locking, which also makes cross-thread sharing
safe — Python's sqlite3 serializes calls per connection under a lock).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2orc_mirror import jsonio, schema


def _decode_row(cols: list[str], row: tuple) -> dict:
    """Assemble a public record dict from a SELECT over ``cols``."""
    d = dict(zip(cols, row))
    out: dict = {
        "license": d.get("license"),
        "status": d.get("status"),
        "openAccessUrl": d.get("oaurl"),
    }
    ext = d.get("externalids")
    if ext:
        try:
            out["externalIds"] = jsonio.loads(ext)
        except Exception:
            out["externalIds"] = {}
    if "body" in d:
        out["text"] = zlib.decompress(d["body"]).decode("utf-8") if d["body"] else ""
    if "annos" in d:
        out["annotations"] = jsonio.loads(zlib.decompress(d["annos"])) if d["annos"] else {}
    return out


class MirrorStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta: dict = {}
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
        self._conns: dict[str, tuple[sqlite3.Connection | None, threading.Lock | None]] = {}
        self._conn_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="s2orcfan")

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

    def resolve_many(self, raw_ids: list[str]) -> dict[str, int | None]:
        """Bulk id resolution: one IN query per ids shard. Unparseable -> None."""
        out: dict[str, int | None] = {}
        by_shard: dict[int, list[str]] = {}
        key_for: dict[str, list[str]] = {}
        for raw in raw_ids:
            if raw in out or raw in key_for:
                continue
            parsed = schema.parse_paper_id(raw)
            if parsed is None:
                out[raw] = None
            elif parsed[0] == "corpus":
                out[raw] = int(parsed[1])
            else:
                key = str(parsed[1])
                key_for.setdefault(key, []).append(raw)
                by_shard.setdefault(schema.id_shard(key), []).append(key)

        def lookup(shard: int, keys: list[str]) -> list[tuple[str, int]]:
            rows: list[tuple[str, int]] = []
            for i in range(0, len(keys), 400):
                chunk = keys[i: i + 400]
                marks = ",".join("?" * len(chunk))
                rows.extend(self._q(
                    f"ids_{shard:02d}",
                    f"SELECT k, corpusid FROM idmap WHERE k IN ({marks})", chunk,
                ))
            return rows

        if by_shard:
            if len(by_shard) <= 2:
                results = [lookup(s, k) for s, k in by_shard.items()]
            else:
                results = list(self._pool.map(lambda sk: lookup(*sk), by_shard.items()))
            found = {k: cid for rows in results for k, cid in rows}
            for key, raws in key_for.items():
                cid = found.get(key)
                for raw in raws:
                    out[raw] = cid
        return out

    # ---- full text -------------------------------------------------------

    def _cols(self, want_text: bool, want_annos: bool) -> list[str]:
        cols = ["license", "status", "oaurl", "externalids"]
        if want_text:
            cols.append("body")
        if want_annos:
            cols.append("annos")
        return cols

    def has(self, corpusid: int) -> bool:
        return bool(self._q(f"fulltext_{schema.text_shard(corpusid):02d}",
                            "SELECT 1 FROM fulltext WHERE corpusid = ?", (corpusid,)))

    def get_fulltext(self, corpusid: int, *, want_text: bool = True,
                     want_annos: bool = False) -> dict | None:
        cols = self._cols(want_text, want_annos)
        rows = self._q(
            f"fulltext_{schema.text_shard(corpusid):02d}",
            f"SELECT {', '.join(cols)} FROM fulltext WHERE corpusid = ?", (corpusid,),
        )
        if not rows:
            return None
        rec = _decode_row(cols, rows[0])
        rec["corpusid"] = corpusid
        return rec

    def get_many_fulltext(self, corpusids: list[int], *, want_text: bool = True,
                          want_annos: bool = False) -> dict[int, dict]:
        cols = self._cols(want_text, want_annos)
        col_sql = ", ".join(["corpusid", *cols])
        by_shard: dict[int, list[int]] = {}
        for cid in corpusids:
            by_shard.setdefault(schema.text_shard(cid), []).append(cid)

        def fetch(shard: int, cids: list[int]) -> list[tuple[int, dict]]:
            rows: list[tuple] = []
            for i in range(0, len(cids), 400):
                chunk = cids[i: i + 400]
                marks = ",".join("?" * len(chunk))
                rows.extend(self._q(
                    f"fulltext_{shard:02d}",
                    f"SELECT {col_sql} FROM fulltext WHERE corpusid IN ({marks})", chunk,
                ))
            out = []
            for row in rows:
                rec = _decode_row(cols, row[1:])
                rec["corpusid"] = row[0]
                out.append((row[0], rec))
            return out

        results = ([fetch(s, c) for s, c in by_shard.items()]
                   if len(by_shard) <= 2
                   else list(self._pool.map(lambda sc: fetch(*sc), by_shard.items())))
        out: dict[int, dict] = {}
        for rows in results:
            out.update(rows)
        return out


class CurrentStore:
    """Resolves ``<base>/CURRENT`` (a text file holding a version dir name)
    to a :class:`MirrorStore`, re-checking every ``ttl`` seconds so a
    weekly rebuild can flip versions under a running server."""

    def __init__(self, base: str | Path, ttl: float = 30.0, reload_hook=None) -> None:
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
