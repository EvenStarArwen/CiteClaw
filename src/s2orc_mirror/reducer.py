"""Reduce phase: partition files -> one read-only SQLite shard DB.

Every reducer builds its DB on *local* scratch disk (SQLite must never
write through a FUSE volume) and copies the finished file into place in
one shot. Rerunning a reducer overwrites its shard.

Shard families::

    fulltext_<NN>.db  fulltext(corpusid PK, license, status, oaurl,
                               externalids TEXT, body BLOB, annos BLOB)  [TEXT_SHARDS]
    ids_<NN>.db       idmap(k TEXT PK, corpusid) WITHOUT ROWID           [ID_SHARDS]

``body`` = zlib(text); ``annos`` = zlib(json of the annotation spans),
NULL when a record carries none. Bodies are large (tens of KB to MB), so
the insert batch flushes on an accumulated-bytes budget, not a row count.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
import time
import zlib
from pathlib import Path

from s2orc_mirror import jsonio, schema

_BATCH_BYTES = 64 << 20  # flush the insert batch once it holds ~64 MB of blobs


def _new_db(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def _finish(conn: sqlite3.Connection, built: Path, out_path: Path) -> int:
    conn.commit()
    conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    shutil.copyfile(built, tmp)
    tmp.replace(out_path)
    built.unlink(missing_ok=True)
    return out_path.stat().st_size


def _part_files(parts_root: Path, kind: str, shard: int, ext: str) -> list[Path]:
    d = parts_root / kind / f"s{shard:02d}"
    return sorted(d.glob(f"*.{ext}")) if d.exists() else []


def reduce_text(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    built = scratch / f"fulltext_{shard:02d}.db"
    conn = _new_db(built)
    conn.execute(
        "CREATE TABLE fulltext (corpusid INTEGER PRIMARY KEY, license TEXT, "
        "status TEXT, oaurl TEXT, externalids TEXT, body BLOB, annos BLOB)"
    )
    n = 0
    batch: list[tuple] = []
    batch_bytes = 0

    def _flush() -> None:
        nonlocal batch, batch_bytes
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO fulltext VALUES (?, ?, ?, ?, ?, ?, ?)", batch
            )
            batch = []
            batch_bytes = 0

    for f in _part_files(parts_root, "text", shard, "jsonl.gz"):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = jsonio.loads(line)
                body = zlib.compress(r["text"].encode("utf-8"), 6)
                ann = r.get("annotations") or {}
                annos = zlib.compress(jsonio.dumps(ann), 6) if ann else None
                batch.append((
                    r["corpusid"], r.get("license"), r.get("status"), r.get("oaurl"),
                    jsonio.dumps(r.get("externalids") or {}).decode("utf-8"),
                    body, annos,
                ))
                batch_bytes += len(body) + (len(annos) if annos else 0)
                n += 1
                if batch_bytes >= _BATCH_BYTES:
                    _flush()
    _flush()
    size = _finish(conn, built, out_path)
    return {"fulltext": n, "bytes": size, "secs": round(time.time() - t0, 1)}


def reduce_ids(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    rows: list[tuple[str, int]] = []
    for f in _part_files(parts_root, "ids", shard, "tsv.gz"):
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                k, sep, cid = line.rstrip("\n").partition("\t")
                if sep and cid:
                    try:
                        rows.append((k, int(cid)))
                    except ValueError:
                        continue
    rows.sort()
    built = scratch / f"ids_{shard:02d}.db"
    conn = _new_db(built)
    conn.execute(
        "CREATE TABLE idmap (k TEXT PRIMARY KEY, corpusid INTEGER) WITHOUT ROWID"
    )
    for i in range(0, len(rows), 50000):
        conn.executemany("INSERT OR REPLACE INTO idmap VALUES (?, ?)", rows[i: i + 50000])
    size = _finish(conn, built, out_path)
    return {"keys": len(rows), "bytes": size, "secs": round(time.time() - t0, 1)}


# Registry key == the DB-file prefix the store queries (fulltext_NN.db /
# ids_NN.db). The full-text reducer reads the mapper's "text/" partition
# dir internally, but its shard family is named "fulltext" to match
# store.get_fulltext()'s ``fulltext_{shard}`` lookup.
REDUCERS = {
    "fulltext": (reduce_text, schema.TEXT_SHARDS),
    "ids": (reduce_ids, schema.ID_SHARDS),
}


def reduce_shard(kind: str, shard: int, parts_root: str | Path,
                 store_root: str | Path, scratch: str | Path = "/tmp") -> dict:
    fn, n_shards = REDUCERS[kind]
    if not 0 <= shard < n_shards:
        raise ValueError(f"{kind} shard {shard} out of range 0..{n_shards - 1}")
    out_path = Path(store_root) / f"{kind}_{shard:02d}.db"
    stats = fn(Path(parts_root), shard, out_path, Path(scratch))
    stats.update({"kind": kind, "shard": shard})
    return stats
