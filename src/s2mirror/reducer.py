"""Reduce phase: partition files -> one read-only SQLite shard DB.

Every reducer builds its DB on *local* scratch disk (SQLite must never
write through a FUSE volume) and then copies the finished file to its
final location in one shot. Rerunning a reducer overwrites its shard.

Shard families::

    graph_<NN>.db    refs(corpusid PK, adj BLOB)  + citers(...)   [PAPER_SHARDS]
    papers_<NN>.db   papers(corpusid PK, sha, js BLOB zlib)       [PAPER_SHARDS]
    ids_<NN>.db      idmap(k TEXT PK, corpusid) WITHOUT ROWID     [ID_SHARDS]
    authors_<NN>.db  authors(authorid PK, js BLOB zlib)
                     + author_papers(authorid PK, corpusids BLOB) [AUTHOR_SHARDS]

Adjacency blobs are packed ``(<int64 other><uint8 flags>)*`` rows sorted
newest-first (descending corpusid ~ descending recency); author paper
lists are packed descending ``int64`` corpusids.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np

from s2mirror import jsonio, schema

EDGE_DTYPE = np.dtype([("key", "<i8"), ("other", "<i8"), ("flags", "u1")])
ADJ_DTYPE = np.dtype([("other", "<i8"), ("flags", "u1")])
PAIR_DTYPE = np.dtype([("a", "<i8"), ("b", "<i8")])


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


def _load_edges(parts_root: Path, kind: str, shard: int) -> np.ndarray:
    chunks = [f.read_bytes() for f in _part_files(parts_root, kind, shard, "bin")]
    if not chunks:
        return np.empty(0, dtype=EDGE_DTYPE)
    return np.frombuffer(b"".join(chunks), dtype=EDGE_DTYPE)


def _grouped_adjacency(arr: np.ndarray):
    """Yield (key, packed_adj_bytes) for a raw edge array."""
    if len(arr) == 0:
        return
    order = np.lexsort((-arr["other"], arr["key"]))
    key_s = arr["key"][order]
    adj = np.empty(len(arr), dtype=ADJ_DTYPE)
    adj["other"] = arr["other"][order]
    adj["flags"] = arr["flags"][order]
    keys, starts = np.unique(key_s, return_index=True)
    bounds = np.append(starts, len(key_s))
    for i, k in enumerate(keys):
        yield int(k), adj[bounds[i]: bounds[i + 1]].tobytes()


def reduce_graph(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    built = scratch / f"graph_{shard:02d}.db"
    conn = _new_db(built)
    stats = {}
    for table, kind in (("refs", "edges_refs"), ("citers", "edges_cits")):
        conn.execute(f"CREATE TABLE {table} (corpusid INTEGER PRIMARY KEY, adj BLOB)")
        arr = _load_edges(parts_root, kind, shard)
        rows = ((k, b) for k, b in _grouped_adjacency(arr))
        conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", rows)
        stats[table + "_edges"] = int(len(arr))
        del arr
    size = _finish(conn, built, out_path)
    return {**stats, "bytes": size, "secs": round(time.time() - t0, 1)}


def reduce_papers(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    abstracts: dict[int, dict] = {}
    for f in _part_files(parts_root, "abstracts", shard, "jsonl.gz"):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if line.strip():
                    row = jsonio.loads(line)
                    abstracts[row["corpusid"]] = row

    built = scratch / f"papers_{shard:02d}.db"
    conn = _new_db(built)
    conn.execute("CREATE TABLE papers (corpusid INTEGER PRIMARY KEY, sha TEXT, js BLOB)")
    n = with_abs = 0
    batch: list[tuple[int, str, bytes]] = []
    for f in _part_files(parts_root, "papers", shard, "jsonl.gz"):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = jsonio.loads(line)
                corpusid = rec["corpusId"]
                extra = abstracts.get(corpusid)
                if extra is not None:
                    with_abs += 1
                    rec["abstract"] = extra["abstract"]
                    if extra.get("oa_url"):
                        rec["openAccessPdf"] = {
                            "url": extra["oa_url"],
                            "status": extra.get("oa_status"),
                        }
                batch.append((corpusid, rec["paperId"], zlib.compress(jsonio.dumps(rec), 6)))
                n += 1
                if len(batch) >= 20000:
                    conn.executemany("INSERT OR REPLACE INTO papers VALUES (?, ?, ?)", batch)
                    batch.clear()
    if batch:
        conn.executemany("INSERT OR REPLACE INTO papers VALUES (?, ?, ?)", batch)
    size = _finish(conn, built, out_path)
    return {"papers": n, "with_abstract": with_abs, "bytes": size,
            "secs": round(time.time() - t0, 1)}


def reduce_ids(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    rows: list[tuple[str, int]] = []
    for kind in ("ids", "xids"):
        for f in _part_files(parts_root, kind, shard, "tsv.gz"):
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


def reduce_authors(parts_root: Path, shard: int, out_path: Path, scratch: Path) -> dict:
    t0 = time.time()
    built = scratch / f"authors_{shard:02d}.db"
    conn = _new_db(built)
    conn.execute("CREATE TABLE authors (authorid INTEGER PRIMARY KEY, js BLOB)")
    conn.execute(
        "CREATE TABLE author_papers (authorid INTEGER PRIMARY KEY, corpusids BLOB)"
    )
    n_auth = 0
    batch: list[tuple[int, bytes]] = []
    for f in _part_files(parts_root, "authors", shard, "jsonl.gz"):
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = jsonio.loads(line)
                batch.append((int(rec["authorId"]), zlib.compress(jsonio.dumps(rec), 6)))
                n_auth += 1
                if len(batch) >= 20000:
                    conn.executemany("INSERT OR REPLACE INTO authors VALUES (?, ?)", batch)
                    batch.clear()
    if batch:
        conn.executemany("INSERT OR REPLACE INTO authors VALUES (?, ?)", batch)

    chunks = [f.read_bytes() for f in _part_files(parts_root, "apapers", shard, "bin")]
    n_pairs = 0
    if chunks:
        arr = np.frombuffer(b"".join(chunks), dtype=PAIR_DTYPE)
        n_pairs = len(arr)
        order = np.lexsort((-arr["b"], arr["a"]))
        a_s, b_s = arr["a"][order], arr["b"][order]
        keys, starts = np.unique(a_s, return_index=True)
        bounds = np.append(starts, len(a_s))
        rows = (
            (int(k), b_s[bounds[i]: bounds[i + 1]].astype("<i8").tobytes())
            for i, k in enumerate(keys)
        )
        conn.executemany("INSERT OR REPLACE INTO author_papers VALUES (?, ?)", rows)
    size = _finish(conn, built, out_path)
    return {"authors": n_auth, "pairs": n_pairs, "bytes": size,
            "secs": round(time.time() - t0, 1)}


REDUCERS = {
    "graph": (reduce_graph, schema.PAPER_SHARDS),
    "papers": (reduce_papers, schema.PAPER_SHARDS),
    "ids": (reduce_ids, schema.ID_SHARDS),
    "authors": (reduce_authors, schema.AUTHOR_SHARDS),
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
