"""Map phase: one S2AG dataset file -> per-shard partition files.

Each mapper invocation handles exactly one ``.jsonl.gz`` dump file and
writes its rows, re-partitioned by the *target* shard key, under::

    <out_root>/
      edges_refs/s<NN>/f<FFF>.bin    (citations file: partitioned by citing)
      edges_cits/s<NN>/f<FFF>.bin    (citations file: partitioned by cited)
      papers/s<NN>/f<FFF>.jsonl.gz   (papers file: rendered API records)
      xids/s<NN>/f<FFF>.tsv.gz       (papers file: DOI/ArXiv/... -> corpusid)
      apapers/s<NN>/f<FFF>.bin       (papers file: authorid,corpusid pairs)
      abstracts/s<NN>/f<FFF>.jsonl.gz
      ids/s<NN>/f<FFF>.tsv.gz        (paper-ids file: sha -> corpusid)
      authors/s<NN>/f<FFF>.jsonl.gz

Partitions are buffered fully in RAM (a dump file is ~1 GB compressed;
its partitions total the same) and written once at the end, so a
retried mapper simply overwrites its own output files — no partial
state is possible as long as reducers run only after all mappers.
"""

from __future__ import annotations

import gzip
import io
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from s2mirror import jsonio, render, schema

_EDGE = struct.Struct(schema.EDGE_ROW)
_PAIR = struct.Struct(schema.PAIR_ROW)


# ---- input streaming -------------------------------------------------------

def fetch_to_disk(url: str, dest: Path, attempts: int = 4) -> Path:
    """Download a dump file to local disk with whole-file retries."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return dest
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))
    return dest


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield jsonio.loads(line)


# ---- partition sinks -------------------------------------------------------

class _BinSink:
    def __init__(self, n: int) -> None:
        self.parts = [bytearray() for _ in range(n)]

    def flush(self, root: Path, kind: str, file_index: int) -> None:
        for shard, buf in enumerate(self.parts):
            if not buf:
                continue
            d = root / kind / f"s{shard:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"f{file_index:03d}.bin").write_bytes(bytes(buf))


class _GzSink:
    def __init__(self, n: int) -> None:
        self.bufs = [io.BytesIO() for _ in range(n)]
        self.gz = [gzip.GzipFile(fileobj=b, mode="wb", compresslevel=2) for b in self.bufs]

    def write(self, shard: int, line: bytes) -> None:
        self.gz[shard].write(line)
        self.gz[shard].write(b"\n")

    def flush(self, root: Path, kind: str, file_index: int, ext: str) -> None:
        for shard, (g, b) in enumerate(zip(self.gz, self.bufs)):
            g.close()
            data = b.getvalue()
            if len(data) <= 26:  # empty gzip member
                continue
            d = root / kind / f"s{shard:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"f{file_index:03d}.{ext}").write_bytes(data)


# ---- per-dataset mappers ---------------------------------------------------

def _map_citations(rows, out_root: Path, file_index: int) -> dict:
    refs = _BinSink(schema.PAPER_SHARDS)
    cits = _BinSink(schema.PAPER_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        citing = row.get("citingcorpusid")
        cited = row.get("citedcorpusid")
        if not isinstance(citing, int) or not isinstance(cited, int):
            continue
        kept += 1
        flags = schema.pack_flags(bool(row.get("isinfluential")), row.get("intents"))
        refs.parts[schema.paper_shard(citing)] += _EDGE.pack(citing, cited, flags)
        cits.parts[schema.paper_shard(cited)] += _EDGE.pack(cited, citing, flags)
    refs.flush(out_root, "edges_refs", file_index)
    cits.flush(out_root, "edges_cits", file_index)
    return {"rows": n, "edges": kept}


def _map_papers(rows, out_root: Path, file_index: int) -> dict:
    papers = _GzSink(schema.PAPER_SHARDS)
    xids = _GzSink(schema.ID_SHARDS)
    apapers = _BinSink(schema.AUTHOR_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        rendered = render.render_paper(row)
        if rendered is None:
            continue
        kept += 1
        corpusid, _sha, rec = rendered
        papers.write(schema.paper_shard(corpusid), jsonio.dumps(rec))
        for dkey, prefix in schema.XID_PREFIX.items():
            val = (row.get("externalids") or {}).get(dkey)
            if val is not None:
                key = f"{prefix}:{str(val).lower()}"
                xids.write(schema.id_shard(key), f"{key}\t{corpusid}".encode())
        for a in rec["authors"]:
            aid = a.get("authorId")
            if aid:
                try:
                    aid_int = int(aid)
                except ValueError:
                    continue
                apapers.parts[schema.author_shard(aid_int)] += _PAIR.pack(aid_int, corpusid)
    papers.flush(out_root, "papers", file_index, "jsonl.gz")
    xids.flush(out_root, "xids", file_index, "tsv.gz")
    apapers.flush(out_root, "apapers", file_index)
    return {"rows": n, "papers": kept}


def _map_abstracts(rows, out_root: Path, file_index: int) -> dict:
    sink = _GzSink(schema.PAPER_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        corpusid = row.get("corpusid")
        abstract = row.get("abstract")
        if not isinstance(corpusid, int) or not abstract:
            continue
        kept += 1
        oa = row.get("openaccessinfo") or {}
        slim: dict[str, Any] = {"corpusid": corpusid, "abstract": abstract}
        # opportunistic OA-pdf recovery — the papers dump has no pdf url,
        # but openaccessinfo sometimes does.
        oa_url = oa.get("url") if isinstance(oa, dict) else None
        if oa_url:
            slim["oa_url"] = oa_url
            if isinstance(oa.get("status"), str):
                slim["oa_status"] = oa["status"]
        sink.write(schema.paper_shard(corpusid), jsonio.dumps(slim))
    sink.flush(out_root, "abstracts", file_index, "jsonl.gz")
    return {"rows": n, "abstracts": kept}


def _map_paper_ids(rows, out_root: Path, file_index: int) -> dict:
    sink = _GzSink(schema.ID_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        sha = row.get("sha")
        corpusid = row.get("corpusid")
        if not sha or not isinstance(corpusid, int):
            continue
        kept += 1
        key = str(sha).lower()
        sink.write(schema.id_shard(key), f"{key}\t{corpusid}".encode())
    sink.flush(out_root, "ids", file_index, "tsv.gz")
    return {"rows": n, "shas": kept}


def _map_authors(rows, out_root: Path, file_index: int) -> dict:
    sink = _GzSink(schema.AUTHOR_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        rendered = render.render_author(row)
        if rendered is None:
            continue
        kept += 1
        aid_int, rec = rendered
        sink.write(schema.author_shard(aid_int), jsonio.dumps(rec))
    sink.flush(out_root, "authors", file_index, "jsonl.gz")
    return {"rows": n, "authors": kept}


_MAPPERS: dict[str, Callable] = {
    "citations": _map_citations,
    "papers": _map_papers,
    "abstracts": _map_abstracts,
    "paper-ids": _map_paper_ids,
    "authors": _map_authors,
}

DATASETS = tuple(_MAPPERS)


def map_dataset_file(
    dataset: str, src: str | Path, out_root: str | Path, file_index: int,
    scratch: str | Path = "/tmp",
) -> dict:
    """Run the map phase for one dump file (local path or presigned URL)."""
    out_root = Path(out_root)
    src_str = str(src)
    if src_str.startswith("http"):
        local = Path(scratch) / f"{dataset}_{file_index:03d}.jsonl.gz"
        fetch_to_disk(src_str, local)
    else:
        local = Path(src)
    try:
        stats = _MAPPERS[dataset](iter_rows(local), out_root, file_index)
    finally:
        if src_str.startswith("http"):
            local.unlink(missing_ok=True)
    stats["dataset"] = dataset
    stats["file_index"] = file_index
    return stats
