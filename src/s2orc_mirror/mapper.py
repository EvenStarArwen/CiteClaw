"""Map phase: one ``s2orc`` dump file -> per-shard partition files.

Each invocation handles exactly one ``.jsonl.gz`` dump file and writes
its rows, re-partitioned by the *target* shard key, under::

    <out_root>/
      text/s<NN>/f<FFF>.jsonl.gz   (slim full-text records, by corpusid)
      ids/s<NN>/f<FFF>.tsv.gz      (doi/arxiv/pmid/pmcid -> corpusid)

Partitions stream into per-shard gzip buffers (compressed as they fill,
so peak RAM ~= the file's compressed size) and are written once at the
end; a retried mapper simply overwrites its own output files — no
partial state is possible as long as reducers run only after all mappers.
"""

from __future__ import annotations

import gzip
import io
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from s2orc_mirror import jsonio, schema


# ---- input streaming -------------------------------------------------------

def fetch_to_disk(url: str, dest: Path, attempts: int = 4) -> Path:
    """Download a dump file to local disk with whole-file retries."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
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


# ---- partition sink --------------------------------------------------------

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


# ---- the (single) dataset mapper -------------------------------------------

def _map_s2orc(rows, out_root: Path, file_index: int) -> dict:
    text = _GzSink(schema.TEXT_SHARDS)
    ids = _GzSink(schema.ID_SHARDS)
    n = kept = 0
    for row in rows:
        n += 1
        slim = schema.slim_record(row)
        if slim is None:
            continue
        kept += 1
        cid = slim["corpusid"]
        text.write(schema.text_shard(cid), jsonio.dumps(slim))
        for key in schema.idmap_keys(slim["externalids"]):
            ids.write(schema.id_shard(key), f"{key}\t{cid}".encode())
    text.flush(out_root, "text", file_index, "jsonl.gz")
    ids.flush(out_root, "ids", file_index, "tsv.gz")
    return {"rows": n, "kept": kept}


_MAPPERS: dict[str, Callable] = {"s2orc": _map_s2orc}
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
