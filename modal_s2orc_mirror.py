"""Modal app: self-hosted S2ORC full-text mirror for CiteClaw.

Serves parsed open-access full text (body + GROBID annotation spans +
license) from the Semantic Scholar ``s2orc`` bulk dataset, keyed by
corpusid. Ingest (fan-out over the dump shards) and serving share one
app + the ``citeclaw-s2orc`` volume::

    # one-time secret: bearer key(s) for the mirror + the S2 key used ONLY
    # at ingest to (re)list presigned dump URLs (never used for serving):
    modal secret create citeclaw-s2orc \
        MIRROR_KEYS=<comma-separated bearer tokens> \
        S2_API_KEY=<real S2 api key>

    modal deploy modal_s2orc_mirror.py                 # serve endpoint
    modal run modal_s2orc_mirror.py::ingest --subset 1 # smoke: 1 dump file
    modal run modal_s2orc_mirror.py::ingest            # full build
    modal run modal_s2orc_mirror.py::probe --paper-id CorpusId:89669906
    modal run modal_s2orc_mirror.py::status

Volume layout::

    /data/ingest/<release>/parts/{text,ids}/s<NN>/f<FFF>.*   (transient)
    /data/store/<release>/{fulltext_NN,ids_NN}.db + meta.json
    /data/store/CURRENT                                       (version pointer)

The endpoint serves ``/s2orc/v1/*``; point CiteClaw's ``s2orc_mirror_url``
at it with one of the MIRROR_KEYS. There is no upstream fallback — S2ORC
is bulk-only, so a miss is simply "not open-access in S2ORC".
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import modal

APP_NAME = os.environ.get("S2ORC_APP_NAME", "citeclaw-s2orc")
DATA = "/data"
DATASET = "s2orc"
DATASETS_API = "https://api.semanticscholar.org/datasets/v1"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("citeclaw-s2orc", create_if_missing=True)
secret = modal.Secret.from_name("citeclaw-s2orc")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi>=0.110", "orjson>=3.9")
    .add_local_dir("src/s2orc_mirror", remote_path="/root/s2orc_mirror")
)


# ---- datasets API helpers (need the S2 key) --------------------------------

def _datasets_get(path: str, api_key: str, attempts: int = 8) -> dict:
    for i in range(attempts):
        req = urllib.request.Request(
            f"{DATASETS_API}{path}", headers={"x-api-key": api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and i < attempts - 1:
                time.sleep(2.5 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _file_list(release: str, api_key: str) -> list[str]:
    return _datasets_get(f"/release/{release}/dataset/{DATASET}", api_key)["files"]


# ---- ingest: map -----------------------------------------------------------

@app.function(
    image=image, volumes={DATA: vol}, secrets=[secret],
    cpu=3, memory=12288, timeout=2400, retries=2, max_containers=40,
)
def map_file(file_index: int, url: str, release: str) -> dict:
    from s2orc_mirror import mapper

    out_root = f"{DATA}/ingest/{release}/parts"
    try:
        stats = mapper.map_dataset_file(DATASET, url, out_root, file_index, scratch="/tmp")
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 403):  # presigned URL expired -> refresh + retry
            raise
        fresh = _file_list(release, os.environ["S2_API_KEY"])[file_index]
        stats = mapper.map_dataset_file(DATASET, fresh, out_root, file_index, scratch="/tmp")
    vol.commit()
    return stats


# ---- ingest: reduce --------------------------------------------------------

@app.function(
    image=image, volumes={DATA: vol},
    cpu=4, memory=16384, timeout=7200, retries=2, max_containers=24,
)
def reduce_shard(kind: str, shard: int, release: str) -> dict:
    from s2orc_mirror import reducer

    stats = reducer.reduce_shard(
        kind, shard,
        parts_root=f"{DATA}/ingest/{release}/parts",
        store_root=f"{DATA}/store/{release}",
        scratch="/tmp",
    )
    vol.commit()
    return stats


@app.function(image=image, volumes={DATA: vol}, timeout=7200)
def finalize(release: str, meta: dict, drop_parts: bool = True) -> dict:
    import shutil
    from pathlib import Path

    store_dir = Path(DATA) / "store" / release
    (store_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (Path(DATA) / "store" / "CURRENT").write_text(release)
    if drop_parts:
        shutil.rmtree(Path(DATA) / "ingest" / release, ignore_errors=True)
    vol.commit()
    total = sum(p.stat().st_size for p in store_dir.glob("*.db"))
    return {"release": release, "store_bytes": total,
            "files": len(list(store_dir.glob("*.db")))}


# ---- serving ---------------------------------------------------------------

@app.function(
    image=image, volumes={DATA: vol}, secrets=[secret],
    cpu=2, memory=8192, timeout=3600, scaledown_window=300, max_containers=2,
)
@modal.concurrent(max_inputs=32, target_inputs=24)
@modal.asgi_app()
def serve():
    from s2orc_mirror.server import create_app
    from s2orc_mirror.store import CurrentStore

    last_reload = [0.0]

    def _reload():  # throttled: warm containers pick up version flips
        if time.time() - last_reload[0] > 3.0:
            last_reload[0] = time.time()
            vol.reload()

    keys = {k.strip() for k in os.environ.get("MIRROR_KEYS", "").split(",") if k.strip()}
    current = CurrentStore(f"{DATA}/store", reload_hook=_reload)
    return create_app(current, api_keys=keys)


# ---- admin / verification --------------------------------------------------

@app.function(image=image, volumes={DATA: vol}, timeout=600)
def probe(paper_id: str = "") -> dict:
    from s2orc_mirror.store import CurrentStore

    store = CurrentStore(f"{DATA}/store", reload_hook=vol.reload).get()
    if store is None:
        return {"error": "no store loaded (CURRENT missing)"}
    out: dict = {"release": store.meta.get("release", "?"), "root": str(store.root)}
    if paper_id:
        cid = store.resolve(paper_id)
        out["resolve"] = cid
        if cid is not None:
            rec = store.get_fulltext(cid, want_text=True, want_annos=True)
            if rec:
                out["license"] = rec.get("license")
                out["status"] = rec.get("status")
                out["text_chars"] = len(rec.get("text") or "")
                out["annotation_kinds"] = sorted((rec.get("annotations") or {}).keys())
                out["externalIds"] = rec.get("externalIds")
            else:
                out["found"] = False
    return out


@app.function(image=image, volumes={DATA: vol}, timeout=600)
def status() -> dict:
    from pathlib import Path

    vol.reload()
    base = Path(DATA) / "store"
    out: dict = {"current": "", "versions": {}}
    cur = base / "CURRENT"
    if cur.exists():
        out["current"] = cur.read_text().strip()
    for d in sorted(base.iterdir()) if base.exists() else []:
        if d.is_dir():
            dbs = list(d.glob("*.db"))
            out["versions"][d.name] = {
                "dbs": len(dbs),
                "gb": round(sum(p.stat().st_size for p in dbs) / 1e9, 2),
            }
    ingest = Path(DATA) / "ingest"
    if ingest.exists():
        out["ingest_dirs"] = [d.name for d in ingest.iterdir() if d.is_dir()]
    return out


# ---- orchestration ---------------------------------------------------------

@app.local_entrypoint()
def ingest(release: str = "latest", subset: int = 0, skip_map: bool = False) -> None:
    """Full build: map every ``s2orc`` dump file, reduce every shard, flip CURRENT.

    ``--subset N`` limits to the first N dump files (smoke runs; stored as a
    separate ``smoke-<release>`` version so it never overwrites prod).
    ``--skip-map`` reuses existing partitions (reduce-only rerun).
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from s2orc_mirror.reducer import REDUCERS

    api_key = os.environ.get("S2_API_KEY")
    if not api_key:
        raise SystemExit("export S2_API_KEY=<key> (needed for dataset file listings)")

    if release == "latest":
        release = _datasets_get("/release/latest", api_key)["release_id"]
        time.sleep(1.3)
    tag = release if not subset else f"smoke-{release}"
    print(f"release {release} -> store version '{tag}'")

    n_files = 0
    if not skip_map:
        files = _file_list(release, api_key)
        if subset:
            files = files[:subset]
        n_files = len(files)
        jobs = [(i, url, tag) for i, url in enumerate(files)]
        print(f"map: {len(jobs)} dump files")
        t0 = time.time()
        done = 0
        for stats in map_file.starmap(jobs, order_outputs=False):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  map {done}/{len(jobs)} ({time.time() - t0:.0f}s) last={stats}")

    rjobs = [(kind, shard, tag)
             for kind, (_fn, n) in REDUCERS.items() for shard in range(n)]
    print(f"reduce: {len(rjobs)} shards")
    t0 = time.time()
    done = 0
    totals: dict[str, int] = {}
    for stats in reduce_shard.starmap(rjobs, order_outputs=False):
        done += 1
        for k, v in stats.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        if done % 20 == 0 or done == len(rjobs):
            print(f"  reduce {done}/{len(rjobs)} ({time.time() - t0:.0f}s)")

    meta = {"release": release, "tag": tag,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "subset": subset, "map_files": n_files, "totals": totals}
    result = finalize.remote(tag, meta, drop_parts=not subset)
    print("finalize:", result)
