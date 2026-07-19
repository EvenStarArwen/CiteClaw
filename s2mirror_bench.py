"""Benchmark + golden-equivalence harness for the self-hosted S2 mirror.

Measures the operations CiteClaw actually performs, with fixed anchors so
runs are comparable across server refinements:

    point_warm      hot single-paper GETs (edge-hydration field set)
    point_scatter   2000 distinct ids, 64-way concurrent  -> QPS
    batch500        POST /paper/batch x500 ids, PAPER_FIELDS
    refs_page       full-field references page
    cits_page       1000-row citations pages (rotating offsets)
    mega_seq        complete citer walk of a 184k-citer paper (client pattern)
    mega_par        same pages, 16-way concurrent (server throughput ceiling)
    author_batch    POST /author/batch x100
    mixed_wave      20x (citations page + batch 200 + references) @ 8 conc

Modes:
    modal run s2mirror_bench.py                     # in-region (public-app view)
    python s2mirror_bench.py --local                # WAN view from this machine
    python s2mirror_bench.py --local --capture golden.json
    python s2mirror_bench.py --local --verify golden.json   # equivalence gate

Env: S2_MIRROR_URL / S2_MIRROR_KEY (the Modal function reads the
citeclaw-s2-mirror secret's MIRROR_KEYS instead).

Golden mode records parsed JSON bodies for a fixed request set and
compares by deep equality — any refinement must keep them identical.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import time

import modal

APP_URL_DEFAULT = "https://cola-lab--citeclaw-s2-mirror-serve.modal.run"

PAPER_FIELDS = ("paperId,title,abstract,venue,year,publicationDate,citationCount,"
                "referenceCount,influentialCitationCount,openAccessPdf,externalIds,"
                "authors.authorId,authors.name,fieldsOfStudy,s2FieldsOfStudy,publicationTypes")
EDGE_IDS = "paperId,year,publicationDate,citationCount"
ATTENTION = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
PAPERQA = "ARXIV:2312.07559"

app = modal.App("citeclaw-s2-bench")
bench_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx>=0.27")


# --------------------------------------------------------------------------
# harness core (runs identically locally and inside the Modal function)
# --------------------------------------------------------------------------

def _client(base: str, key: str):
    import httpx
    return httpx.Client(
        base_url=base, timeout=180,
        headers={"x-api-key": key},
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=128),
    )

def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]

def _summ(lat_ms: list[float], wall_s: float, n: int, unit_count: int | None = None):
    out = {
        "n": n, "wall_s": round(wall_s, 2),
        "p50_ms": round(_pct(lat_ms, 0.50), 1),
        "p90_ms": round(_pct(lat_ms, 0.90), 1),
        "p99_ms": round(_pct(lat_ms, 0.99), 1),
        "rps": round(n / wall_s, 1),
    }
    if unit_count:
        out["items_per_s"] = round(unit_count / wall_s, 0)
    return out

def _run_conc(fn, jobs, conc):
    lat = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        for ms in pool.map(fn, jobs):
            lat.append(ms)
    return lat, time.perf_counter() - t0


class Bench:
    def __init__(self, base: str, key: str):
        self.http = _client(base, key)
        self.server_ms: dict[str, list[float]] = {}

    def _track(self, op: str, resp) -> None:
        st = resp.headers.get("x-server-ms")
        if st:
            self.server_ms.setdefault(op, []).append(float(st))

    def get(self, op, path, params):
        t0 = time.perf_counter()
        r = self.http.get(path, params=params)
        r.raise_for_status()
        self._track(op, r)
        return (time.perf_counter() - t0) * 1000

    def post(self, op, path, params, body):
        t0 = time.perf_counter()
        r = self.http.post(path, params=params, json=body)
        r.raise_for_status()
        self._track(op, r)
        return (time.perf_counter() - t0) * 1000

    # ---- fixed anchors ----------------------------------------------------

    def setup(self):
        ids = []
        for off in (0, 1000):
            d = self.http.get(f"/graph/v1/paper/{ATTENTION}/citations",
                              params={"fields": "citingPaper.paperId",
                                      "limit": 1000, "offset": off}).json()
            ids += [e["citingPaper"]["paperId"] for e in d["data"]
                    if e["citingPaper"]["paperId"]]
        self.ids = ids                       # ~2000 deterministic sha ids
        meta = self.http.get(f"/graph/v1/paper/{ATTENTION}",
                             params={"fields": "citationCount,authors.authorId"}).json()
        self.mega_total = meta["citationCount"]
        d = self.http.post("/graph/v1/paper/batch",
                           params={"fields": "authors.authorId"},
                           json={"ids": self.ids[:200]}).json()
        aids = []
        for entry in d:
            for a in (entry or {}).get("authors") or []:
                if a.get("authorId"):
                    aids.append(a["authorId"])
        self.author_ids = aids[:100]
        # warm-up: touch every op path once so a fresh container's caches load
        self.get("warm", f"/graph/v1/paper/{self.ids[0]}", {"fields": EDGE_IDS})
        self.post("warm", "/graph/v1/paper/batch", {"fields": PAPER_FIELDS},
                  {"ids": self.ids[:500]})
        self.get("warm", f"/graph/v1/paper/{PAPERQA}/references",
                 {"fields": f"citedPaper.paperId,citedPaper.title,contexts,intents,isInfluential",
                  "limit": 1000})
        self.get("warm", f"/graph/v1/paper/{ATTENTION}/citations",
                 {"fields": f"citingPaper.{EDGE_IDS},contexts,intents,isInfluential",
                  "limit": 1000})

    # ---- ops --------------------------------------------------------------

    def point_warm(self):
        hot = self.ids[:50]
        jobs = [hot[i % 50] for i in range(300)]
        lat, wall = _run_conc(
            lambda pid: self.get("point_warm", f"/graph/v1/paper/{pid}",
                                 {"fields": EDGE_IDS}), jobs, 1)
        return _summ(lat, wall, len(jobs))

    def point_scatter(self):
        jobs = self.ids[:2000]
        lat, wall = _run_conc(
            lambda pid: self.get("point_scatter", f"/graph/v1/paper/{pid}",
                                 {"fields": EDGE_IDS}), jobs, 64)
        return _summ(lat, wall, len(jobs))

    def batch500(self):
        chunks = [self.ids[i:i + 500] for i in range(0, 2000, 500)] * 3
        lat, wall = _run_conc(
            lambda c: self.post("batch500", "/graph/v1/paper/batch",
                                {"fields": PAPER_FIELDS}, {"ids": c}), chunks, 4)
        return _summ(lat, wall, len(chunks), unit_count=sum(len(c) for c in chunks))

    def refs_page(self):
        jobs = list(range(60))
        lat, wall = _run_conc(
            lambda _i: self.get("refs_page", f"/graph/v1/paper/{PAPERQA}/references",
                                {"fields": "citedPaper.paperId,citedPaper.title,"
                                           "citedPaper.year,citedPaper.venue,"
                                           "contexts,intents,isInfluential",
                                 "limit": 1000}), jobs, 8)
        return _summ(lat, wall, len(jobs))

    def cits_page(self):
        offs = [(i % 20) * 1000 for i in range(80)]
        lat, wall = _run_conc(
            lambda off: self.get("cits_page", f"/graph/v1/paper/{ATTENTION}/citations",
                                 {"fields": f"citingPaper.{EDGE_IDS},contexts,intents,isInfluential",
                                  "limit": 1000, "offset": off}), offs, 8)
        return _summ(lat, wall, len(offs), unit_count=len(offs) * 1000)

    def _mega_pages(self):
        # first 60k citers — same per-page work as the full walk, bounded time
        return list(range(0, min(self.mega_total, 60_000), 1000))

    def mega_seq(self):
        pages = self._mega_pages()
        lat, wall = _run_conc(
            lambda off: self.get("mega_seq", f"/graph/v1/paper/{ATTENTION}/citations",
                                 {"fields": f"citingPaper.{EDGE_IDS},contexts,intents,isInfluential",
                                  "limit": 1000, "offset": off}), pages, 1)
        return _summ(lat, wall, len(pages), unit_count=self.mega_total)

    def mega_par(self):
        pages = self._mega_pages()
        lat, wall = _run_conc(
            lambda off: self.get("mega_par", f"/graph/v1/paper/{ATTENTION}/citations",
                                 {"fields": f"citingPaper.{EDGE_IDS},contexts,intents,isInfluential",
                                  "limit": 1000, "offset": off}), pages, 16)
        return _summ(lat, wall, len(pages), unit_count=self.mega_total)

    def author_batch(self):
        jobs = list(range(40))
        lat, wall = _run_conc(
            lambda _i: self.post("author_batch", "/graph/v1/author/batch",
                                 {"fields": "name,citationCount,hIndex,paperCount,affiliations"},
                                 {"ids": self.author_ids}), jobs, 8)
        return _summ(lat, wall, len(jobs))

    def mixed_wave(self):
        anchors = self.ids[:20]
        def wave(pid):
            t0 = time.perf_counter()
            self.get("mixed", f"/graph/v1/paper/{pid}/citations",
                     {"fields": f"citingPaper.{EDGE_IDS}", "limit": 1000})
            self.post("mixed", "/graph/v1/paper/batch", {"fields": PAPER_FIELDS},
                      {"ids": self.ids[200:400]})
            self.get("mixed", f"/graph/v1/paper/{pid}/references",
                     {"fields": "citedPaper.paperId,citedPaper.title", "limit": 1000})
            return (time.perf_counter() - t0) * 1000
        lat, wall = _run_conc(wave, anchors, 8)
        return _summ(lat, wall, len(anchors))

    OPS = ("point_warm", "point_scatter", "batch500", "refs_page", "cits_page",
           "mega_seq", "mega_par", "author_batch", "mixed_wave")

    def run(self, ops=None):
        self.setup()
        results = {}
        for op in (ops or self.OPS):
            results[op] = getattr(self, op)()
            srv = self.server_ms.get(op)
            if srv:
                results[op]["srv_p50_ms"] = round(_pct(srv, 0.5), 1)
            print(f"{op:14s} {json.dumps(results[op])}", flush=True)
        return results


# --------------------------------------------------------------------------
# golden equivalence
# --------------------------------------------------------------------------

def golden_requests(b: "Bench"):
    """Fixed request set covering every locally-served route + field shape."""
    b.setup()
    ids = b.ids
    return [
        ("paper_sha_edge", "GET", f"/graph/v1/paper/{ids[3]}", {"fields": EDGE_IDS}, None),
        ("paper_sha_full", "GET", f"/graph/v1/paper/{ids[7]}", {"fields": PAPER_FIELDS}, None),
        ("paper_arxiv", "GET", f"/graph/v1/paper/{PAPERQA}", {"fields": PAPER_FIELDS}, None),
        ("paper_default_fields", "GET", f"/graph/v1/paper/{ids[11]}", {}, None),
        ("batch_full", "POST", "/graph/v1/paper/batch", {"fields": PAPER_FIELDS},
         {"ids": ids[:50] + ["DOI:10.404/definitely-missing"]}),
        ("batch_edge", "POST", "/graph/v1/paper/batch", {"fields": EDGE_IDS},
         {"ids": ids[50:100]}),
        ("refs_full", "GET", f"/graph/v1/paper/{PAPERQA}/references",
         {"fields": "citedPaper.paperId,citedPaper.title,citedPaper.year,"
                    "citedPaper.venue,contexts,intents,isInfluential", "limit": 1000}, None),
        ("cits_p0", "GET", f"/graph/v1/paper/{ATTENTION}/citations",
         {"fields": f"citingPaper.{EDGE_IDS},contexts,intents,isInfluential",
          "limit": 100}, None),
        ("cits_deep", "GET", f"/graph/v1/paper/{ATTENTION}/citations",
         {"fields": "citingPaper.paperId", "limit": 100, "offset": 5000}, None),
        ("author_batch", "POST", "/graph/v1/author/batch",
         {"fields": "name,citationCount,hIndex,paperCount,affiliations"},
         {"ids": b.author_ids[:40] + ["999999999"]}),
        ("author_papers", "GET", f"/graph/v1/author/{b.author_ids[0]}/papers",
         {"fields": "paperId,title,year,venue,citationCount", "limit": 100}, None),
    ]

def _fetch_golden(b: Bench):
    out = {}
    for name, method, path, params, body in golden_requests(b):
        if method == "GET":
            r = b.http.get(path, params=params)
        else:
            r = b.http.post(path, params=params, json=body)
        r.raise_for_status()
        out[name] = r.json()
    return out

def _diff(a, b, path="$"):
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(f"{path}.{k}: missing in golden")
            elif k not in b: out.append(f"{path}.{k}: missing in current")
            else: out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out[:5]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------

def _main(base, key, args):
    b = Bench(base, key)
    if args.capture:
        json.dump(_fetch_golden(b), open(args.capture, "w"), sort_keys=True)
        print(f"golden captured: {args.capture}")
        return {}
    if args.verify:
        golden = json.load(open(args.verify))
        current = _fetch_golden(b)
        bad = 0
        for name, want in golden.items():
            diffs = _diff(want, current.get(name))
            if diffs:
                bad += 1
                print(f"MISMATCH {name}:")
                for d in diffs[:6]:
                    print("   ", d)
            else:
                print(f"ok {name}")
        print("EQUIVALENT" if not bad else f"{bad} MISMATCHED RESPONSES")
        return {"equivalent": not bad}
    return b.run(args.ops.split(",") if args.ops else None)


@app.function(image=bench_image, cpu=4, memory=4096, timeout=1800,
              secrets=[modal.Secret.from_name("citeclaw-s2-mirror")])
def bench_remote(url: str, ops: str = "") -> dict:
    key = os.environ["MIRROR_KEYS"].split(",")[0].strip()
    ns = argparse.Namespace(capture=None, verify=None, ops=ops or None)
    return _main(url, key, ns)


@app.local_entrypoint()
def remote(url: str = APP_URL_DEFAULT, ops: str = ""):
    results = bench_remote.remote(url, ops)
    print("\nRESULTS_JSON " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--url", default=os.environ.get("S2_MIRROR_URL", APP_URL_DEFAULT))
    ap.add_argument("--key", default=os.environ.get("S2_MIRROR_KEY", ""))
    ap.add_argument("--ops", default="")
    ap.add_argument("--capture")
    ap.add_argument("--verify")
    a = ap.parse_args()
    out = _main(a.url, a.key, a)
    if out:
        print("\nRESULTS_JSON " + json.dumps(out, sort_keys=True))
