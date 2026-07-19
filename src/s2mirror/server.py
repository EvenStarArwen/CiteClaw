"""FastAPI app that speaks the S2 graph-API dialect CiteClaw uses.

Local shard store first; anything it can't answer (unknown ids, papers
newer than the loaded release, ``embedding.*`` fields, search routes)
is proxied to the real api.semanticscholar.org with a server-side key
at a polite ~1 rps, with an in-RAM memo so repeated misses are free.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from s2mirror import schema
from s2mirror.store import CurrentStore, MirrorStore

_EDGE_FIELDS = ("contexts", "intents", "isInfluential")
_UPSTREAM_MEMO_MAX = 50_000
_MAX_BATCH = 1000


class Upstream:
    """Throttled pass-through to the real S2 API (server-side key)."""

    def __init__(self, api_key: str, base: str = "https://api.semanticscholar.org",
                 min_interval: float = 1.05) -> None:
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0
        self._http = httpx.Client(
            timeout=60, headers={"x-api-key": api_key, "Accept": "application/json"},
        )

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def request(self, method: str, path: str, params: dict | None = None,
                json_body: Any = None) -> tuple[int, Any]:
        self._throttle()
        r = self._http.request(method, f"{self.base}{path}", params=params, json=json_body)
        try:
            payload = r.json()
        except Exception:
            payload = {"error": r.text[:500]}
        return r.status_code, payload


class _Memo:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            while len(self._d) > self.cap:
                self._d.popitem(last=False)


def _split_edge_fields(fields: str | None, inner_key: str) -> tuple[set[str], dict]:
    """Split a references/citations ``fields`` param into
    (edge-level fields, inner-paper parsed wants)."""
    edge: set[str] = set()
    inner_parts: list[str] = []
    for part in (fields or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part in _EDGE_FIELDS:
            edge.add(part)
        elif part == inner_key:
            inner_parts.append("paperId,title")
        elif part.startswith(inner_key + "."):
            inner_parts.append(part[len(inner_key) + 1:])
    wants = schema.parse_fields(",".join(inner_parts) or None, default="paperId")
    return edge, wants


def create_app(
    store_source: CurrentStore | MirrorStore,
    api_keys: set[str] | frozenset[str] = frozenset(),
    upstream: Upstream | None = None,
) -> FastAPI:
    app = FastAPI(title="s2mirror", docs_url=None, redoc_url=None, openapi_url=None)
    memo = _Memo(_UPSTREAM_MEMO_MAX)
    stats = {"local": 0, "upstream": 0, "miss": 0, "started": time.time()}
    stats_lock = threading.Lock()

    def _bump(key: str, n: int = 1) -> None:
        with stats_lock:
            stats[key] = stats.get(key, 0) + n

    def _store() -> MirrorStore:
        s = store_source.get() if isinstance(store_source, CurrentStore) else store_source
        if s is None:
            raise HTTPException(503, detail="store not loaded")
        return s

    def _auth(request: Request) -> None:
        if not api_keys:
            return
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
        if supplied not in api_keys:
            raise HTTPException(401, detail="invalid or missing api key")

    def _proxy(method: str, path: str, params: dict | None = None,
               json_body: Any = None, memo_key: Any = None) -> JSONResponse:
        if upstream is None:
            _bump("miss")
            return JSONResponse({"error": "not found in mirror (no upstream configured)"},
                                status_code=404)
        if memo_key is not None:
            hit = memo.get(memo_key)
            if hit is not None:
                return JSONResponse(hit[1], status_code=hit[0])
        _bump("upstream")
        status, payload = upstream.request(method, path, params, json_body)
        if memo_key is not None and status in (200, 404):
            memo.put(memo_key, (status, payload))
        return JSONResponse(payload, status_code=status)

    # ---- meta ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        version = ""
        ok = True
        try:
            s = _store()
            version = str(s.meta.get("release", "")) or getattr(store_source, "version", "")
        except HTTPException:
            ok = False
        return {"ok": ok, "release": version}

    @app.get("/stats", dependencies=[Depends(_auth)])
    def get_stats() -> dict:
        with stats_lock:
            return dict(stats)

    # ---- search passthrough (kept upstream by design) --------------------

    @app.get("/graph/v1/paper/search/{rest:path}", dependencies=[Depends(_auth)])
    def search(rest: str, request: Request):
        return _proxy("GET", f"/graph/v1/paper/search/{rest}",
                      params=dict(request.query_params))

    # ---- paper batch -----------------------------------------------------

    @app.post("/graph/v1/paper/batch", dependencies=[Depends(_auth)])
    def paper_batch(body: dict, fields: str | None = None):
        ids = body.get("ids") or []
        if not isinstance(ids, list) or len(ids) > _MAX_BATCH:
            raise HTTPException(400, detail="ids must be a list of at most 1000")
        wants = schema.parse_fields(fields)
        if any(f.startswith("embedding") for f in wants):
            return _proxy("POST", "/graph/v1/paper/batch", params={"fields": fields},
                          json_body={"ids": ids})
        store = _store()
        resolved = [(raw, store.resolve(str(raw))) for raw in ids]
        found = store.get_papers([cid for _, cid in resolved if cid is not None])
        out: list[dict | None] = []
        missing: list[tuple[int, str]] = []
        for pos, (raw, cid) in enumerate(resolved):
            rec = found.get(cid) if cid is not None else None
            if rec is not None:
                out.append(schema.project(rec, wants))
            else:
                out.append(None)
                missing.append((pos, str(raw)))
        _bump("local", len(ids) - len(missing))
        if missing and upstream is not None and len(missing) <= 100:
            _bump("upstream")
            status, payload = upstream.request(
                "POST", "/graph/v1/paper/batch", params={"fields": fields},
                json_body={"ids": [raw for _, raw in missing]},
            )
            if status == 200 and isinstance(payload, list):
                for (pos, _raw), entry in zip(missing, payload):
                    out[pos] = entry
        elif missing:
            _bump("miss", len(missing))
        return JSONResponse(out)

    # ---- references / citations -----------------------------------------

    def _edges(paper_id: str, table: str, inner_key: str, request: Request,
               fields: str | None, offset: int, limit: int):
        store = _store()
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        cid = store.resolve(paper_id)
        if cid is None or not store.has_paper(cid):
            path = request.url.path
            return _proxy("GET", path, params=dict(request.query_params),
                          memo_key=("edges", path, fields, offset, limit))
        _bump("local")
        edge_fields, wants = _split_edge_fields(fields, inner_key)
        arr = store.adjacency(table, cid)
        total = len(arr)
        page = arr[offset: offset + limit]
        recs = store.get_papers([int(r["other"]) for r in page])
        data = []
        for row in page:
            rec = recs.get(int(row["other"]))
            inner = schema.project(rec, wants) if rec is not None else {"paperId": None}
            entry: dict[str, Any] = {}
            if edge_fields:
                influential, intents = schema.unpack_flags(int(row["flags"]))
                if "isInfluential" in edge_fields:
                    entry["isInfluential"] = influential
                if "intents" in edge_fields:
                    entry["intents"] = intents
                if "contexts" in edge_fields:
                    entry["contexts"] = []
            entry[inner_key] = inner
            data.append(entry)
        envelope: dict[str, Any] = {"offset": offset, "data": data}
        if offset + len(page) < total:
            envelope["next"] = offset + len(page)
        return JSONResponse(envelope)

    @app.get("/graph/v1/paper/{paper_id:path}/references", dependencies=[Depends(_auth)])
    def references(paper_id: str, request: Request, fields: str | None = None,
                   offset: int = 0, limit: int = 100):
        return _edges(paper_id, "refs", "citedPaper", request, fields, offset, limit)

    @app.get("/graph/v1/paper/{paper_id:path}/citations", dependencies=[Depends(_auth)])
    def citations(paper_id: str, request: Request, fields: str | None = None,
                  offset: int = 0, limit: int = 100):
        return _edges(paper_id, "citers", "citingPaper", request, fields, offset, limit)

    # ---- single paper (declared last: greedy path param) -----------------

    @app.get("/graph/v1/paper/{paper_id:path}", dependencies=[Depends(_auth)])
    def paper(paper_id: str, request: Request, fields: str | None = None):
        wants = schema.parse_fields(fields)
        if any(f.startswith("embedding") for f in wants):
            return _proxy("GET", f"/graph/v1/paper/{paper_id}",
                          params=dict(request.query_params),
                          memo_key=("emb", paper_id, fields))
        store = _store()
        cid = store.resolve(paper_id)
        rec = store.get_paper(cid) if cid is not None else None
        if rec is None:
            return _proxy("GET", f"/graph/v1/paper/{paper_id}",
                          params=dict(request.query_params),
                          memo_key=("paper", paper_id, fields))
        _bump("local")
        return JSONResponse(schema.project(rec, wants))

    # ---- authors ---------------------------------------------------------

    @app.post("/graph/v1/author/batch", dependencies=[Depends(_auth)])
    def author_batch(body: dict, fields: str | None = None):
        ids = body.get("ids") or []
        if not isinstance(ids, list) or len(ids) > _MAX_BATCH:
            raise HTTPException(400, detail="ids must be a list of at most 1000")
        store = _store()
        wants = schema.parse_fields(fields, default="name")
        wants.setdefault("authorId", None)
        wants.pop("paperId", None)
        out: list[dict | None] = []
        for raw in ids:
            try:
                rec = store.get_author(int(str(raw)))
            except ValueError:
                rec = None
            out.append(schema.project(rec, wants) if rec else None)
        _bump("local", len(ids))
        return JSONResponse(out)

    @app.get("/graph/v1/author/{author_id}/papers", dependencies=[Depends(_auth)])
    def author_papers(author_id: str, request: Request, fields: str | None = None,
                      offset: int = 0, limit: int = 100):
        store = _store()
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        try:
            aid = int(author_id)
        except ValueError:
            aid = -1
        if aid < 0 or store.get_author(aid) is None:
            path = f"/graph/v1/author/{author_id}/papers"
            return _proxy("GET", path, params=dict(request.query_params),
                          memo_key=("apapers", author_id, fields, offset, limit))
        _bump("local")
        wants = schema.parse_fields(fields)
        cids = store.author_paper_ids(aid)
        total = len(cids)
        page = [int(c) for c in cids[offset: offset + limit]]
        recs = store.get_papers(page)
        data = [schema.project(recs[c], wants) for c in page if c in recs]
        envelope: dict[str, Any] = {"offset": offset, "data": data}
        if offset + len(page) < total:
            envelope["next"] = offset + len(page)
        return JSONResponse(envelope)

    return app
