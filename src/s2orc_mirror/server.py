"""FastAPI app serving S2ORC full text keyed by corpusid.

Surface (all under ``/s2orc/v1``)::

    GET  /paper/{id}?include=text,annotations
    POST /paper/batch?include=text          body {"ids": [...]}

``id`` is ``CorpusId:<n>`` / ``DOI:<doi>`` / ``ARXIV:<id>`` / ``PMID:<n>``
/ ``PMCID:<n>`` (bare shas are not indexed). A paper absent from S2ORC is
a 404 (single) or ``null`` (batch, aligned to the input) — there is no
upstream to fall back to. ``include`` defaults to ``text``; ``meta`` alone
(``?include=meta``) is a cheap membership + license check that skips the
body blob entirely.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response

try:  # orjson: faster render, present in the serving image
    from fastapi.responses import ORJSONResponse as JSONResponse
    import orjson  # noqa: F401 — ORJSONResponse needs it at render time
except ImportError:  # pragma: no cover - local test envs without orjson
    from fastapi.responses import JSONResponse

from s2orc_mirror import jsonio, schema
from s2orc_mirror.store import CurrentStore, MirrorStore

_MAX_BATCH = 1000


def _wants(include: str | None) -> tuple[bool, bool]:
    """(want_text, want_annotations) from an ``include`` param."""
    if include is None:
        return True, False
    parts = {p.strip().lower() for p in include.split(",") if p.strip()}
    return ("text" in parts), ("annotations" in parts or "annos" in parts)


def create_app(
    store_source: CurrentStore | MirrorStore,
    api_keys: set[str] | frozenset[str] = frozenset(),
) -> FastAPI:
    def _warm_store() -> None:
        """Touch every shard's btree root so a fresh container's first real
        requests don't eat the cold-FUSE tax."""
        try:
            s = store_source.get() if isinstance(store_source, CurrentStore) else store_source
            if s is None:
                return
            for fam, n, table in (("fulltext", schema.TEXT_SHARDS, "fulltext"),
                                  ("ids", schema.ID_SHARDS, "idmap")):
                for i in range(n):
                    s._q(f"{fam}_{i:02d}", f"SELECT 1 FROM {table} LIMIT 1")
        except Exception:
            pass

    @asynccontextmanager
    async def _lifespan(_app):
        from anyio import to_thread
        to_thread.current_default_thread_limiter().total_tokens = 96
        threading.Thread(target=_warm_store, daemon=True).start()
        yield

    app = FastAPI(title="s2orc-mirror", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=_lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=1)

    stats = {"local": 0, "miss": 0, "started": time.time()}
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

    # ---- meta ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        version, ok = "", True
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

    # ---- batch (declared before the greedy single-paper path) ------------

    @app.post("/s2orc/v1/paper/batch", dependencies=[Depends(_auth)])
    def paper_batch(body: dict, include: str | None = None):
        ids = body.get("ids") or []
        if not isinstance(ids, list) or len(ids) > _MAX_BATCH:
            raise HTTPException(400, detail="ids must be a list of at most 1000")
        want_text, want_annos = _wants(include)
        store = _store()
        cid_map = store.resolve_many([str(raw) for raw in ids])
        want_cids = sorted({c for c in cid_map.values() if c is not None})
        recs = store.get_many_fulltext(want_cids, want_text=want_text, want_annos=want_annos)
        parts: list[bytes] = []
        hits = 0
        for raw in ids:
            cid = cid_map.get(str(raw))
            rec = recs.get(cid) if cid is not None else None
            if rec is None:
                parts.append(b"null")
            else:
                hits += 1
                parts.append(jsonio.dumps(rec))
        _bump("local", hits)
        _bump("miss", len(ids) - hits)
        return Response(b"[" + b",".join(parts) + b"]", media_type="application/json")

    # ---- single paper (greedy path param, declared last) -----------------

    @app.get("/s2orc/v1/paper/{paper_id:path}", dependencies=[Depends(_auth)])
    def paper(paper_id: str, include: str | None = None):
        want_text, want_annos = _wants(include)
        store = _store()
        cid = store.resolve(paper_id)
        rec = (store.get_fulltext(cid, want_text=want_text, want_annos=want_annos)
               if cid is not None else None)
        if rec is None:
            _bump("miss")
            raise HTTPException(404, detail="paper not found in s2orc")
        _bump("local")
        return JSONResponse(rec)

    return app
