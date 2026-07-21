"""Public (multi-tenant) CiteClaw web server.

Same single-origin FastAPI shape as the local ``web/live`` server — same
frontend assembly, same API surface the JSX already speaks — with the
public-deployment layer on top:

  * every route behind an invite-code session cookie (``auth``)
  * per-session keys / settings / runs; nothing global, nothing in env
  * hard caps on papers, S2 rate, concurrency (``limits``)
  * artifact downloads (users can't read the server's disk)
  * shared S2/LLM cache wired through per-run symlinks (``manager``)

The live modules are imported, not copied — the local UI keeps working
unchanged, and this server reuses its translation/snapshot/search code.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from web.live.backend import models_catalog
from web.live.backend import server as live_server
from web.live.backend.abstracts import fetch_abstract
from web.live.backend.config_translate import TranslationError, build_config
from web.live.backend.explore_runs import (
    list_explore_runs,
    load_explore_collab,
    load_explore_run,
    load_explore_upload,
)
from web.live.backend.s2_seeds import S2SearchError, search_seeds
from web.live.backend.server import _iter_llm_blocks, assemble_index
from web.live.backend.snapshots import build_graph, build_metrics, build_rejected_page

from . import auth, cache_sync, limits, paths, runs_fs, tenants
from .manager import CapacityError, manager

PUBLIC_JSX = Path(__file__).resolve().parent.parent / "static" / "jsx"

# Key env vars are scrubbed at startup: with many tenants in one process,
# an ambient GEMINI_API_KEY would silently become *everyone's* key (env
# overrides beat Settings overrides in load_settings).
_SCRUB_ENV = (
    "OPENAI_API_KEY", "CITECLAW_OPENAI_API_KEY",
    "GEMINI_API_KEY", "CITECLAW_GEMINI_API_KEY",
    "S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "CITECLAW_S2_API_KEY",
)


def _css_version() -> str:
    try:
        return str(int((live_server.STATIC_DIR / "app.css").stat().st_mtime))
    except OSError:
        return "0"


def assemble_public_index() -> str:
    html = assemble_index()
    html = html.replace("window.__TWEAKS =",
                        "window.__PUBLIC__ = true;\nwindow.__TWEAKS =")
    # Cache-bust the stylesheet per deploy — without it, browsers reuse a
    # heuristically-cached app.css and users see stale styling after updates.
    html = html.replace('href="/static/app.css"',
                        f'href="/static/app.css?v={_css_version()}"')
    extras = PUBLIC_JSX / "public-extras.jsx"
    if extras.exists():
        block = "\n// === public-extras.jsx ===\n" + extras.read_text(encoding="utf-8")
        html = html.replace("\n</script>\n</body>", block + "\n</script>\n</body>")
    return html


_GATE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CiteClaw — invite required</title>
<link rel="icon" href="/static/assets/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: Inter, system-ui, sans-serif; background: #faf9f5; color: #1a1915;
         min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .card { width: min(400px, 92vw); background: #fff; border: 1px solid #e4e2dc;
          border-radius: 14px; padding: 30px 30px 26px; }
  .brand { display: flex; align-items: center; gap: 9px; margin-bottom: 18px; }
  .brand b { font-size: 15.5px; font-weight: 600; letter-spacing: -0.01em; }
  p { font-size: 13px; line-height: 1.55; color: #57544c; margin-bottom: 16px; }
  input { width: 100%; font: 500 14px "IBM Plex Mono", monospace; letter-spacing: 0.04em;
          padding: 10px 12px; border: 1px solid #d6d3cb; border-radius: 8px;
          background: #faf9f5; outline: none; text-transform: uppercase; }
  input:focus { border-color: #1a1915; }
  button { width: 100%; margin-top: 12px; padding: 10px 12px; font: 600 13.5px Inter, sans-serif;
           color: #fff; background: #1a1915; border: none; border-radius: 8px; cursor: pointer; }
  button:hover { background: #33312b; }
  .err { display: none; margin-top: 11px; font-size: 12.5px; color: #a03424; }
  .hint { margin-top: 15px; font-size: 11.5px; color: #8b877d; }
</style>
</head>
<body>
<form class="card" id="f">
  <div class="brand">
    <svg width="19" height="19" viewBox="0 0 48 46" fill="none"><path fill="#863bff" d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z"/></svg>
    <b>CiteClaw</b>
  </div>
  <p>This is a private beta. Enter your invite code to open the workspace —
     you'll add your own API keys inside.</p>
  <input id="code" placeholder="CC-XXXX-XXXX" autocomplete="off" autofocus>
  <button type="submit">Enter</button>
  <div class="err" id="err"></div>
  <div class="hint">No code? Ask the person who sent you here.</div>
</form>
<script>
document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("err");
  err.style.display = "none";
  try {
    const r = await fetch("/api/auth/join", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({code: document.getElementById("code").value}),
    });
    if (r.ok) { location.reload(); return; }
    const d = await r.json().catch(() => ({}));
    err.textContent = d.detail || "That code didn't work.";
    err.style.display = "block";
  } catch (_) {
    err.textContent = "Network error — try again.";
    err.style.display = "block";
  }
});
</script>
</body>
</html>
"""


class _Hardening(BaseHTTPMiddleware):
    """Body-size cap + response security headers."""

    async def dispatch(self, request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                if int(request.headers.get("content-length") or 0) > limits.MAX_BODY_BYTES:
                    return JSONResponse({"detail": "Request too large."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Bad content-length."}, status_code=400)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            # Always revalidate app HTML + assets (304s are cheap) so a
            # deploy is visible on the next plain reload, no hard-refresh.
            resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    for var in _SCRUB_ENV:
        os.environ.pop(var, None)
    paths.ensure_layout()
    cache_sync.start()
    manager.post_run_hook = cache_sync.sync_now
    manager.attach_loop(asyncio.get_running_loop())
    yield
    # SIGTERM (scaledown / redeploy): ask runs to stop, snapshot the cache.
    for rs in list(manager.runs.values()):
        rs.stop_requested = True
        rs.cap_event.set()
    cache_sync.stop()


app = FastAPI(title="CiteClaw", lifespan=lifespan)
app.add_middleware(_Hardening)
app.mount("/static", StaticFiles(directory=str(live_server.STATIC_DIR)), name="static")


# ------------------------------------------------------------------ pages

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> str:
    if auth.session_from_request(request) is None:
        return _GATE_HTML
    return assemble_public_index()


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


# ------------------------------------------------------------------- auth

@app.post("/api/auth/join")
async def join(request: Request) -> Response:
    if not auth.join_allowed(auth.client_ip(request)):
        raise HTTPException(status_code=429,
                            detail="Too many attempts — wait a few minutes.")
    body = await request.json()
    code_hash = auth.check_code(str(body.get("code") or ""))
    if code_hash is None:
        raise HTTPException(status_code=403, detail="Invalid or disabled invite code.")
    sid = auth.create_session(code_hash)
    # Persist eagerly: with a 1-minute scaledown a fresh session must reach
    # the volume before the container can die, or the cookie orphans.
    await asyncio.to_thread(cache_sync.sync_now)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(auth.COOKIE_NAME, auth.make_cookie(sid), **auth.cookie_kwargs())
    return resp


@app.get("/api/auth/me")
async def me(request: Request) -> dict:
    sess = auth.session_from_request(request)
    if sess is None:
        return {"authed": False}
    active = manager.active_for(sess["sid"])
    return {
        "authed": True,
        "runs_today": tenants.runs_today(sess),
        "runs_per_day": limits.RUNS_PER_DAY,
        "max_papers_ceiling": limits.MAX_PAPERS_CEILING,
        "active_run": active[0].run_id if active else None,
    }


@app.post("/api/auth/leave")
async def leave() -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE_NAME, path="/")
    return resp


def _require(request: Request) -> dict:
    sess = auth.session_from_request(request)
    if sess is None:
        raise HTTPException(status_code=401,
                            detail="Not signed in — enter an invite code.")
    auth.touch_session(sess)
    return sess


def _effective_model(stored: str | None) -> str:
    """A real catalog model to screen with. Empty OR unsupported (e.g. a
    legacy 'stub' session persisted in the volume, from when key-free demos
    were enabled) coerces to the default so screening always runs on a real
    LLM — never the accept-all stub that silently passes every paper."""
    m = (stored or "").strip()
    return m if models_catalog.is_catalog_model(m) else models_catalog.SUPPORTED_MODEL


# --------------------------------------------------------------- settings

@app.get("/api/settings")
async def get_settings(request: Request) -> dict:
    sess = _require(request)
    s = tenants.get_settings(sess)
    return {
        "keys": tenants.key_presence(sess),
        "model": _effective_model(s["model"]),
        "reasoning_effort": s["reasoning_effort"] or models_catalog.DEFAULT_EFFORT,
        "max_papers": s["max_papers"],
        "supported_model": models_catalog.SUPPORTED_MODEL,
        "max_papers_ceiling": limits.MAX_PAPERS_CEILING,
    }


@app.post("/api/settings")
async def post_settings(request: Request) -> dict:
    sess = _require(request)
    body = await request.json()
    tenants.update_keys(sess, body)
    tenants.update_settings(sess, body)
    await asyncio.to_thread(cache_sync.sync_now)  # keys must survive scaledown
    s = tenants.get_settings(sess)
    return {
        "keys": tenants.key_presence(sess),
        "model": _effective_model(s["model"]),
        "reasoning_effort": s["reasoning_effort"] or models_catalog.DEFAULT_EFFORT,
        "max_papers": s["max_papers"],
    }


@app.get("/api/models")
async def get_models(request: Request) -> list[dict]:
    _require(request)
    return models_catalog.catalog()


# ------------------------------------------------------------ seed search

@app.get("/api/seeds/search")
async def seeds_search(request: Request, q: str = "", limit: int = 100,
                       year: str = "", min_cites: int = 0, offset: int = 0) -> JSONResponse:
    sess = _require(request)
    key = tenants.get_key(sess, "s2_api_key")
    try:
        page = await asyncio.to_thread(search_seeds, q, limit, year, min_cites,
                                       offset, key)
        return JSONResponse(page)
    except S2SearchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Semantic Scholar search failed: {e}")


@app.post("/api/seeds/abstract")
async def seeds_abstract(request: Request) -> JSONResponse:
    _require(request)
    body = await request.json()
    try:
        result = await asyncio.to_thread(
            fetch_abstract, str(body.get("paper_id") or ""),
            body.get("externalIds") or {}, str(body.get("title") or ""),
            body.get("year"))
        return JSONResponse(result)
    except Exception:  # noqa: BLE001 - fallback is best-effort
        return JSONResponse({"abstract": "", "source": None})


# ---------------------------------------------------------------- explore

@app.get("/api/explore/runs")
async def explore_runs(request: Request) -> list[dict]:
    sess = _require(request)
    return await asyncio.to_thread(list_explore_runs, paths.session_runs_dir(sess["sid"]))


@app.get("/api/explore/run")
async def explore_run(request: Request, path: str) -> dict:
    sess = _require(request)
    try:
        return await asyncio.to_thread(load_explore_run, path,
                                       paths.session_runs_dir(sess["sid"]))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/explore/collab")
async def explore_collab(request: Request, path: str) -> dict:
    sess = _require(request)
    try:
        return await asyncio.to_thread(load_explore_collab, path,
                                       paths.session_runs_dir(sess["sid"]))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/explore/upload")
async def explore_upload(request: Request, name: str = "graph") -> dict:
    _require(request)
    data = await request.body()
    if len(data) > live_server._UPLOAD_MAX:
        raise HTTPException(status_code=400,
                            detail=f"'{name}' is larger than 60 MB — too big to explore in the browser.")
    try:
        return await asyncio.to_thread(load_explore_upload, name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------- runs

def _clamp_caps(cfg: dict) -> None:
    cfg["max_papers_total"] = min(int(cfg.get("max_papers_total") or 200),
                                  limits.MAX_PAPERS_CEILING)
    cfg["s2_requests_per_second"] = min(float(cfg.get("s2_requests_per_second") or 0.9), 1.0)
    cfg["llm_concurrency"] = min(int(cfg.get("llm_concurrency") or 8), 16)
    cfg["llm_batch_size"] = min(int(cfg.get("llm_batch_size") or 10), 50)
    cfg["max_llm_tokens"] = min(int(cfg.get("max_llm_tokens") or 5_000_000), 50_000_000)


@app.post("/api/run")
async def create_run(request: Request) -> dict:
    sess = _require(request)
    sid = sess["sid"]
    body = await request.json()
    stored = tenants.get_settings(sess)
    # Coerce empty / unsupported (legacy 'stub') to a real model — a snowball
    # run must never screen with the accept-all stub.
    model = _effective_model(body.get("model") or stored["model"])
    effort = (body.get("reasoning_effort") or stored["reasoning_effort"]
              or models_catalog.DEFAULT_EFFORT).strip()

    if not models_catalog.is_supported(model, effort):
        raise HTTPException(status_code=400, detail=models_catalog.support_error(model, effort))
    resolved = models_catalog.resolve_model(model)

    presence = tenants.key_presence(sess)
    need = models_catalog.required_key(resolved)
    if need and not presence[need]:
        provider = "Gemini" if need == "gemini_api_key" else "OpenAI"
        raise HTTPException(status_code=400,
                            detail=f"{provider} API key not set — the selected model needs it. "
                                   "Open Settings (gear, top-right) and add your own key; "
                                   "this server never supplies one.")

    quota_err = tenants.can_start_run(sess)
    if quota_err:
        raise HTTPException(status_code=429, detail=quota_err)

    from citeclaw.config import load_settings

    run_id = manager.new_run_id()
    data_dir = manager.prepare_run_dir(sid, run_id)
    try:
        cfg_dict = build_config(body, data_dir=data_dir, screening_model=resolved,
                                reasoning_effort=models_catalog.effort_for(resolved, effort))
    except TranslationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _clamp_caps(cfg_dict)
    # The session's own keys ride in through Settings overrides — the
    # process env carries no keys (scrubbed at startup), so these win.
    cfg_dict.update(tenants.key_overrides(sess))

    for blk in _iter_llm_blocks(cfg_dict):
        m = (blk.get("model") or "").strip()
        if not m:
            continue
        e = (blk.get("reasoning_effort") or effort).strip()
        if not models_catalog.is_supported(m, e):
            raise HTTPException(status_code=400, detail=(
                f"An LLM filter overrides its model to '{m}' (effort '{e}'). "
                + models_catalog.support_error(m, e)))
        blk["model"] = models_catalog.resolve_model(m)
        blk["reasoning_effort"] = models_catalog.effort_for(blk["model"], e)
        blk_need = models_catalog.required_key(blk["model"])
        if blk_need and not presence[blk_need]:
            provider = "Gemini" if blk_need == "gemini_api_key" else "OpenAI"
            raise HTTPException(status_code=400,
                                detail=f"An LLM filter uses a {provider} model but no "
                                       f"{provider} API key is set.")

    try:
        settings = load_settings(None, overrides=cfg_dict)
    except Exception as e:  # noqa: BLE001 - config validation error → 400
        raise HTTPException(status_code=400, detail=f"Invalid pipeline config: {e}")

    try:
        run_id, steps = manager.start_run_for(sid, settings, run_id)
    except CapacityError as e:
        raise HTTPException(status_code=429, detail=str(e))
    tenants.note_run_started(sess)
    return {"run_id": run_id, "steps": steps, "model": resolved, "reasoning_effort": effort}


@app.post("/api/run/{run_id}/stop")
async def stop_run(request: Request, run_id: str) -> dict:
    sess = _require(request)
    if manager.get_owned(run_id, sess["sid"]) is None:
        raise HTTPException(status_code=404, detail="run not found")
    manager.stop_run(run_id)
    return {"status": "stopping"}


@app.post("/api/run/{run_id}/pause")
async def pause_run(request: Request, run_id: str) -> dict:
    sess = _require(request)
    if manager.get_owned(run_id, sess["sid"]) is None:
        raise HTTPException(status_code=404, detail="run not found")
    if not manager.pause_run(run_id):
        raise HTTPException(status_code=409, detail="run not pausable")
    return {"status": "paused"}


@app.post("/api/run/{run_id}/resume")
async def resume_run(request: Request, run_id: str) -> dict:
    sess = _require(request)
    if manager.get_owned(run_id, sess["sid"]) is None:
        raise HTTPException(status_code=404, detail="run not found")
    manager.resume_run(run_id)
    return {"status": "running"}


@app.post("/api/run/{run_id}/cap")
async def cap_decision(request: Request, run_id: str, body: dict) -> dict:
    sess = _require(request)
    if manager.get_owned(run_id, sess["sid"]) is None:
        raise HTTPException(status_code=404, detail="run not found")
    action = str(body.get("action") or "stop")
    new_max = body.get("max")
    manager.cap_decide(run_id, action, int(new_max) if new_max else None)
    return {"ok": True}


@app.get("/api/run/{run_id}/status")
async def run_status(request: Request, run_id: str) -> dict:
    sess = _require(request)
    rs = manager.get_owned(run_id, sess["sid"])
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    out = {"run_id": run_id, "status": rs.status, "progress": rs.progress.snapshot(),
           "error": rs.error, "summary": rs.summary}
    if rs.ctx is not None:
        try:
            out["metrics"] = build_metrics(rs.ctx)
        except Exception:
            pass
    return out


@app.get("/api/run/{run_id}/graph")
async def run_graph(request: Request, run_id: str) -> dict:
    sess = _require(request)
    rs = manager.get_owned(run_id, sess["sid"])
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    if rs.ctx is None:
        return {"nodes": [], "edges": []}
    return build_graph(rs.ctx)


@app.get("/api/run/{run_id}/rejected")
async def run_rejected(request: Request, run_id: str, offset: int = 0,
                       limit: int = 25, sort: str = "recent", q: str = "") -> dict:
    sess = _require(request)
    rs = manager.get_owned(run_id, sess["sid"])
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    if rs.ctx is None:
        return {"total": 0, "offset": 0, "limit": limit, "sort": sort,
                "capped": False, "items": []}
    return build_rejected_page(rs.ctx, offset=offset, limit=limit, sort=sort, q=q)


_S2ORC = {"init": False, "client": None}


def _s2orc_client():
    """Lazily-built shared S2ORC full-text client (None if no mirror is set)."""
    if not _S2ORC["init"]:
        _S2ORC["init"] = True
        try:
            from citeclaw.clients.s2orc import build_s2orc_client
            from citeclaw.config import load_settings
            _S2ORC["client"] = build_s2orc_client(load_settings(None))
        except Exception:
            _S2ORC["client"] = None
    return _S2ORC["client"]


@app.get("/api/run/{run_id}/paper/{paper_id}/fulltext")
async def run_paper_fulltext(request: Request, run_id: str, paper_id: str) -> dict:
    """Parsed open-access full text for one accepted paper, when it has an
    S2ORC record — the data source for the (future) per-paper chat panel.

    ``available: false`` (with a ``reason``) rather than a 404 when the
    paper simply isn't open-access in S2ORC, so the UI can tell "no full
    text" apart from "no such paper".
    """
    sess = _require(request)
    rs = manager.get_owned(run_id, sess["sid"])
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    paper = rs.ctx.collection.get(paper_id) if rs.ctx is not None else None
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not in run")
    client = _s2orc_client()
    if client is None:
        return {"available": False, "reason": "mirror_not_configured", "paper_id": paper_id}
    res = client.fetch_full_text(paper, cache=getattr(rs.ctx, "cache", None))
    if res is None:
        return {"available": False, "reason": "not_open_access_in_s2orc", "paper_id": paper_id}
    return {
        "available": True, "paper_id": paper_id, "source": res.source,
        "chars": res.chars, "text": res.text, "license": res.license,
        "status": res.status, "openAccessUrl": res.open_access_url,
    }


@app.websocket("/api/run/{run_id}/stream")
async def run_stream(ws: WebSocket, run_id: str) -> None:
    sid = auth.parse_cookie(ws.cookies.get(auth.COOKIE_NAME, ""))
    rs = manager.get_owned(run_id, sid) if sid else None
    await ws.accept()
    if rs is None:
        await ws.send_text(json.dumps({"type": "error", "message": "run not found"}))
        await ws.close()
        return
    q, backlog = manager.subscribe(rs)
    try:
        done_seen = False
        for ev in backlog:
            await ws.send_json(ev)
            if ev.get("type") == "done":
                done_seen = True
        if not done_seen:
            while True:
                ev = await q.get()
                await ws.send_json(ev)
                if ev.get("type") == "done":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.unsubscribe(rs, q)


# -------------------------------------------------------------- downloads

@app.get("/api/session/runs")
async def session_runs(request: Request) -> list[dict]:
    sess = _require(request)
    sid = sess["sid"]
    live = {rs.run_id: ("stopping" if rs.stop_requested and rs.status == "running"
                        else rs.status)
            for rs in manager.runs.values() if rs.owner == sid}
    return await asyncio.to_thread(runs_fs.list_session_runs, sid, live)


@app.get("/api/download/{run_id}/zip")
async def download_zip(request: Request, run_id: str) -> Response:
    sess = _require(request)
    if not paths.valid_rid(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    data = await asyncio.to_thread(runs_fs.make_zip, sess["sid"], run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="no artifacts yet for this run")
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="citeclaw_{run_id}.zip"'})


@app.get("/api/download/{run_id}/{artifact}")
async def download_artifact(request: Request, run_id: str, artifact: str) -> Response:
    sess = _require(request)
    if not paths.valid_rid(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    p = runs_fs.artifact_path(sess["sid"], run_id, artifact)
    if p is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    media = {"json": "application/json", "bib": "text/plain",
             "graphml": "application/xml"}.get(p.suffix.lstrip("."), "application/octet-stream")
    return Response(p.read_bytes(), media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{p.name}"'})
