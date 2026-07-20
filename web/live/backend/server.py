"""FastAPI server: serves the v3 design + exposes the live-run API.

Single origin, single port. ``GET /`` assembles the design's HTML from the
split ``static/jsx`` files (swapping in the live-data layer), so there is
no build step. All ``/api/*`` routes drive real CiteClaw runs.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import keys_store, models_catalog
from .abstracts import fetch_abstract
from .config_translate import TranslationError, build_config
from .explore_runs import (
    list_explore_runs,
    load_explore_collab,
    load_explore_run,
    load_explore_upload,
)
from .run_manager import manager
from .s2_seeds import S2SearchError, search_seeds
from .snapshots import build_graph, build_metrics, build_rejected_page

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
JSX_DIR = STATIC_DIR / "jsx"

# Concatenation order for the single text/babel block (live-store first,
# app.jsx last so it renders after every component is defined).
ASSEMBLY_ORDER = [
    "icon.jsx",
    "data.jsx",
    "live-store.jsx",
    "top-bar.jsx",
    "bottom-bar.jsx",
    "tweaks-panel.jsx",
    "pane-splitter.jsx",
    "col-splitter.jsx",
    "build-seeds.jsx",
    "build-pipeline.jsx",
    "build-config.jsx",
    "build-step-config.jsx",
    "run-progress.jsx",
    "cite-graph.jsx",
    "run-network.jsx",
    "run-dashboard.jsx",
    "run-accepted.jsx",
    "explore-network.jsx",
    "explore-list.jsx",
    "explore-detail.jsx",
    "settings-modal.jsx",
    "app.jsx",
]

_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CiteClaw</title>
<link rel="icon" href="/static/assets/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/app.css">
<script src="https://unpkg.com/react@18.3.1/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone@7.29.0/babel.min.js"></script>
<script src="https://unpkg.com/lucide@0.447.0/dist/umd/lucide.min.js"></script>
<script type="module">
// Graph engine for the Exploration page (same stack + pins as the design's
// module_a_network_f5.html reference: graphology + ForceAtlas2 + sigma).
// Loaded fire-and-forget so a CDN hiccup never blocks the rest of the app;
// ExploreNetwork waits on window.GraphLibs / the "graphlibs" event.
(async () => {
  try {
    const [g, s, f] = await Promise.all([
      import("https://esm.sh/graphology@0.26.0"),
      import("https://esm.sh/sigma@3.0.3"),
      import("https://esm.sh/graphology-layout-forceatlas2@0.10.1/worker"),
    ]);
    window.GraphLibs = { Graph: g.default, Sigma: s.default, FA2Layout: f.default };
    try {
      // synchronous entry of the SAME package: instant one-shot layout
      // bursts (static datasets) next to the worker's live streaming mode.
      // Optional — the worker alone still lays everything out.
      const fs = await import("https://esm.sh/graphology-layout-forceatlas2@0.10.1");
      window.GraphLibs.fa2Sync = fs.default;
    } catch (e) {}
    try {
      // seed halo / node stroke / selection ring; plain circles without it.
      // ?deps pins the peer sigma to the SAME build as the main import —
      // esm.sh otherwise resolves a beta whose /rendering entry mismatches.
      const nb = await import("https://esm.sh/@sigma/node-border@3.0.0?deps=sigma@3.0.3");
      window.GraphLibs.createNodeBorderProgram = nb.createNodeBorderProgram;
    } catch (e) {}
  } catch (e) {
    window.GraphLibs = { error: String((e && e.message) || e) };
  }
  window.dispatchEvent(new Event("graphlibs"));
})();
</script>
</head>
<body>
<div id="root"></div>
<script>
// Design defaults lifted verbatim from the original v3.html __TWEAKS block
// (the "mono" palette + variants). These control the exact skin. `mode` is
// set to "build" so a live session starts in the configurator (the design's
// original "run" default shows demo data that a live app doesn't have yet).
window.__TWEAKS = {
  "palette": "mono", "theme": "light", "mode": "build", "showBottomBar": true,
  "monoRadius": "pillowy", "monoPrimary": "outline", "monoKind": "tag-mono",
  "monoLeaf": "card", "monoSeed": "inline", "monoCompLine": "gray",
  "monoRunPill": "plain", "monoTrail": "black", "monoAcc": "black",
  "monoRunDot": "green", "buildSplit": 0.42, "monoSeedFill": "orange-chip",
  "monoCanvas": "paper", "monoAccent": "cobalt", "blockStyle": "specimen",
  "colLeft": 257, "runSplit": 0.604
};
</script>
<script type="text/babel">
"""

_TAIL = """
</script>
</body>
</html>
"""


def assemble_index() -> str:
    parts = [_HEAD]
    for name in ASSEMBLY_ORDER:
        f = JSX_DIR / name
        if f.exists():
            parts.append(f"\n// === {name} ===\n")
            parts.append(f.read_text(encoding="utf-8"))
    parts.append(_TAIL)
    return "".join(parts)


# in-memory default model/effort (the Settings modal's picker); the run POST
# always carries its own, so this is just what the picker preloads.
_defaults = {"model": models_catalog.SUPPORTED_MODEL, "reasoning_effort": models_catalog.DEFAULT_EFFORT,
             "max_papers": 200}


@asynccontextmanager
async def lifespan(app: FastAPI):
    keys_store.load_into_environ()
    manager.attach_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="CiteClaw Live", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return assemble_index()


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/settings")
async def get_settings() -> dict:
    return {
        "keys": keys_store.key_presence(),
        "model": _defaults["model"],
        "reasoning_effort": _defaults["reasoning_effort"],
        "max_papers": _defaults["max_papers"],
        "supported_model": models_catalog.SUPPORTED_MODEL,
    }


@app.post("/api/settings")
async def post_settings(req: Request) -> dict:
    body = await req.json()
    keys_store.update_keys(body)
    if body.get("model"):
        _defaults["model"] = str(body["model"]).strip()
    if body.get("reasoning_effort"):
        _defaults["reasoning_effort"] = str(body["reasoning_effort"]).strip()
    if body.get("max_papers") is not None:
        try:
            _defaults["max_papers"] = max(1, min(5000, int(body["max_papers"])))
        except (TypeError, ValueError):
            pass
    return {
        "keys": keys_store.key_presence(),
        "model": _defaults["model"],
        "reasoning_effort": _defaults["reasoning_effort"],
        "max_papers": _defaults["max_papers"],
    }


@app.get("/api/models")
async def get_models() -> list[dict]:
    return models_catalog.catalog()


@app.get("/api/seeds/search")
async def seeds_search(q: str = "", limit: int = 100, year: str = "",
                       min_cites: int = 0, offset: int = 0) -> JSONResponse:
    try:
        page = await asyncio.to_thread(search_seeds, q, limit, year, min_cites, offset)
        return JSONResponse(page)
    except S2SearchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Semantic Scholar search failed: {e}")


@app.post("/api/seeds/abstract")
async def seeds_abstract(req: Request) -> JSONResponse:
    body = await req.json()
    try:
        result = await asyncio.to_thread(
            fetch_abstract,
            str(body.get("paper_id") or ""),
            body.get("externalIds") or {},
            str(body.get("title") or ""),
            body.get("year"),
        )
        return JSONResponse(result)
    except Exception:  # noqa: BLE001 - fallback is best-effort; never 500 the UI
        return JSONResponse({"abstract": "", "source": None})


@app.get("/api/explore/runs")
async def explore_runs() -> list[dict]:
    return await asyncio.to_thread(list_explore_runs)


@app.get("/api/explore/run")
async def explore_run(path: str) -> dict:
    try:
        return await asyncio.to_thread(load_explore_run, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/explore/collab")
async def explore_collab(path: str) -> dict:
    """Author co-authorship view of a run (graphml, or derived from JSON)."""
    try:
        return await asyncio.to_thread(load_explore_collab, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


_UPLOAD_MAX = 60 * 1024 * 1024  # graph XML this big would stall the browser anyway


@app.post("/api/explore/upload")
async def explore_upload(req: Request, name: str = "graph") -> dict:
    """Open a local GraphML/GEXF picked in the browser. Raw body upload (no
    multipart dependency); parsed in memory, never written to disk."""
    data = await req.body()
    if len(data) > _UPLOAD_MAX:
        raise HTTPException(status_code=400,
                            detail=f"'{name}' is larger than 60 MB — too big to explore in the browser.")
    try:
        return await asyncio.to_thread(load_explore_upload, name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _iter_llm_blocks(obj):
    """Yield every LLMFilter block dict anywhere in a translated config."""
    if isinstance(obj, dict):
        if obj.get("type") == "LLMFilter":
            yield obj
        for v in obj.values():
            yield from _iter_llm_blocks(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_llm_blocks(v)


@app.post("/api/run")
async def create_run(req: Request) -> dict:
    body = await req.json()
    model = (body.get("model") or _defaults["model"]).strip()
    effort = (body.get("reasoning_effort") or _defaults["reasoning_effort"]).strip()

    # --- model guard: any catalog model (efforts validated) ---
    if not models_catalog.is_supported(model, effort):
        raise HTTPException(status_code=400, detail=models_catalog.support_error(model, effort))
    resolved = models_catalog.resolve_model(model)

    # --- required keys ---
    presence = keys_store.key_presence()
    need = models_catalog.required_key(resolved)
    if need and not presence[need]:
        provider = "Gemini" if need == "gemini_api_key" else "OpenAI"
        raise HTTPException(status_code=400,
                            detail=f"{provider} API key not set — the selected model needs it. "
                                   "Open Settings (gear, top-right) and add it.")

    # --- translate + build Settings ---
    import uuid

    from citeclaw.config import load_settings  # local import: keeps startup light

    run_dir = Path("runs") / "webui"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(run_dir / uuid.uuid4().hex[:12])
    try:
        cfg_dict = build_config(body, data_dir=data_dir, screening_model=resolved,
                                reasoning_effort=models_catalog.effort_for(resolved, effort))
    except TranslationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- per-filter model overrides: same support + key rules as the default ---
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
                                       f"{provider} API key is set. Open Settings "
                                       "(gear, top-right) and add it.")

    try:
        settings = load_settings(None, overrides=cfg_dict)
    except Exception as e:  # noqa: BLE001 - config validation error → 400
        raise HTTPException(status_code=400, detail=f"Invalid pipeline config: {e}")

    run_id, steps = manager.start_run(settings)
    return {"run_id": run_id, "steps": steps, "model": resolved, "reasoning_effort": effort}


@app.post("/api/run/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    ok = manager.stop_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="run not found")
    return {"status": "stopping"}


@app.post("/api/run/{run_id}/cap")
async def cap_decision(run_id: str, body: dict) -> dict:
    """Answer the paper-cap modal: {action: 'stop'|'raise', max?: int}."""
    action = str(body.get("action") or "stop")
    new_max = body.get("max")
    ok = manager.cap_decide(run_id, action, int(new_max) if new_max else None)
    if not ok:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True}


@app.get("/api/run/{run_id}/status")
async def run_status(run_id: str) -> dict:
    rs = manager.get(run_id)
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
async def run_graph(run_id: str) -> dict:
    rs = manager.get(run_id)
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    if rs.ctx is None:
        return {"nodes": [], "edges": []}
    return build_graph(rs.ctx)


@app.get("/api/run/{run_id}/rejected")
async def run_rejected(run_id: str, offset: int = 0, limit: int = 25,
                       sort: str = "recent") -> dict:
    """A page of rejected papers + reasons for the Run sidebar's Rejected tab."""
    rs = manager.get(run_id)
    if not rs:
        raise HTTPException(status_code=404, detail="run not found")
    if rs.ctx is None:
        return {"total": 0, "offset": 0, "limit": limit, "sort": sort,
                "capped": False, "items": []}
    return build_rejected_page(rs.ctx, offset=offset, limit=limit, sort=sort)


@app.websocket("/api/run/{run_id}/stream")
async def run_stream(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    rs = manager.get(run_id)
    if not rs:
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
