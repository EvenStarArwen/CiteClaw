"""FastAPI application for the local CiteClaw Web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.catalog import router as catalog_router
from api.configs import router as configs_router
from api.runs import router as runs_router
from runtime import PROJECT_ROOT


def create_app(*, frontend_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="CiteClaw local Web UI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:9999",
            "http://localhost:9999",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(catalog_router)
    app.include_router(configs_router)
    app.include_router(runs_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": "local"}

    dist = frontend_dist or (PROJECT_ROOT / "web" / "frontend" / "dist")
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


app = create_app()
