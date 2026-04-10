"""CiteClaw web server — serves the FastAPI backend + built frontend.

Usage::

    python -m citeclaw web --port 9999

The server mounts the React production bundle (``web/frontend/dist/``)
as static files at ``/`` and the API routers at ``/api/*``.  The
backend modules live in ``web/backend/`` and are imported by
temporarily extending ``sys.path``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("citeclaw.web")

# Resolve paths relative to the *project root* (two levels up from
# src/citeclaw/).
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "web" / "backend"
_FRONTEND_DIST = _PROJECT_ROOT / "web" / "frontend" / "dist"


def _build_app():
    """Construct the FastAPI application with API routers + static files."""
    # Make the ``web/backend/`` directory importable so ``from api.configs
    # import router`` works inside the backend modules.
    backend_str = str(_BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)

    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    # Import routers from the backend package (now on sys.path).
    from api.configs import router as configs_router  # type: ignore[import-not-found]
    from api.papers import router as papers_router  # type: ignore[import-not-found]
    from api.runs import router as runs_router  # type: ignore[import-not-found]

    app = FastAPI(title="CiteClaw", version="0.1.0")

    app.include_router(configs_router)
    app.include_router(papers_router)
    app.include_router(runs_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Serve the React production bundle.  Must be mounted LAST so the
    # ``/api/*`` routes take precedence.
    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIST), html=True),
            name="frontend",
        )
    else:
        log.warning(
            "Frontend build not found at %s — run 'pnpm build' in web/frontend/",
            _FRONTEND_DIST,
        )

    return app


def serve(*, host: str = "0.0.0.0", port: int = 9999) -> None:
    """Start the uvicorn server."""
    import uvicorn

    app = _build_app()
    log.info("Starting CiteClaw web UI on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
