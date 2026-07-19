"""Modal deployment for the public CiteClaw web app.

Single always-scalable-to-zero CPU container serving the multi-tenant
FastAPI app in ``web/public/backend/server.py`` over HTTPS + WebSockets.
Durable state (invites, sessions, run artifacts, cache snapshots) lives on
the ``citeclaw-web-data`` Volume; the hot SQLite cache stays on container
disk and is snapshotted back by ``cache_sync``.

One-time setup:

    modal secret create citeclaw-web CITECLAW_SESSION_SECRET=$(openssl rand -hex 32)

Deploy / operate:

    modal deploy modal_webapp.py
    modal run modal_webapp.py::mint --n 3 --note "first beta users"
    modal run modal_webapp.py::codes
    modal run modal_webapp.py::revoke --prefix ab12cd

The app answers at  https://<workspace>--citeclaw-web-serve.modal.run

Knobs (env at deploy time): CITECLAW_WEB_APP_NAME, CITECLAW_WEB_SCALEDOWN
(seconds, default 60 — the user chose an aggressive sleep; an open tab's
15s /me poll keeps the container warm, closing the tab lets it die).
"""

from __future__ import annotations

import os

import modal

APP_NAME = os.environ.get("CITECLAW_WEB_APP_NAME", "citeclaw-web")
SCALEDOWN = int(os.environ.get("CITECLAW_WEB_SCALEDOWN", "60"))

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)

_IGNORE = ["**/__pycache__/**", "**/.DS_Store", "**/*.db", "**/*.log",
           "**/runs/**", "**/scratch/**", "**/.pytest_cache/**"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # citeclaw core (mirrors pyproject) + web serving + key encryption
        "httpx>=0.27,<1", "pydantic>=2.5,<3", "pydantic-settings>=2.1,<3",
        "python-igraph>=0.11,<1", "tenacity>=8.2,<10", "openai>=1.10,<2",
        "google-genai>=1.0,<2", "pyyaml>=6.0,<7", "rich>=13.0,<15",
        "fastapi>=0.111,<1", "cryptography>=42,<46",
    )
    .env({
        "PYTHONPATH": "/root/app:/root/app/src",
        "CITECLAW_PUBLIC_DATA": "/data",
        # beta conveniences: deterministic stub model for key-free demos,
        # OpenAlex polite-pool contact for the abstract fallback
        "CITECLAW_WEBUI_ALLOW_STUB": "1",
        "OPENALEX_EMAIL": "m.huang.gla@outlook.com",
    })
    .add_local_dir("src/citeclaw", "/root/app/src/citeclaw", ignore=_IGNORE)
    .add_local_dir("web/live", "/root/app/web/live", ignore=_IGNORE)
    .add_local_dir("web/public", "/root/app/web/public", ignore=_IGNORE)
)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("citeclaw-web")],
    cpu=2.0,
    memory=2048,
    scaledown_window=SCALEDOWN,
    # The in-process run manager + shared SQLite cache assume one replica.
    max_containers=1,
    # WebSocket streams count as held inputs; give long runs headroom.
    timeout=2 * 3600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    # `web` is a PEP 420 namespace package — /root/app on PYTHONPATH makes
    # web.live / web.public importable without an __init__.py.
    from web.public.backend import auth, cache_sync, server

    cache_sync.VOLUME_COMMIT = vol.commit
    auth.RELOAD_HOOK = vol.reload
    return server.app


# ------------------------------------------------------------- admin CLI

@app.function(image=image, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("citeclaw-web")])
def _admin(op: str, n: int = 3, note: str = "", prefix: str = ""):
    from web.public.backend import auth
    if op == "mint":
        out = auth.mint_codes(n, note)
    elif op == "codes":
        out = auth.list_codes()
    elif op == "revoke":
        out = auth.disable_code(prefix)
    else:
        raise ValueError(f"unknown op {op!r}")
    vol.commit()
    return out


@app.local_entrypoint()
def mint(n: int = 3, note: str = ""):
    """Mint invite codes (plaintext shown once — save them)."""
    for code in _admin.remote("mint", n=n, note=note):
        print(code)


@app.local_entrypoint()
def codes():
    """List codes (hash prefix, uses, disabled)."""
    for row in _admin.remote("codes"):
        flag = " DISABLED" if row.get("disabled") else ""
        print(f"{row['hash']}  uses={row.get('uses', 0)}  note={row.get('note', '')!r}{flag}")


@app.local_entrypoint()
def revoke(prefix: str):
    """Disable codes whose hash starts with the given prefix."""
    print(f"disabled {_admin.remote('revoke', prefix=prefix)} code(s)")
