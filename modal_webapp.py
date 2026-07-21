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
S2_MIRROR_URL = os.environ.get(
    "CITECLAW_WEB_S2_MIRROR_URL",
    "https://cola-lab--citeclaw-s2-mirror-serve.modal.run",
)
S2ORC_MIRROR_URL = os.environ.get(
    "CITECLAW_WEB_S2ORC_MIRROR_URL",
    "https://cola-lab--citeclaw-s2orc-serve.modal.run",
)

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
        # NOTE: the deterministic stub model is deliberately NOT enabled here.
        # The stub accepts every paper (it's a test double), so a real snowball
        # run screened by it silently returns garbage that looks like results —
        # a footgun for a public research tool. Production requires a real model
        # + the tenant's own API key. (OpenAlex polite-pool contact for the
        # abstract fallback.)
        "OPENALEX_EMAIL": "m.huang.gla@outlook.com",
    })
    .add_local_dir("src/citeclaw", "/root/app/src/citeclaw", ignore=_IGNORE)
    # citeclaw.clients.pdf imports pdfclaw.parsers at module load; pdfclaw's
    # heavy deps (pymupdf, playwright) are function-local, so source alone is
    # enough — public v1 never exercises the PDF paths anyway.
    .add_local_dir("src/pdfclaw", "/root/app/src/pdfclaw", ignore=_IGNORE)
    .add_local_dir("web/live", "/root/app/web/live", ignore=_IGNORE)
    .add_local_dir("web/public", "/root/app/web/public", ignore=_IGNORE)
)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("citeclaw-web"),
        # Self-hosted S2 graph mirror (MIRROR_KEYS): all tenants' graph
        # traffic bypasses the 1 rps S2 cap; search still uses each
        # tenant's own S2 key. See modal_s2_mirror.py.
        modal.Secret.from_name("citeclaw-s2-mirror"),
        # Self-hosted S2ORC full-text mirror bearer key (S2ORC_MIRROR_KEY,
        # a DISTINCT env name so it never collides with the graph mirror's
        # MIRROR_KEYS): lets accepted OA papers' full text be fetched for
        # the chat panel. See modal_s2orc_mirror.py.
        modal.Secret.from_name("citeclaw-s2orc-web"),
    ],
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
    # Ambient mirror env is intentional: _SCRUB_ENV leaves these alone, and
    # env-beats-overrides gives every tenant the mirror automatically.
    mirror_keys = os.environ.get("MIRROR_KEYS", "")
    if mirror_keys:
        os.environ.setdefault("CITECLAW_S2_MIRROR_URL", S2_MIRROR_URL)
        os.environ.setdefault("CITECLAW_S2_MIRROR_KEY",
                              mirror_keys.split(",")[0].strip())
    # Same idea for the S2ORC full-text mirror (distinct env name).
    s2orc_key = os.environ.get("S2ORC_MIRROR_KEY", "")
    if s2orc_key:
        os.environ.setdefault("CITECLAW_S2ORC_MIRROR_URL", S2ORC_MIRROR_URL)
        os.environ.setdefault("CITECLAW_S2ORC_MIRROR_KEY", s2orc_key)
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
