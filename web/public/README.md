# CiteClaw public web app

Multi-tenant, invite-gated deployment of the live web UI (`web/live/`) on
Modal. The local UI is untouched — this package imports its backend modules
and serves its frontend with a public layer on top.

**URL**: https://cola-lab--citeclaw-web-serve.modal.run

## Operate

```bash
# one-time
modal secret create citeclaw-web CITECLAW_SESSION_SECRET=$(openssl rand -hex 32)

modal deploy modal_webapp.py                      # deploy / redeploy
modal run modal_webapp.py::mint --n 3 --note "…"  # invite codes (shown once)
modal run modal_webapp.py::codes                  # list (hash prefix, uses)
modal run modal_webapp.py::revoke --prefix ab12cd # disable
modal app logs citeclaw-web                       # tail server logs
```

## Architecture

* **One container, scale-to-zero** (`scaledown_window=60`, `max_containers=1`).
  An open tab's 15 s `/api/auth/me` poll keeps it warm; ~1 min after the last
  visitor leaves it sleeps. Cold start ≈ a few seconds. A run whose tab
  closes is reaped at scaledown — the UI warns users to keep the tab open
  (SIGTERM asks runs to stop and snapshots state on the way down).
* **Volume** `citeclaw-web-data` at `/data`: invite hashes, per-session
  `session.json` (keys AES-GCM-encrypted with a key derived from the session
  secret), run artifacts, and cache snapshots. Session-critical writes
  (join, settings) commit the volume eagerly.
* **Shared cache**: every run's `cache.db` is a symlink to one
  container-local SQLite (S2 metadata / references / embeddings / LLM
  responses shared across all tenants). SQLite never touches the FUSE
  volume; `cache_sync` moves consistent snapshots both ways.
* **Sessions**: invite code → HMAC-signed HttpOnly cookie (90 d) → private
  workspace. Keys are never placed in `os.environ` (shared process); they
  ride into each run through `Settings` overrides, and the server scrubs
  ambient key env vars at startup.
* **Isolation details**: run ownership checked on every run/stream/download
  route; explore endpoints are rooted at the session's runs dir; the
  per-run log bridge is scoped to its run thread so concurrent tenants
  can't see each other's log lines.

## Caps (env-tunable, see `backend/limits.py`)

≤500 papers/run · 1 concurrent run per session · 2 global · 20 runs/day ·
runs kept 30 days or last 20 · S2 rate clamped to 1 rps · join attempts
rate-limited per IP · 2 MB request bodies.

## v1 scope

Users bring their own keys (Gemini for the supported model; S2 optional but
recommended — keyless shares the app egress IP's pool). Everything
PDF-related (fetch-pdfs, ExpandByPDF, full-text scope) is out — it depends
on a local Chrome + institutional SSO. `stub` model stays enabled
(`CITECLAW_WEBUI_ALLOW_STUB=1`) for key-free demo runs.

## Local development

```bash
CITECLAW_PUBLIC_DATA=/tmp/pubdata CITECLAW_PUBLIC_LOCAL=/tmp/publocal \
CITECLAW_PUBLIC_INSECURE_COOKIE=1 CITECLAW_WEBUI_ALLOW_STUB=1 \
CITECLAW_SESSION_SECRET=dev PYTHONPATH=src:. \
python -m uvicorn web.public.backend.server:app --port 8898
# mint a local code:
CITECLAW_PUBLIC_DATA=/tmp/pubdata PYTHONPATH=src:. python -c \
  "from web.public.backend import auth, paths; paths.ensure_layout(); print(auth.mint_codes(1)[0])"
```

Tests: `PYTHONPATH=src python -m pytest tests/test_web_public_auth.py tests/test_web_public_server.py`
