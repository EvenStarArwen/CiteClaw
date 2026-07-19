"""Filesystem layout for the public (multi-tenant) web app.

Everything durable lives under ``DATA_ROOT`` — a Modal Volume in
production (``/data``), any temp dir locally via ``CITECLAW_PUBLIC_DATA``.
SQLite must NOT live on the volume (FUSE file locking is unreliable), so
the shared cache sits on container-local disk under ``LOCAL_ROOT`` and is
backed up to / restored from the volume by ``cache_sync``.

Layout:

  DATA_ROOT/
    auth/invites.json            invite codes (sha256 hashes + metadata)
    sessions/<sid>/session.json  per-user keys / settings / quota state
    sessions/<sid>/runs/<rid>/   run artifacts (graphml, json, bib)
    cache/cache.db               volume backup of the shared cache
  LOCAL_ROOT/
    cache.db                     live shared SQLite cache (all runs symlink it)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CITECLAW_PUBLIC_DATA", "/data"))
LOCAL_ROOT = Path(os.environ.get("CITECLAW_PUBLIC_LOCAL", "/tmp/citeclaw-public"))

AUTH_DIR = DATA_ROOT / "auth"
INVITES_FILE = AUTH_DIR / "invites.json"
SESSIONS_DIR = DATA_ROOT / "sessions"
VOLUME_CACHE = DATA_ROOT / "cache" / "cache.db"
LOCAL_CACHE = LOCAL_ROOT / "cache.db"

_SID_RE = re.compile(r"^[0-9a-f]{32}$")
_RID_RE = re.compile(r"^[0-9a-f]{12}$")


def valid_sid(sid: str) -> bool:
    return bool(_SID_RE.match(sid or ""))


def valid_rid(rid: str) -> bool:
    return bool(_RID_RE.match(rid or ""))


def session_dir(sid: str) -> Path:
    if not valid_sid(sid):
        raise ValueError("bad session id")
    return SESSIONS_DIR / sid


def session_file(sid: str) -> Path:
    return session_dir(sid) / "session.json"


def session_runs_dir(sid: str) -> Path:
    return session_dir(sid) / "runs"


def run_dir(sid: str, rid: str) -> Path:
    if not valid_rid(rid):
        raise ValueError("bad run id")
    return session_runs_dir(sid) / rid


def ensure_layout() -> None:
    for d in (AUTH_DIR, SESSIONS_DIR, VOLUME_CACHE.parent, LOCAL_ROOT):
        d.mkdir(parents=True, exist_ok=True)
