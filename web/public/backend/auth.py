"""Invite codes + signed session cookies for the public app.

Codes are stored as sha256 hashes in ``invites.json`` on the volume — the
plaintext exists only in the mint command's output. A valid code creates a
session (a directory keyed by a random sid) and sets an HMAC-signed,
HttpOnly cookie; every ``/api/*`` route then requires that cookie via the
``require_session`` dependency. No passwords, no accounts, nothing for a
password manager to hold — possession of an unexpired cookie IS the login.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Any

from fastapi import HTTPException, Request

from . import limits, paths

COOKIE_NAME = "ccs"
COOKIE_TTL_S = 90 * 24 * 3600

_lock = threading.Lock()

_secret: str | None = None


def get_secret() -> str:
    """Session-signing secret from the environment (Modal Secret in prod).

    Falls back to an ephemeral random secret so local testing works, at
    the cost of sessions dying with the process — fine locally, loud in
    the logs so a misconfigured deploy is noticed.
    """
    global _secret
    if _secret is None:
        v = os.environ.get("CITECLAW_SESSION_SECRET", "").strip()
        if not v:
            v = secrets.token_hex(32)
            print("[citeclaw-public] WARNING: CITECLAW_SESSION_SECRET unset — "
                  "using an ephemeral secret; sessions will not survive restarts.")
        _secret = v
    return _secret


# ---------------------------------------------------------------- invites

def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode()).hexdigest()


def _read_invites() -> dict[str, Any]:
    try:
        return json.loads(paths.INVITES_FILE.read_text())
    except (OSError, ValueError):
        return {"codes": {}}


def _write_json_atomic(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def mint_codes(n: int, note: str = "") -> list[str]:
    """Create ``n`` invite codes; returns the plaintexts (shown once)."""
    paths.ensure_layout()
    out = []
    with _lock:
        data = _read_invites()
        for _ in range(max(1, int(n))):
            code = "CC-" + secrets.token_hex(2).upper() + "-" + secrets.token_hex(2).upper()
            data["codes"][_hash_code(code)] = {
                "note": note, "created": time.time(),
                "disabled": False, "uses": 0, "last_used": None,
            }
            out.append(code)
        _write_json_atomic(paths.INVITES_FILE, data)
    return out


def list_codes() -> list[dict[str, Any]]:
    data = _read_invites()
    rows = []
    for h, meta in sorted(data["codes"].items(), key=lambda kv: kv[1].get("created", 0)):
        rows.append({"hash": h[:12], **meta})
    return rows


def disable_code(hash_prefix: str) -> int:
    """Disable every code whose hash starts with ``hash_prefix``."""
    n = 0
    with _lock:
        data = _read_invites()
        for h, meta in data["codes"].items():
            if h.startswith(hash_prefix.lower()) and not meta.get("disabled"):
                meta["disabled"] = True
                n += 1
        if n:
            _write_json_atomic(paths.INVITES_FILE, data)
    return n


def check_code(code: str) -> str | None:
    """Return the code's hash when valid + enabled, else None."""
    h = _hash_code(code or "")
    with _lock:
        data = _read_invites()
        meta = data["codes"].get(h)
        if not meta or meta.get("disabled"):
            return None
        meta["uses"] = int(meta.get("uses", 0)) + 1
        meta["last_used"] = time.time()
        _write_json_atomic(paths.INVITES_FILE, data)
    return h


# ------------------------------------------------------------- brute force

_join_attempts: dict[str, list[float]] = {}


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def join_allowed(ip: str) -> bool:
    now = time.time()
    with _lock:
        window = [t for t in _join_attempts.get(ip, []) if now - t < limits.JOIN_WINDOW_S]
        if len(window) >= limits.JOIN_ATTEMPTS:
            _join_attempts[ip] = window
            return False
        window.append(now)
        _join_attempts[ip] = window
    return True


# ---------------------------------------------------------------- sessions

def create_session(code_hash: str) -> str:
    sid = secrets.token_hex(16)
    d = paths.session_dir(sid)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    _write_json_atomic(paths.session_file(sid), {
        "sid": sid, "code_hash": code_hash,
        "created": time.time(), "last_seen": time.time(),
        "keys": {}, "settings": {}, "usage": {},
    })
    return sid


def load_session(sid: str) -> dict[str, Any] | None:
    if not paths.valid_sid(sid):
        return None
    try:
        return json.loads(paths.session_file(sid).read_text())
    except (OSError, ValueError):
        return None


def save_session(sess: dict[str, Any]) -> None:
    with _lock:
        _write_json_atomic(paths.session_file(sess["sid"]), sess)


def touch_session(sess: dict[str, Any]) -> None:
    """Refresh last_seen, throttled so routine traffic doesn't spam writes."""
    now = time.time()
    if now - float(sess.get("last_seen", 0)) > 600:
        sess["last_seen"] = now
        save_session(sess)


# ----------------------------------------------------------------- cookies

def _sign(payload: str) -> str:
    return hmac.new(get_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_cookie(sid: str) -> str:
    exp = int(time.time()) + COOKIE_TTL_S
    payload = f"{sid}.{exp}"
    return f"{payload}.{_sign(payload)}"


def parse_cookie(value: str) -> str | None:
    try:
        sid, exp_s, sig = value.split(".")
    except (AttributeError, ValueError):
        return None
    payload = f"{sid}.{exp_s}"
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    try:
        if int(exp_s) < time.time():
            return None
    except ValueError:
        return None
    return sid if paths.valid_sid(sid) else None


def cookie_kwargs() -> dict[str, Any]:
    # Local http testing needs secure=False; Modal serves https.
    insecure = os.environ.get("CITECLAW_PUBLIC_INSECURE_COOKIE", "") not in ("", "0")
    return {"httponly": True, "samesite": "lax", "secure": not insecure,
            "max_age": COOKIE_TTL_S, "path": "/"}


def session_from_request(request: Request) -> dict[str, Any] | None:
    sid = parse_cookie(request.cookies.get(COOKIE_NAME, ""))
    if not sid:
        return None
    return load_session(sid)


async def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency: 401 unless a valid session cookie is presented."""
    sess = session_from_request(request)
    if sess is None:
        raise HTTPException(status_code=401, detail="Not signed in — enter an invite code.")
    touch_session(sess)
    return sess
