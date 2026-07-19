"""Per-session keys, defaults and quota bookkeeping.

Keys are the sensitive part: they are the *user's own* Gemini / OpenAI /
S2 credentials, so they are AES-GCM-encrypted at rest with a key derived
from the server's session secret (if ``cryptography`` is importable —
otherwise stored plain with an explicit flag, never silently). They are
NEVER exported to ``os.environ``: with many tenants in one process the
environment is shared global state, so keys flow per-run through
``Settings`` overrides and per-request through explicit parameters.
"""

from __future__ import annotations

import base64
import hashlib
import time
from typing import Any

from . import auth, limits

# Same UI field names the live settings modal already posts.
KEY_FIELDS = ("gemini_api_key", "openai_api_key", "s2_api_key", "voyage_api_key")

try:  # optional dependency; the Modal image always has it
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAVE_AES = True
except Exception:  # noqa: BLE001 - any import failure → plaintext fallback
    _HAVE_AES = False


def _aes() -> "AESGCM":
    return AESGCM(hashlib.sha256(("keys:" + auth.get_secret()).encode()).digest())


def _enc(value: str) -> dict[str, Any]:
    if not _HAVE_AES:
        return {"enc": False, "v": value}
    import os as _os
    nonce = _os.urandom(12)
    ct = _aes().encrypt(nonce, value.encode(), None)
    return {"enc": True, "v": base64.b64encode(nonce + ct).decode()}


def _dec(rec: dict[str, Any]) -> str:
    if not rec:
        return ""
    if not rec.get("enc"):
        return str(rec.get("v") or "")
    if not _HAVE_AES:
        return ""
    try:
        raw = base64.b64decode(rec["v"])
        return _aes().decrypt(raw[:12], raw[12:], None).decode()
    except Exception:  # noqa: BLE001 - secret rotated → treat as unset
        return ""


# ------------------------------------------------------------------- keys

def update_keys(sess: dict[str, Any], patch: dict[str, Any]) -> None:
    """Store non-empty fields from ``patch``; empty/missing left untouched."""
    keys = sess.setdefault("keys", {})
    changed = False
    for field in KEY_FIELDS:
        val = str(patch.get(field) or "").strip()
        if val:
            keys[field] = _enc(val)
            changed = True
    if changed:
        auth.save_session(sess)


def get_key(sess: dict[str, Any], field: str) -> str:
    return _dec((sess.get("keys") or {}).get(field) or {})


def key_presence(sess: dict[str, Any]) -> dict[str, bool]:
    return {f: bool(get_key(sess, f)) for f in KEY_FIELDS}


def key_overrides(sess: dict[str, Any]) -> dict[str, str]:
    """Settings-override dict carrying this session's keys into a run."""
    out = {}
    for field in ("gemini_api_key", "openai_api_key", "s2_api_key"):
        v = get_key(sess, field)
        if v:
            out[field] = v
    return out


# --------------------------------------------------------------- settings

_DEFAULTS = {"model": "", "reasoning_effort": "", "max_papers": 200}


def get_settings(sess: dict[str, Any]) -> dict[str, Any]:
    s = dict(_DEFAULTS)
    s.update(sess.get("settings") or {})
    s["max_papers"] = min(int(s.get("max_papers") or 200), limits.MAX_PAPERS_CEILING)
    return s


def update_settings(sess: dict[str, Any], patch: dict[str, Any]) -> None:
    s = sess.setdefault("settings", {})
    if patch.get("model"):
        s["model"] = str(patch["model"]).strip()
    if patch.get("reasoning_effort"):
        s["reasoning_effort"] = str(patch["reasoning_effort"]).strip()
    if patch.get("max_papers") is not None:
        try:
            s["max_papers"] = max(1, min(limits.MAX_PAPERS_CEILING, int(patch["max_papers"])))
        except (TypeError, ValueError):
            pass
    auth.save_session(sess)


# ----------------------------------------------------------------- quotas

def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def runs_today(sess: dict[str, Any]) -> int:
    u = sess.get("usage") or {}
    return int(u.get("runs", 0)) if u.get("day") == _today() else 0


def can_start_run(sess: dict[str, Any]) -> str | None:
    """None when allowed, else a user-facing refusal."""
    if runs_today(sess) >= limits.RUNS_PER_DAY:
        return (f"Daily limit reached ({limits.RUNS_PER_DAY} runs). "
                "It resets at midnight UTC.")
    return None


def note_run_started(sess: dict[str, Any]) -> None:
    day = _today()
    u = sess.setdefault("usage", {})
    if u.get("day") != day:
        u["day"], u["runs"] = day, 0
    u["runs"] = int(u.get("runs", 0)) + 1
    auth.save_session(sess)
