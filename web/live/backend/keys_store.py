"""Read/write API keys to a private local ``.env.local`` file.

The design's Settings panel POSTs keys here; we persist them to
``<repo>/.env.local`` (git-ignored) and mirror them into ``os.environ``
so ``citeclaw.config.load_settings`` picks them up on the next run. Keys
are never sent back to the browser — only their presence.
"""

from __future__ import annotations

import os
from pathlib import Path

# repo root = .../CiteClaw  (this file is web/live/backend/keys_store.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env.local"

# UI field name -> canonical env var CiteClaw reads.
FIELD_ENV = {
    "gemini_api_key": "GEMINI_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "s2_api_key": "S2_API_KEY",
}


def _read_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_PATH.exists():
        return out
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env_file(values: dict[str, str]) -> None:
    lines = ["# CiteClaw live WebUI — local API keys. Git-ignored. Do not commit.\n"]
    for k in sorted(values):
        if values[k]:
            lines.append(f"{k}={values[k]}\n")
    ENV_PATH.write_text("".join(lines))
    try:
        ENV_PATH.chmod(0o600)
    except OSError:
        pass


def load_into_environ() -> None:
    """Load ``.env.local`` values into ``os.environ`` (existing env wins)."""
    for k, v in _read_env_file().items():
        if v and not os.environ.get(k):
            os.environ[k] = v


def update_keys(patch: dict[str, str]) -> None:
    """Persist provided keys and mirror them into the current process env.

    ``patch`` maps UI field names (``gemini_api_key`` …) to raw key values.
    Empty / missing fields are left untouched (not cleared).
    """
    current = _read_env_file()
    for field, env_var in FIELD_ENV.items():
        val = (patch.get(field) or "").strip()
        if val:
            current[env_var] = val
            os.environ[env_var] = val
    _write_env_file(current)


def key_presence() -> dict[str, bool]:
    """Return which keys are set (env var present and non-empty)."""
    presence: dict[str, bool] = {}
    for field, env_var in FIELD_ENV.items():
        presence[field] = bool(os.environ.get(env_var, "").strip())
    return presence
