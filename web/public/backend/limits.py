"""Abuse caps for the public deployment.

The server enforces these regardless of what the client sends — an invite
code that leaks can waste at most this much. All tunable via env so the
Modal deploy can adjust without a code change.
"""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Hard ceiling on max_papers_total no matter what the UI asks for.
MAX_PAPERS_CEILING = _int("CITECLAW_PUBLIC_MAX_PAPERS", 500)
# One run at a time per session; a couple in flight across everyone.
PER_SESSION_CONCURRENT = _int("CITECLAW_PUBLIC_SESSION_CONCURRENT", 1)
GLOBAL_CONCURRENT = _int("CITECLAW_PUBLIC_GLOBAL_CONCURRENT", 2)
RUNS_PER_DAY = _int("CITECLAW_PUBLIC_RUNS_PER_DAY", 20)
# Old runs are pruned (oldest first) beyond this per-session count / age.
MAX_RUNS_KEPT = _int("CITECLAW_PUBLIC_MAX_RUNS_KEPT", 20)
RUN_TTL_DAYS = _int("CITECLAW_PUBLIC_RUN_TTL_DAYS", 30)
# Invite brute-force: join attempts per IP per window.
JOIN_ATTEMPTS = _int("CITECLAW_PUBLIC_JOIN_ATTEMPTS", 10)
JOIN_WINDOW_S = _int("CITECLAW_PUBLIC_JOIN_WINDOW_S", 600)
# Request body ceiling for config/JSON posts (bytes).
MAX_BODY_BYTES = _int("CITECLAW_PUBLIC_MAX_BODY", 2 * 1024 * 1024)
