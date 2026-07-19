"""Multi-tenant run manager for the public app.

Subclasses the live ``RunManager`` with:

  * ownership — every run belongs to a session; lookups are scoped
  * concurrency gates — per-session and global caps
  * shared-cache wiring — each run's ``cache.db`` is a symlink to one
    container-local SQLite file (SQLite on the FUSE volume is unsafe;
    ``cache_sync`` persists the local file to the volume instead)
  * retention — old runs pruned per session after each run ends
  * log scoping — ``scope_logs_to_thread`` keeps one tenant's log lines
    out of another tenant's browser
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from typing import Any, Callable

from web.live.backend.run_manager import RunManager, RunState

from . import limits, paths


class CapacityError(RuntimeError):
    """Run refused because a concurrency cap is hit (user-facing message)."""


class PublicRunManager(RunManager):
    scope_logs_to_thread = True

    def __init__(self) -> None:
        super().__init__()
        self._gate = threading.Lock()
        # Called after every run ends (the Modal layer hooks volume commits).
        self.post_run_hook: Callable[[], None] | None = None

    # ---- gates --------------------------------------------------------
    def _active(self) -> list[RunState]:
        return [rs for rs in self.runs.values()
                if rs.status in ("starting", "running")]

    def active_for(self, sid: str) -> list[RunState]:
        return [rs for rs in self._active() if rs.owner == sid]

    def new_run_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start_run_for(self, sid: str, config, run_id: str) -> tuple[str, list[dict[str, Any]]]:
        with self._gate:
            if len(self.active_for(sid)) >= limits.PER_SESSION_CONCURRENT:
                raise CapacityError(
                    "You already have a run in progress — stop it or let it "
                    "finish before starting another.")
            if len(self._active()) >= limits.GLOBAL_CONCURRENT:
                raise CapacityError(
                    "The server is at its concurrent-run capacity right now. "
                    "Try again in a few minutes.")
            return self.start_run(config, owner=sid, run_id=run_id)

    def get_owned(self, run_id: str, sid: str) -> RunState | None:
        rs = self.get(run_id)
        return rs if rs is not None and rs.owner == sid else None

    # ---- run dir / shared cache --------------------------------------
    @staticmethod
    def prepare_run_dir(sid: str, run_id: str) -> str:
        """Create the session run dir with cache.db pre-linked to the shared
        local cache, so ``build_context`` opens one cross-tenant SQLite."""
        d = paths.run_dir(sid, run_id)
        d.mkdir(parents=True, exist_ok=True)
        paths.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        link = d / "cache.db"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(paths.LOCAL_CACHE)
        return str(d)

    # ---- retention ----------------------------------------------------
    def on_run_end(self, rs: RunState) -> None:
        if rs.owner:
            try:
                self._prune_session_runs(rs.owner)
            except Exception:  # noqa: BLE001 - retention is best-effort
                pass
        if self.post_run_hook is not None:
            self.post_run_hook()

    def _prune_session_runs(self, sid: str) -> None:
        root = paths.session_runs_dir(sid)
        if not root.is_dir():
            return
        active_dirs = {str(paths.run_dir(sid, rs.run_id))
                       for rs in self.active_for(sid) if paths.valid_rid(rs.run_id)}
        dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        cutoff = time.time() - limits.RUN_TTL_DAYS * 86400
        for i, p in enumerate(dirs):
            if str(p) in active_dirs:
                continue
            if i >= limits.MAX_RUNS_KEPT or p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)


manager = PublicRunManager()
