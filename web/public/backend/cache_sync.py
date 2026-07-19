"""Shared-cache persistence between container-local disk and the volume.

SQLite needs real file locking, which FUSE-backed volumes don't reliably
provide — so the live database every run writes to sits on container-local
disk (``paths.LOCAL_CACHE``) and this module moves consistent snapshots
to/from the volume:

  boot     volume copy -> local   (cold start restores the shared cache)
  running  every ``INTERVAL``, if the local db changed: sqlite online
           backup -> local tmp -> byte-copy to volume -> atomic replace,
           then ``VOLUME_COMMIT()`` (set by the Modal layer; no-op locally)
  shutdown one final ``sync_now()``

Losing the seconds since the last snapshot is fine — it's a cache.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from typing import Callable

from . import paths

INTERVAL_S = int(os.environ.get("CITECLAW_PUBLIC_CACHE_SYNC_S", "120") or 120)

# The Modal entrypoint points this at Volume.commit; None → nothing to do.
VOLUME_COMMIT: Callable[[], None] | None = None

_last_sig: tuple[float, int] | None = None
_lock = threading.Lock()
_stop = threading.Event()
_thread: threading.Thread | None = None


def _sig() -> tuple[float, int] | None:
    try:
        st = paths.LOCAL_CACHE.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def restore() -> None:
    """Cold start: bring the volume's snapshot to local disk (if any)."""
    paths.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    if paths.VOLUME_CACHE.is_file() and not paths.LOCAL_CACHE.exists():
        shutil.copy2(paths.VOLUME_CACHE, paths.LOCAL_CACHE)


def backup() -> bool:
    """Consistent snapshot local -> volume. Returns True when copied."""
    global _last_sig
    with _lock:
        if not paths.LOCAL_CACHE.is_file():
            return False
        sig = _sig()
        if sig == _last_sig:
            return False
        snap = paths.LOCAL_CACHE.with_suffix(".snapshot")
        src = sqlite3.connect(str(paths.LOCAL_CACHE))
        try:
            dst = sqlite3.connect(str(snap))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        paths.VOLUME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = paths.VOLUME_CACHE.with_suffix(".tmp")
        shutil.copy2(snap, tmp)
        os.replace(tmp, paths.VOLUME_CACHE)
        snap.unlink(missing_ok=True)
        _last_sig = sig
    return True


def sync_now() -> None:
    try:
        backup()
        if VOLUME_COMMIT is not None:
            VOLUME_COMMIT()
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        print(f"[citeclaw-public] cache sync failed: {e}")


def _loop() -> None:
    while not _stop.wait(INTERVAL_S):
        sync_now()


def start() -> None:
    global _thread
    restore()
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=_loop, daemon=True, name="cache-sync")
        _thread.start()


def stop() -> None:
    _stop.set()
    sync_now()
