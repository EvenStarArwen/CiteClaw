"""Session-scoped run listing + artifact downloads.

Public users cannot browse the server's filesystem, so this module is
their only window onto run outputs: a listing of the session's run dirs
(merged with live in-memory state) and a whitelist of downloadable
artifacts — by name or bundled as one zip. Nothing outside the session's
own directory is ever reachable (run ids are strict hex, no paths).
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from . import paths

# artifact key -> filename inside the run dir
ARTIFACTS = {
    "collection": "literature_collection.json",
    "bib": "literature_collection.bib",
    "citation": "citation_network.graphml",
    "collab": "collaboration_network.graphml",
    "state": "run_state.json",
}
# what the zip bundles (cache.db is a shared symlink — never shipped)
_ZIP_NAMES = tuple(ARTIFACTS.values()) + ("shape_summary.txt",)


def artifact_path(sid: str, rid: str, key: str) -> Path | None:
    name = ARTIFACTS.get(key)
    if not name:
        return None
    p = paths.run_dir(sid, rid) / name
    return p if p.is_file() else None


def artifact_presence(run_dir: Path) -> dict[str, bool]:
    return {k: (run_dir / n).is_file() for k, n in ARTIFACTS.items()}


def make_zip(sid: str, rid: str) -> bytes | None:
    d = paths.run_dir(sid, rid)
    files = [d / n for n in _ZIP_NAMES if (d / n).is_file()]
    if not files:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f"citeclaw_{rid}/{f.name}")
    return buf.getvalue()


def _dir_summary(p: Path) -> dict[str, Any]:
    n_papers = None
    jf = p / ARTIFACTS["collection"]
    if jf.is_file():
        try:
            data = json.loads(jf.read_text())
            n_papers = int((data.get("summary") or {}).get("total_accepted")
                           or len(data.get("papers") or []))
        except (OSError, ValueError):
            pass
    return {
        "run_id": p.name,
        "mtime": p.stat().st_mtime,
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
        "papers": n_papers,
        "artifacts": artifact_presence(p),
    }


def list_session_runs(sid: str, live_status: dict[str, str]) -> list[dict[str, Any]]:
    """All of a session's runs, newest first.

    ``live_status`` maps run_id -> status for runs the manager still holds
    in memory; disk-only runs (from before a container restart) show as
    ``finished`` when artifacts exist.
    """
    root = paths.session_runs_dir(sid)
    out = []
    if root.is_dir():
        for p in sorted((d for d in root.iterdir() if d.is_dir()),
                        key=lambda d: d.stat().st_mtime, reverse=True):
            if not paths.valid_rid(p.name):
                continue
            row = _dir_summary(p)
            row["status"] = live_status.get(
                p.name, "finished" if any(row["artifacts"].values()) else "empty")
            out.append(row)
    return out
