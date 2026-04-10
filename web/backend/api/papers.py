"""Paper lookup endpoint — reads from the SQLite cache."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/papers", tags=["papers"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _cache_db_path() -> Path:
    """Resolve the cache.db path from env or default data_dir."""
    data_dir = os.environ.get("CITECLAW_DATA_DIR", str(PROJECT_ROOT / "data_bio"))
    return Path(data_dir) / "cache.db"


@router.get("/{paper_id:path}")
async def get_paper(paper_id: str) -> dict[str, Any]:
    """Return paper metadata from the SQLite cache."""
    db_path = _cache_db_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Cache database not found")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT data FROM paper_metadata WHERE key = ?", (paper_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    return json.loads(row[0])
