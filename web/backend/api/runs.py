"""Run management endpoints — read state, trigger new runs."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/runs", tags=["runs"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _data_dir() -> Path:
    return Path(os.environ.get("CITECLAW_DATA_DIR", str(PROJECT_ROOT / "data_bio")))


class RunRequest(BaseModel):
    config_name: str


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Return run state from ``<data_dir>/run_state.json``.

    For now ``run_id`` is treated as the data directory name. The
    default data directory is used when ``run_id`` equals ``"latest"``.
    """
    if run_id == "latest":
        state_path = _data_dir() / "run_state.json"
    else:
        state_path = PROJECT_ROOT / run_id / "run_state.json"

    if not state_path.exists():
        raise HTTPException(status_code=404, detail=f"Run state not found: {run_id}")
    with open(state_path) as f:
        return json.load(f)


@router.post("")
async def create_run(req: RunRequest) -> dict[str, str]:
    """Trigger a new pipeline run (placeholder — returns a run_id).

    Full implementation will spawn a background task with the pipeline
    runner. For now it validates the config exists and returns an ID.
    """
    config_path = PROJECT_ROOT / req.config_name
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Config not found: {req.config_name}",
        )
    run_id = str(uuid.uuid4())[:8]
    return {"run_id": run_id, "config_name": req.config_name, "status": "accepted"}
