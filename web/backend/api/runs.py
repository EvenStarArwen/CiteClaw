"""Run management endpoints — read state, trigger new runs.

PE-09 adds ``POST /api/runs/{run_id}/hitl`` which accepts user labels
for the ``HumanInTheLoop`` step's web-mode gate. The gate registry
maps run_id → ``HitlGate`` and is populated by the pipeline runner
when it starts a web-mode run.
"""

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

# In-memory registry of active HITL gates, keyed by run_id.
# The pipeline runner registers a gate here when starting a web-mode
# run; the POST /hitl endpoint looks it up to deliver user labels.
_hitl_gates: dict[str, Any] = {}


def register_hitl_gate(run_id: str, gate: Any) -> None:
    """Register a ``HitlGate`` for a run so the HITL endpoint can find it."""
    _hitl_gates[run_id] = gate


def unregister_hitl_gate(run_id: str) -> None:
    """Remove a gate when the run finishes."""
    _hitl_gates.pop(run_id, None)


def _data_dir() -> Path:
    return Path(os.environ.get("CITECLAW_DATA_DIR", str(PROJECT_ROOT / "data_bio")))


class RunRequest(BaseModel):
    config_name: str


class HitlResponse(BaseModel):
    """User labels submitted from the frontend HitlModal."""
    labels: dict[str, bool]
    stop_requested: bool = False


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


@router.post("/{run_id}/hitl")
async def submit_hitl_labels(
    run_id: str, body: HitlResponse,
) -> dict[str, str]:
    """Receive user labels from the frontend HitlModal and unblock the
    ``HumanInTheLoop`` step's gate.

    The pipeline thread is blocked on ``gate.event.wait()``; this
    endpoint writes labels into the gate and sets the event.
    """
    gate = _hitl_gates.get(run_id)
    if gate is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active HITL gate for run {run_id}",
        )
    gate.labels.update(body.labels)
    gate.stop_requested = body.stop_requested
    gate.event.set()
    return {"status": "accepted", "labels_count": str(len(body.labels))}
