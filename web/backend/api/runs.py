"""Live run lifecycle and WebSocket endpoints."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from runtime import (
    RunConflictError,
    UnsupportedRunConfiguration,
    run_manager,
)


router = APIRouter(prefix="/api/runs", tags=["runs"])


class Credentials(BaseModel):
    s2_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""


class RunRequest(BaseModel):
    config_yaml: str = Field(min_length=1)
    config_name: str = "config.yaml"
    credentials: Credentials = Field(default_factory=Credentials)


class HitlResponse(BaseModel):
    labels: dict[str, bool]
    stop_requested: bool = False


@router.get("")
async def list_runs():
    return run_manager.list()


@router.get("/credentials/status")
async def credential_status():
    return {
        "s2": bool(
            os.environ.get("CITECLAW_S2_API_KEY")
            or os.environ.get("S2_API_KEY")
            or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        ),
        "gemini": bool(os.environ.get("CITECLAW_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "openai": bool(os.environ.get("CITECLAW_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    }


@router.post("")
async def create_run(req: RunRequest):
    try:
        session = run_manager.start(
            config_yaml=req.config_yaml,
            config_name=req.config_name,
            credentials=req.credentials.model_dump(),
        )
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (UnsupportedRunConfiguration, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return session.snapshot()


@router.get("/{run_id}")
async def get_run(run_id: str):
    session = run_manager.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return session.snapshot(include_graph=True)


@router.post("/{run_id}/hitl")
async def submit_hitl(run_id: str, body: HitlResponse):
    session = run_manager.get(run_id)
    if session is None or session.hitl_gate is None:
        raise HTTPException(status_code=404, detail=f"No active HITL request for {run_id}")
    session.hitl_gate.labels.update(body.labels)
    session.hitl_gate.stop_requested = body.stop_requested
    session.hitl_gate.event.set()
    return {"status": "accepted", "labels_count": len(body.labels)}


@router.websocket("/{run_id}/stream")
async def run_stream(websocket: WebSocket, run_id: str):
    session = run_manager.get(run_id)
    if session is None:
        await websocket.close(code=4404, reason="Run not found")
        return
    await websocket.accept()
    index = 0
    try:
        while True:
            events, next_index = session.events_since(index)
            for event in events:
                await websocket.send_json(event)
            index = next_index
            if session.status in {"completed", "failed"} and index >= len(session.events):
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
