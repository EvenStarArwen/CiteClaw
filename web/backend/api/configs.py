"""Config YAML endpoints: list, read, write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/configs", tags=["configs"])

# Project root is two levels up from web/backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _config_dir() -> Path:
    """Return the directory where YAML configs live (project root)."""
    return PROJECT_ROOT


@router.get("")
async def list_configs() -> list[dict[str, str]]:
    """List all ``config*.yaml`` files in the project root."""
    configs = sorted(_config_dir().glob("config*.yaml"))
    return [{"name": p.name} for p in configs]


@router.get("/{name}")
async def get_config(name: str) -> dict[str, Any]:
    """Read a YAML config and return it as JSON."""
    path = _config_dir() / name
    if not path.exists() or not path.name.startswith("config"):
        raise HTTPException(status_code=404, detail=f"Config not found: {name}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


@router.post("/{name}")
async def save_config(name: str, body: dict[str, Any]) -> dict[str, str]:
    """Write a config from JSON payload to YAML."""
    if not name.startswith("config") or not name.endswith(".yaml"):
        raise HTTPException(
            status_code=400,
            detail="Config name must match config*.yaml",
        )
    path = _config_dir() / name
    with open(path, "w") as f:
        yaml.dump(body, f, default_flow_style=False, sort_keys=False)
    return {"status": "saved", "name": name}
