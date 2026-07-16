"""Safe local YAML configuration endpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from citeclaw.config import Settings, _normalize_yaml
from runtime import PROJECT_ROOT


router = APIRouter(prefix="/api/configs", tags=["configs"])
CONFIG_DIR = PROJECT_ROOT / "configs"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.ya?ml$")


class YamlBody(BaseModel):
    yaml: str


def _path_for(name: str) -> Path:
    if not _SAFE_NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail="Config name must be a simple .yaml filename.")
    return CONFIG_DIR / name


def _parse(text: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="Config must contain a top-level mapping.")
    try:
        return _normalize_yaml(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
async def list_configs() -> list[dict[str, str]]:
    configs = sorted((*CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml")))
    return [{"name": path.name} for path in configs]


@router.get("/{name}")
async def get_config(name: str) -> dict[str, Any]:
    path = _path_for(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {name}")
    text = path.read_text(encoding="utf-8")
    return {"name": name, "yaml": text, "config": _parse(text)}


@router.put("/{name}")
async def save_config(name: str, body: YamlBody) -> dict[str, str]:
    path = _path_for(name)
    _parse(body.yaml)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(body.yaml.rstrip() + "\n", encoding="utf-8")
    return {"status": "saved", "name": name}


@router.post("/validate/yaml")
async def validate_config(body: YamlBody) -> dict[str, Any]:
    raw = _parse(body.yaml)
    try:
        settings = Settings(**raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "valid": True,
        "summary": {
            "model": settings.screening_model,
            "reasoning_effort": settings.reasoning_effort,
            "seeds": len(settings.seed_papers),
            "blocks": len(settings.blocks),
            "pipeline_steps": len(settings.pipeline),
        },
    }
