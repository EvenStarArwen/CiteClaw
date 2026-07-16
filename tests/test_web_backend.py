"""Tests for the first live local Web UI backend."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1] / "web" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import runtime  # noqa: E402
from citeclaw.models import PaperRecord  # noqa: E402
from catalog import SUPPORTED_MODEL, catalog_payload  # noqa: E402
from main import create_app  # noqa: E402
from runtime import (  # noqa: E402
    RunManager,
    UnsupportedRunConfiguration,
    prepare_config,
)


def minimal_yaml(*, model: str = SUPPORTED_MODEL) -> str:
    return f"""
screening_model: {model}
reasoning_effort: minimal
topic_description: local backend test
seed_papers: []
pipeline: []
"""


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_catalog_marks_only_requested_model_runnable():
    payload = catalog_payload()
    runnable = [model["id"] for model in payload["models"] if model["runnable"]]
    assert runnable == [SUPPORTED_MODEL]
    assert payload["sources"]["openai"].startswith("https://developers.openai.com/")
    assert payload["sources"]["gemini"].startswith("https://ai.google.dev/")


def test_validate_endpoint_accepts_real_config(client: TestClient):
    response = client.post(
        "/api/configs/validate/yaml",
        json={"yaml": (Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text()},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["model"] == SUPPORTED_MODEL


def test_validate_endpoint_rejects_credentials_in_yaml(client: TestClient):
    response = client.post(
        "/api/configs/validate/yaml",
        json={"yaml": minimal_yaml() + "\ngemini_api_key: should-never-be-saved\n"},
    )
    assert response.status_code == 422
    assert "API keys must not be set" in response.json()["detail"]


def test_prepare_config_requires_supported_model(tmp_path: Path):
    with pytest.raises(UnsupportedRunConfiguration, match="Model not supported"):
        prepare_config(
            minimal_yaml(model="gpt-5.6-sol"),
            {"s2_api_key": "s2-test", "gemini_api_key": "gem-test"},
            output_dir=tmp_path,
        )


def test_prepare_config_keeps_keys_out_of_snapshot(tmp_path: Path):
    snapshot, settings = prepare_config(
        minimal_yaml(),
        {
            "s2_api_key": "s2-secret",
            "gemini_api_key": "gem-secret",
            "openai_api_key": "oa-secret",
        },
        output_dir=tmp_path,
    )
    assert settings.s2_api_key == "s2-secret"
    assert settings.gemini_api_key == "gem-secret"
    assert settings.openai_api_key == "oa-secret"
    assert not {"s2_api_key", "gemini_api_key", "openai_api_key"} & snapshot.keys()


def test_graph_payload_contains_papers_and_citation_edges():
    seed = PaperRecord(
        paper_id="seed",
        title="Seed paper",
        source="seed",
        year=2024,
        authors=[{"name": "Ada Author"}],
    )
    citing = PaperRecord(
        paper_id="citing",
        title="Citing paper",
        references=["seed"],
        citation_count=12,
    )
    payload = runtime.graph_payload(SimpleNamespace(collection={"seed": seed, "citing": citing}, seed_ids={"seed"}))

    assert {node["paper_id"] for node in payload["nodes"]} == {"seed", "citing"}
    assert next(node for node in payload["nodes"] if node["paper_id"] == "seed")["seed"] is True
    assert payload["edges"] == [{"source": "seed", "target": "citing"}]


def test_run_log_handler_redacts_credentials(tmp_path: Path):
    session = runtime.RunSession(
        run_id="redaction-test",
        config_name="test.yaml",
        output_dir=tmp_path,
        config_snapshot={},
    )
    handler = runtime._RunLogHandler(session, ["environment-secret"])
    handler.emit(logging.makeLogRecord({"levelname": "INFO", "msg": "key=environment-secret"}))

    assert session.events[-1]["message"] == "key=[redacted]"


def test_run_manager_executes_pipeline_off_request_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runtime, "RUNS_ROOT", tmp_path)
    monkeypatch.setenv("CITECLAW_NO_DASHBOARD", "1")
    manager = RunManager()
    session = manager.start(
        config_yaml=minimal_yaml(),
        config_name="test.yaml",
        credentials={"s2_api_key": "s2-test", "gemini_api_key": "gem-test"},
    )
    deadline = time.monotonic() + 5
    while session.status not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(0.02)

    assert session.status == "completed", session.error
    event_types = [event["type"] for event in session.events]
    assert event_types[0] == "run_started"
    assert "step_start" in event_types
    assert event_types[-1] == "run_complete"
    assert (session.output_dir / "config.yaml").exists()
    written = (session.output_dir / "config.yaml").read_text()
    assert "s2-test" not in written
    assert "gem-test" not in written
