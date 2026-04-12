"""Live smoke tests — real S2 API + real LLM (Gemma 4 31B on Modal).

Gated behind ``CITECLAW_LIVE_SMOKE=1``.  Required env vars::

    CITECLAW_LIVE_SMOKE=1
    S2_API_KEY=<key>
    CITECLAW_VLLM_API_KEY=<key>          # for LLM-using pipelines
    CITECLAW_SMOKE_LLM_URL=<url>         # optional, has default

Run::

    pytest tests/test_smoke_live.py -x -v --timeout=600
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from citeclaw.cache import Cache
from citeclaw.clients.s2.api import SemanticScholarClient
from citeclaw.config import BudgetTracker, SeedPaper, Settings
from citeclaw.context import Context
from citeclaw.event_sink import RecordingEventSink
from citeclaw.pipeline import run_pipeline

from tests.smoke_pipelines import PIPELINES
from tests.smoke_topics import TOPICS

pytestmark = pytest.mark.live_smoke

# ---------------------------------------------------------------------------
# JSONL log handler — writes one JSON object per log record
# ---------------------------------------------------------------------------


class _JSONLHandler(logging.Handler):
    """Append-only JSONL handler for post-hoc inspection."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._fh = open(path, "w")  # noqa: SIM115

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": self.format(record),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        try:
            self._fh.write(json.dumps(entry, default=str) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass  # never let logging kill the test

    def close(self) -> None:
        self._fh.close()
        super().close()


# ---------------------------------------------------------------------------
# Settings builder
# ---------------------------------------------------------------------------

_DEFAULT_LLM_URL = (
    "https://cola-lab--citeclaw-vllm-gemma-serve.modal.run/v1"
)


def _build_settings(
    topic,
    pipeline_cfg: dict,
    data_dir: Path,
) -> Settings:
    llm_url = os.environ.get("CITECLAW_SMOKE_LLM_URL", _DEFAULT_LLM_URL)
    return Settings(
        screening_model="gemma-4-31b",
        models={
            "gemma-4-31b": {
                "base_url": llm_url,
                "served_model_name": "google/gemma-4-31B-it",
                "api_key_env": "CITECLAW_VLLM_API_KEY",
            },
        },
        data_dir=data_dir,
        topic_description=topic.description,
        seed_papers=[SeedPaper(paper_id=sid) for sid in topic.seed_ids],
        blocks=pipeline_cfg["blocks"],
        pipeline=pipeline_cfg["pipeline"],
        max_papers_total=pipeline_cfg["max_papers_total"],
        llm_batch_size=10,
        llm_concurrency=16,
        s2_rps=0.9,
    )


# ---------------------------------------------------------------------------
# 100-case parametrized matrix (5 topics × 20 pipelines)
# ---------------------------------------------------------------------------

_CASES = [(t, p) for t in TOPICS for p in PIPELINES]
_IDS = [f"{t.name}--{p.__name__}" for t, p in _CASES]


@pytest.mark.parametrize("topic,pipeline_fn", _CASES, ids=_IDS)
def test_smoke_pipeline(tmp_path: Path, topic, pipeline_fn, monkeypatch) -> None:
    # ---- gate ----
    if os.environ.get("CITECLAW_LIVE_SMOKE", "") not in ("1", "true"):
        pytest.skip("Set CITECLAW_LIVE_SMOKE=1 to run live smoke tests")

    monkeypatch.setenv("CITECLAW_NO_DASHBOARD", "1")

    # ---- JSONL logging ----
    log_path = tmp_path / "smoke_log.jsonl"
    handler = _JSONLHandler(log_path)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("citeclaw")
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    cache: Cache | None = None
    try:
        # ---- build ----
        pipeline_cfg = pipeline_fn(topic)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        settings = _build_settings(topic, pipeline_cfg, data_dir)
        budget = BudgetTracker()
        cache = Cache(data_dir / "cache.db")
        s2 = SemanticScholarClient(settings, cache, budget)
        ctx = Context(config=settings, s2=s2, cache=cache, budget=budget)

        # ---- run ----
        sink = RecordingEventSink()
        run_pipeline(ctx, event_sink=sink)

        # ---- assertions ----
        # 1. collection is non-empty (at least seed)
        assert len(ctx.collection) >= 1, (
            f"Collection empty — expected at least the seed paper(s)"
        )

        # 2. output artifacts exist
        assert (data_dir / "literature_collection.json").exists(), (
            "literature_collection.json not written"
        )
        # Finalize skips graph export when collection < 2 papers
        if len(ctx.collection) >= 2:
            assert (data_dir / "citation_network.graphml").exists(), (
                "citation_network.graphml not written"
            )
        assert (data_dir / "shape_summary.txt").exists(), (
            "shape_summary.txt not written"
        )

        # 3. real S2 traffic happened
        assert budget.s2_requests > 0, "Expected real S2 API calls"

        # 4. LLM traffic for pipelines that need it
        if pipeline_cfg.get("needs_llm"):
            assert budget.llm_calls > 0, "Expected LLM calls for this pipeline"

        # 5. cluster results for pipelines that need them
        if pipeline_cfg.get("needs_cluster"):
            assert len(ctx.clusters) > 0, "Expected cluster results"

        # 6. log file is a first-class artifact
        assert log_path.exists() and log_path.stat().st_size > 0, (
            "smoke_log.jsonl missing or empty"
        )

    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        handler.close()
        if cache is not None:
            cache.close()
