"""Shared pytest fixtures for the CiteClaw test suite.

Every fixture here is built around two guarantees:

1. **No real LLM traffic.** The ``basic_settings`` fixture forces
   ``screening_model="stub"``, so any ``build_llm_client(...)`` call
   returns the deterministic :class:`StubClient`. Tests that still want
   a hand-rolled LLM client can inject one by writing to the
   ``_LLM_CLIENTS`` dict in ``citeclaw.screening.llm_runner``.

2. **No real S2 traffic by default.** The ``ctx`` fixture wires in
   :class:`tests.fakes.FakeS2Client`, an in-memory duck-type of the
   real Semantic Scholar client. A small ``live_s2`` fixture builds a
   real client — gated behind the ``CITECLAW_LIVE_S2=1`` env var — for the
   handful of tests that exercise the live wire format.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.screening import llm_runner as _llm_runner
from tests.fakes import FakeS2Client, build_chain_corpus


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest doesn't complain."""
    config.addinivalue_line(
        "markers",
        "live_s2: hits the real Semantic Scholar API; opt in with CITECLAW_LIVE_S2=1",
    )
    config.addinivalue_line(
        "markers",
        "live_smoke: real S2 + real LLM smoke tests; opt in with CITECLAW_LIVE_SMOKE=1",
    )


@pytest.fixture(autouse=True)
def _reset_llm_client_cache():
    """Clear the LLM client cache between tests so they never leak state."""
    _llm_runner._LLM_CLIENTS.clear()
    yield
    _llm_runner._LLM_CLIENTS.clear()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def basic_settings(tmp_data_dir: Path) -> Settings:
    """Minimal Settings: stub LLM, tmp data dir, no blocks, no pipeline."""
    return Settings(
        screening_model="stub",
        data_dir=tmp_data_dir,
        topic_description="Deep learning for biology.",
        seed_papers=[],
        blocks={},
        pipeline=[],
        llm_batch_size=4,
        llm_concurrency=2,
    )


@pytest.fixture
def budget() -> BudgetTracker:
    return BudgetTracker()


@pytest.fixture
def cache(tmp_data_dir: Path) -> Cache:
    c = Cache(tmp_data_dir / "cache.db")
    yield c
    c.close()


@pytest.fixture
def fake_s2() -> FakeS2Client:
    return build_chain_corpus()


@pytest.fixture
def ctx(
    basic_settings: Settings,
    fake_s2: FakeS2Client,
    cache: Cache,
    budget: BudgetTracker,
) -> Context:
    """A Context wired up with the fake S2 client — no real network calls."""
    return Context(config=basic_settings, s2=fake_s2, cache=cache, budget=budget)


@pytest.fixture
def live_s2_allowed() -> bool:
    """True when the user has opted in to hitting the real S2 API."""
    return os.environ.get("CITECLAW_LIVE_S2", "").lower() in {"1", "true", "yes"}
