"""Unit tests for the ``rejections.json`` writer in Finalize.

End-to-end coverage lives in ``test_pipeline_e2e.py``; these tests
target the writer's formatting rules directly against a synthetic
context so we can assert the exact shape without running a pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from citeclaw.steps.finalize import _write_rejections_json


class _StubCache:
    """Minimal cache stand-in — returns pre-seeded metadata dicts."""

    def __init__(self, metadata: dict[str, dict]):
        self._metadata = metadata

    def get_metadata(self, paper_id: str):
        return self._metadata.get(paper_id)


class _StubCtx:
    def __init__(self, rejection_ledger, cache_data):
        self.rejection_ledger = rejection_ledger
        self.cache = _StubCache(cache_data)


def test_rejections_json_basic(tmp_path: Path):
    ctx = _StubCtx(
        rejection_ledger={
            "p1": ["year_filter"],
            "p2": ["citation", "llm_topic_llm"],
        },
        cache_data={
            "p1": {"title": "Old Paper", "year": 1980},
            "p2": {"title": "Weak Paper", "year": 2023},
        },
    )
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    data = json.loads(out.read_text())
    assert data["p1"]["categories"] == ["year_filter"]
    assert data["p1"]["title"] == "Old Paper"
    assert data["p1"]["year"] == 1980
    assert data["p2"]["categories"] == ["citation", "llm_topic_llm"]


def test_rejections_json_dedups_categories(tmp_path: Path):
    """A paper rejected by the same category twice gets a single entry."""
    ctx = _StubCtx(
        rejection_ledger={
            "p1": ["llm_topic", "llm_topic", "year_filter"],
        },
        cache_data={},
    )
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    data = json.loads(out.read_text())
    assert data["p1"]["categories"] == ["llm_topic", "year_filter"]


def test_rejections_json_truncates_long_titles(tmp_path: Path):
    long_title = "X" * 500
    ctx = _StubCtx(
        rejection_ledger={"p1": ["year_filter"]},
        cache_data={"p1": {"title": long_title, "year": 2020}},
    )
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    data = json.loads(out.read_text())
    assert len(data["p1"]["title"]) == 200


def test_rejections_json_handles_missing_cache_entry(tmp_path: Path):
    ctx = _StubCtx(
        rejection_ledger={"p1": ["year_filter"]},
        cache_data={},
    )
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    data = json.loads(out.read_text())
    assert data["p1"] == {"categories": ["year_filter"]}


def test_rejections_json_omits_abstract_field(tmp_path: Path):
    """Deliberate design: no abstract in the output, even if the cache has one."""
    ctx = _StubCtx(
        rejection_ledger={"p1": ["year_filter"]},
        cache_data={"p1": {"title": "T", "year": 2020, "abstract": "A" * 5000}},
    )
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    data = json.loads(out.read_text())
    assert "abstract" not in data["p1"]


def test_rejections_json_empty_ledger(tmp_path: Path):
    ctx = _StubCtx(rejection_ledger={}, cache_data={})
    out = tmp_path / "rejections.json"
    _write_rejections_json(ctx, out)
    assert json.loads(out.read_text()) == {}


def test_rejections_json_cache_exception_swallowed(tmp_path: Path):
    """A cache read failure must not crash the writer — just omit metadata."""

    class _BoomCache:
        def get_metadata(self, _pid: str):
            raise RuntimeError("sqlite locked")

    class _BoomCtx:
        rejection_ledger = {"p1": ["year_filter"]}
        cache = _BoomCache()

    out = tmp_path / "rejections.json"
    _write_rejections_json(_BoomCtx(), out)
    data = json.loads(out.read_text())
    assert data["p1"] == {"categories": ["year_filter"]}
