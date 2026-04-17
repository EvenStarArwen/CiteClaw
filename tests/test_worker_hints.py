"""Tests for :func:`_compute_relevant_hints` — the per-turn
situational guidance that replaces the static rules table."""

from __future__ import annotations

import pytest

from citeclaw.agents.state import AgentConfig, QueryMeta, WorkerState
from citeclaw.agents.worker import _compute_relevant_hints


def _state(**kw):
    return WorkerState(sub_topic_id="t", **kw)


def test_no_hints_when_no_prior_tool():
    hints = _compute_relevant_hints("", None, _state(), AgentConfig())
    assert hints == []


def test_hint_when_total_zero():
    hints = _compute_relevant_hints(
        "check_query_size", {"total": 0}, _state(), AgentConfig(),
    )
    assert len(hints) == 1
    assert "over-constrained" in hints[0]
    assert "STAY" in hints[0]


def test_hint_when_total_very_thin():
    hints = _compute_relevant_hints(
        "check_query_size", {"total": 5}, _state(), AgentConfig(),
    )
    assert "very thin" in hints[0] or "Broaden" in hints[0]


def test_hint_when_total_in_sweet_spot():
    hints = _compute_relevant_hints(
        "check_query_size", {"total": 500}, _state(), AgentConfig(),
    )
    assert any("sweet spot" in h for h in hints)


def test_hint_when_total_exceeds_cap():
    cfg = AgentConfig(fetch_total_cap=50_000)
    hints = _compute_relevant_hints(
        "check_query_size", {"total": 80_000}, _state(), cfg,
    )
    assert any("exceeds the fetch cap" in h for h in hints)
    assert any("structural filter" in h for h in hints)


def test_hint_when_total_large_but_under_cap():
    hints = _compute_relevant_hints(
        "check_query_size", {"total": 10_000}, _state(), AgentConfig(),
    )
    assert any("Narrow with a structural filter" in h for h in hints)


def test_query_cap_hit_surfaces_hint():
    cfg = AgentConfig(max_queries_per_worker=3)
    st = _state()
    for i in range(3):
        st.queries[f"fp{i}"] = QueryMeta(
            fingerprint=f"fp{i}", query=f"q{i}", filters={},
        )
    hints = _compute_relevant_hints("", None, st, cfg)
    assert any("cap\n    reached" in h or "cap reached" in h for h in hints)


def test_last_query_slot_warning():
    cfg = AgentConfig(max_queries_per_worker=4)
    st = _state()
    for i in range(3):
        st.queries[f"fp{i}"] = QueryMeta(
            fingerprint=f"fp{i}", query=f"q{i}", filters={},
        )
    hints = _compute_relevant_hints("", None, st, cfg)
    assert any("one slot left" in h for h in hints)


def test_pending_miss_hint():
    """When the auto-verifier flags misses, the hint prompts the
    agent to diagnose them before closing."""
    st = _state()
    st.pending_miss_titles = ["Ref paper 1", "Ref paper 2"]
    # 2 misses, 0 diagnosed → 2 pending
    hints = _compute_relevant_hints("fetch_results", {}, st, AgentConfig())
    assert any("2 auto-detected reference miss" in h for h in hints)
    # Diagnose one — 1 pending remains.
    st.miss_diagnoses.append({"target_title": "Ref paper 1"})
    hints = _compute_relevant_hints("fetch_results", {}, st, AgentConfig())
    assert any("1 auto-detected reference miss" in h for h in hints)


def test_non_size_check_tool_does_not_produce_size_hint():
    """Only ``check_query_size`` results trigger size-band hints."""
    hints = _compute_relevant_hints(
        "fetch_results", {"n_fetched": 300}, _state(), AgentConfig(),
    )
    assert not any("total=" in h for h in hints)
