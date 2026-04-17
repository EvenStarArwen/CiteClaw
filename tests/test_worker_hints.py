"""Tests for :func:`_compute_relevant_hints` — the per-turn situational
guidance that replaces the v1 static TOTAL-SIZE HEURISTICS table."""

from __future__ import annotations

import pytest

from citeclaw.agents.state import AgentConfig, AngleState, WorkerState
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
    assert "STAY" in hints[0]  # no sub-topic switching


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


def test_angle_cap_hit_surfaces_hint():
    cfg = AgentConfig(max_angles_per_worker=3)
    st = _state()
    for i in range(3):
        st.angles[f"fp{i}"] = AngleState(
            fingerprint=f"fp{i}", query=f"q{i}", filters={},
        )
    hints = _compute_relevant_hints("", None, st, cfg)
    # Cap-hit hint is always surfaced, regardless of prior tool.
    assert any("cap reached" in h for h in hints)


def test_last_angle_slot_warning():
    cfg = AgentConfig(max_angles_per_worker=4)
    st = _state()
    for i in range(3):
        st.angles[f"fp{i}"] = AngleState(
            fingerprint=f"fp{i}", query=f"q{i}", filters={},
        )
    hints = _compute_relevant_hints("", None, st, cfg)
    # "one slot left" warning.
    assert any("one slot left" in h for h in hints)


def test_refinement_cap_hit_hint():
    cfg = AgentConfig(max_refinement_per_angle=1)
    st = _state()
    angle = AngleState(fingerprint="fp", query="q", filters={})
    angle.refinement_count = 1
    st.angles["fp"] = angle
    st.active_fingerprint = "fp"
    hints = _compute_relevant_hints("", None, st, cfg)
    assert any("at cap" in h for h in hints)
    assert any("NEW angle" in h for h in hints)


def test_non_size_check_tool_does_not_produce_size_hint():
    """Only ``check_query_size`` results trigger size-band hints.

    Otherwise every turn would spam stale totals from earlier calls.
    """
    hints = _compute_relevant_hints(
        "inspect_angle", {"n_fetched": 300}, _state(), AgentConfig(),
    )
    # No 'total=' string — other hint categories can still fire but not size-band.
    assert not any("total=" in h for h in hints)
