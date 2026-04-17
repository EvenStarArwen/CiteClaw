"""Unit tests for the ``query_diagnostics`` worker tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from citeclaw.agents.search_tools import _handle_query_diagnostics, register_worker_tools
from citeclaw.agents.state import AgentConfig, StructuralPriors, WorkerState
from citeclaw.agents.tool_dispatch import DispatcherError, WorkerDispatcher, is_error


class _RecordingS2:
    """Minimal S2 stand-in that counts invocations of ``search_bulk`` and
    returns a canned ``total`` per exact query string. Unknown queries
    return ``total=0``.
    """

    def __init__(self, totals: dict[str, int]):
        self._totals = totals
        self.calls: list[str] = []

    def search_bulk(self, query, *, filters=None, sort=None, limit=1, **_):
        self.calls.append(query)
        return {
            "data": [],
            "total": self._totals.get(query, 0),
            "token": None,
        }


def _make_dispatcher(s2, priors=None):
    state = WorkerState(
        sub_topic_id="sub1",
        structural_priors=priors or StructuralPriors(),
    )
    ctx = MagicMock()
    ctx.s2 = s2
    return WorkerDispatcher(
        worker_state=state,
        dataframe_store=MagicMock(),
        agent_config=AgentConfig(),
        ctx=ctx,
        worker_id="w1",
    )


def test_flat_or_breakdown():
    """For ``"A" | "B" | "C"``, tool probes 1 full-query + 3 leaves
    in-context + 3 leaves alone = 7 S2 calls. Returns per-leaf
    in-context counts, raw counts, and a flat raw_breakdown map.

    For a flat OR, the substituted-query for each leaf IS the leaf
    alone, so in-context == raw.
    """
    totals = {
        '"A" | "B" | "C"': 500,
        '"A"': 300,
        '"B"': 100,
        '"C"': 50,
    }
    s2 = _RecordingS2(totals)
    d = _make_dispatcher(s2)
    result = _handle_query_diagnostics({"query": '"A" | "B" | "C"'}, d)

    assert not is_error(result)
    assert result["total_full_query"] == 500
    assert result["n_leaves_probed"] == 3
    assert len(result["or_groups"]) == 1
    leaves = result["or_groups"][0]["leaves"]
    # Sorted descending by total_in_context.
    assert [l["leaf"] for l in leaves] == ['"A"', '"B"', '"C"']
    assert [l["total_in_context"] for l in leaves] == [300, 100, 50]
    # New: raw_breakdown flat map {leaf: total_alone}.
    assert result["raw_breakdown"] == {'"A"': 300, '"B"': 100, '"C"': 50}
    # 1 full + 3 in-context + 3 raw = 7 S2 calls.
    assert len(s2.calls) == 7


def test_nested_and_or_returns_two_groups():
    """Nested ``("A" | "B") +("C" | "D")`` → 2 OR groups, 4 leaves.

    Here raw counts differ from in-context counts: raw is the leaf
    alone (e.g. ``"A"`` matches 300), in-context is the leaf inside
    the full query (``"A" + ("C" | "D")`` matches 40). Both surfaced.
    """
    totals = {
        '("A" | "B") +("C" | "D")': 50,
        '"A" +("C" | "D")': 40,
        '"B" +("C" | "D")': 10,
        '("A" | "B") +"C"': 30,
        '("A" | "B") +"D"': 20,
        '"A"': 300,  # raw
        '"B"': 200,
        '"C"': 150,
        '"D"': 100,
    }
    s2 = _RecordingS2(totals)
    d = _make_dispatcher(s2)
    result = _handle_query_diagnostics(
        {"query": '("A" | "B") +("C" | "D")'}, d,
    )
    assert not is_error(result)
    assert len(result["or_groups"]) == 2
    group0 = result["or_groups"][0]["leaves"]
    group1 = result["or_groups"][1]["leaves"]
    assert {l["leaf"] for l in group0} == {'"A"', '"B"'}
    assert {l["leaf"] for l in group1} == {'"C"', '"D"'}
    # Raw breakdown surfaces intrinsic breadth.
    assert result["raw_breakdown"] == {
        '"A"': 300, '"B"': 200, '"C"': 150, '"D"': 100,
    }
    # Each leaf record carries both counts.
    a_entry = next(l for l in group0 if l["leaf"] == '"A"')
    assert a_entry["total_in_context"] == 40
    assert a_entry["total_raw"] == 300
    # 1 full + 4 in-context + 4 raw = 9 S2 calls.
    assert len(s2.calls) == 9


def test_no_or_returns_trivial_note():
    """A query with no OR operators returns early with a note and no
    per-leaf breakdown — no wasted S2 calls.
    """
    s2 = _RecordingS2({'"A" +"B"': 7})
    d = _make_dispatcher(s2)
    result = _handle_query_diagnostics({"query": '"A" +"B"'}, d)
    assert not is_error(result)
    assert result["total_full_query"] == 7
    assert result["or_groups"] == []
    assert "no OR" in result["note"]
    # Only the full-query probe.
    assert len(s2.calls) == 1


def test_invalid_query_returns_error_envelope_via_dispatcher():
    """Lint failures surface as structured error envelopes (no S2 calls).

    Must go through the real dispatcher so the handler's
    :class:`DispatcherError` is converted to the ``{error, hint}``
    envelope the LLM actually sees — direct handler calls raise.
    """
    s2 = _RecordingS2({})
    d = _make_dispatcher(s2)
    register_worker_tools(d)
    result = d.dispatch("query_diagnostics", {"query": '"unterminated'})
    assert is_error(result)
    # Zero S2 calls — rejected before any network work.
    assert s2.calls == []


def test_missing_query_rejects():
    """Empty / missing query raises ``DispatcherError`` (converted to
    envelope by the dispatcher in production)."""
    s2 = _RecordingS2({})
    d = _make_dispatcher(s2)
    with pytest.raises(DispatcherError):
        _handle_query_diagnostics({}, d)
    assert s2.calls == []


def test_cap_on_too_many_or_leaves():
    """Queries whose OR-leaf count would exceed the diagnostics cap
    return an early note instead of spending S2 budget on all branches.
    """
    # 13-leaf flat OR — just over the cap (12).
    leaves = [f'"leaf{i}"' for i in range(13)]
    query = " | ".join(leaves)
    s2 = _RecordingS2({query: 500})
    d = _make_dispatcher(s2)
    result = _handle_query_diagnostics({"query": query}, d)
    assert not is_error(result)
    assert result["or_groups"] == []
    assert "cap" in result["note"]
    # Only the full-query probe — per-leaf loop never starts.
    assert len(s2.calls) == 1
