"""Tests for the supervisor's ``add_sub_topics`` tool.

Addresses the "supervisor locks strategy on turn 1" finding: the
supervisor must be able to append sub_topics mid-run when worker
results reveal gaps.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from citeclaw.agents.state import (
    AgentConfig,
    SearchStrategy,
    StructuralPriors,
    SubTopicSpec,
    SupervisorState,
)
from citeclaw.agents.tool_dispatch import SupervisorDispatcher, is_error


def _build_supervisor_with_tools():
    """Construct a supervisor dispatcher with tools registered.

    The tool registration logic lives inline in ``_register_supervisor_tools``
    and closes over worker-dispatch dependencies we don't need for these
    unit tests — so we re-derive just the set_strategy + add_sub_topics
    handlers by importing the private helper.
    """
    state = SupervisorState()
    dispatcher = SupervisorDispatcher(
        supervisor_state=state,
        agent_config=AgentConfig(),
        ctx=MagicMock(),
    )
    # Minimal stubs for dispatch + logging — add_sub_topics doesn't use them.
    from citeclaw.agents.supervisor import _register_supervisor_tools
    _register_supervisor_tools(
        dispatcher,
        shared_store=MagicMock(),
        topic_description="",
        filter_summary="",
        seed_papers=[],
        llm_client=MagicMock(),
        ctx=MagicMock(),
        agent_config=AgentConfig(),
        logger=MagicMock(),
    )
    return dispatcher, state


def _set_strategy(dispatcher, sub_topics):
    return dispatcher.dispatch("set_strategy", {
        "structural_priors": {},
        "sub_topics": sub_topics,
    })


def test_add_sub_topics_requires_existing_strategy():
    """Calling add_sub_topics before set_strategy rejects with a
    teaching hint — supervisor can't skip the initial decomposition."""
    dispatcher, state = _build_supervisor_with_tools()
    res = dispatcher.dispatch("add_sub_topics", {
        "sub_topics": [{"id": "x", "description": "x"}],
    })
    assert is_error(res)
    assert "set_strategy first" in res["error"]


def test_add_sub_topics_appends_to_existing_strategy():
    """Happy path: after set_strategy, add_sub_topics grows the list
    without disturbing existing entries."""
    dispatcher, state = _build_supervisor_with_tools()
    _set_strategy(dispatcher, [
        {"id": "a", "description": "A"},
        {"id": "b", "description": "B"},
    ])
    assert len(state.strategy.sub_topics) == 2
    res = dispatcher.dispatch("add_sub_topics", {
        "sub_topics": [
            {"id": "c", "description": "C", "initial_query_sketch": "\"c\""},
            {"id": "d", "description": "D"},
        ],
    })
    assert not is_error(res)
    assert res["added"] == ["c", "d"]
    assert res["n_sub_topics_total"] == 4
    assert [s.id for s in state.strategy.sub_topics] == ["a", "b", "c", "d"]


def test_add_sub_topics_rejects_duplicate_id():
    """Cannot overwrite an already-registered spec — the existing
    spec may already have worker results attached."""
    dispatcher, state = _build_supervisor_with_tools()
    _set_strategy(dispatcher, [{"id": "a", "description": "A"}])
    res = dispatcher.dispatch("add_sub_topics", {
        "sub_topics": [{"id": "a", "description": "different"}],
    })
    assert is_error(res)
    assert "already in strategy" in res["error"]
    # Strategy unchanged.
    assert len(state.strategy.sub_topics) == 1


def test_add_sub_topics_enforces_20_cap_across_combined():
    """Combined initial + added specs must stay within the global 20
    ceiling — otherwise the supervisor can pad the strategy indefinitely."""
    dispatcher, state = _build_supervisor_with_tools()
    # Start with 15 specs.
    initial = [{"id": f"s{i}", "description": f"spec {i}"} for i in range(15)]
    _set_strategy(dispatcher, initial)
    # Adding 6 more would push to 21 → reject.
    overflow = [{"id": f"t{i}", "description": f"t{i}"} for i in range(6)]
    res = dispatcher.dispatch("add_sub_topics", {"sub_topics": overflow})
    assert is_error(res)
    assert "would have 21 sub-topics" in res["error"]
    assert len(state.strategy.sub_topics) == 15  # unchanged


def test_add_sub_topics_requires_non_empty_list():
    """Empty sub_topics list is a no-op error, not a silent pass."""
    dispatcher, state = _build_supervisor_with_tools()
    _set_strategy(dispatcher, [{"id": "a", "description": "A"}])
    res = dispatcher.dispatch("add_sub_topics", {"sub_topics": []})
    assert is_error(res)
    assert "non-empty" in res["error"]


def test_add_sub_topics_requires_id_on_each_entry():
    """Every appended spec needs a non-empty id slug."""
    dispatcher, state = _build_supervisor_with_tools()
    _set_strategy(dispatcher, [{"id": "a", "description": "A"}])
    res = dispatcher.dispatch("add_sub_topics", {
        "sub_topics": [{"description": "missing id"}],
    })
    assert is_error(res)
    assert "id missing" in res["error"]
