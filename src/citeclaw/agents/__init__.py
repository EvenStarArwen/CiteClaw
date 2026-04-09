"""LLM-driven search agents.

Currently houses :mod:`citeclaw.agents.iterative_search` — the meta-LLM
agent that designs targeted literature-database queries from a topic
description and a sample of papers already in the collection, then
iteratively refines its query based on what each search returned. The
agent powers Phase C's ``ExpandBySearch`` step.

Re-exports the three core dataclasses so callers can write
``from citeclaw.agents import AgentConfig`` instead of having to know
the submodule path.
"""

from __future__ import annotations

from citeclaw.agents.iterative_search import (
    AgentConfig,
    AgentTurn,
    SearchAgentResult,
)

__all__ = ["AgentConfig", "AgentTurn", "SearchAgentResult"]
