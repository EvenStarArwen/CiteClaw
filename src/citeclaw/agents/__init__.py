"""LLM-driven search agents (v2).

Houses the supervisor/worker two-level agent that powers
``ExpandBySearch``. The v1 single-loop ``run_iterative_search`` module
has been removed — recover from git history if needed.

Re-exports the core dataclasses so callers can write
``from citeclaw.agents import AgentConfig`` without knowing the
submodule path.
"""

from __future__ import annotations

from citeclaw.agents.state import (
    AgentConfig,
    AngleState,
    QueryAngleResult,
    SearchStrategy,
    StructuralPriors,
    SubTopicResult,
    SubTopicSpec,
    SupervisorState,
    WorkerState,
)
from citeclaw.agents.supervisor import run_supervisor
from citeclaw.agents.worker import run_sub_topic_worker

__all__ = [
    "AgentConfig",
    "AngleState",
    "QueryAngleResult",
    "SearchStrategy",
    "StructuralPriors",
    "SubTopicResult",
    "SubTopicSpec",
    "SupervisorState",
    "WorkerState",
    "run_supervisor",
    "run_sub_topic_worker",
]
