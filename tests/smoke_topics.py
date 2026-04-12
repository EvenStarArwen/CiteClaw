"""Five research topics with real S2 seed papers for live smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SmokeTestTopic:
    name: str
    description: str
    seed_ids: tuple[str, ...]
    min_expected_collection: int = 2


TOPICS: list[SmokeTestTopic] = [
    SmokeTestTopic(
        name="transformers",
        description="Attention mechanisms and transformer architectures for NLP and beyond",
        seed_ids=("ARXIV:1706.03762",),  # Attention Is All You Need
    ),
    SmokeTestTopic(
        name="crispr",
        description="CRISPR-Cas9 genome editing technology and applications",
        seed_ids=("DOI:10.1126/science.1225829",),  # Jinek et al. 2012
    ),
    SmokeTestTopic(
        name="gnn",
        description="Graph neural networks and graph attention networks",
        seed_ids=("ARXIV:1710.10903",),  # GAT
    ),
    SmokeTestTopic(
        name="rl_robotics",
        description="Reinforcement learning for robotic control and locomotion",
        seed_ids=("ARXIV:1707.06347",),  # PPO
    ),
    SmokeTestTopic(
        name="llm_agents",
        description="Large language model agents with tool use and reasoning",
        seed_ids=("ARXIV:2210.03629",),  # ReAct
    ),
]
