"""Iterative meta-LLM search agent dataclasses.

This module defines the three pure data containers used by the search
agent's loop body (PB-04 will fill in :func:`run_iterative_search`):

- :class:`AgentConfig` — knobs the caller dials in (iteration cap, token
  cap, target collection size, per-iter search width, etc.). Defaults
  match the architectural decisions captured in the roadmap: four
  iterations, 200k LLM tokens, target 200 papers, 500 results per
  S2 search call, 20-paper sample for the per-turn observation summary,
  and ``reasoning_effort="high"`` so capable models stack native
  reasoning tokens on top of the schema's ``thinking`` field.
- :class:`AgentTurn` — what one iteration produced: the LLM's free-text
  scratchpad, the JSON query it emitted, the size and shape of the
  result set it pulled back, and its decision for the next loop step.
  Persists in :attr:`SearchAgentResult.transcript` so the next
  iteration's user prompt (and any post-hoc inspection) can replay the
  full reasoning trail.
- :class:`SearchAgentResult` — the agent's final report: the cumulative
  hit set, the transcript, the lifecycle decision that broke the loop,
  and bookkeeping totals so callers can audit how much budget the run
  consumed.

PB-03 ships only the dataclasses; the loop itself lands in PB-04 and
its tests in PB-05. Keeping the data shapes in their own module lets
PB-05's test file import them without dragging the full loop in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Caller-tunable knobs for one search-agent run.

    All defaults match the values fixed during the design discussion
    (see roadmap "Architectural decisions" item 2). The two-level
    reasoning split is: ``max_iterations`` controls the outer loop
    where each turn sees the prior transcript, and
    ``reasoning_effort`` is forwarded to the underlying LLM client so
    capable models layer native chain-of-thought tokens on top of the
    response schema's ``thinking`` field.
    """

    max_iterations: int = 4
    max_llm_tokens: int = 200_000
    target_count: int = 200
    search_limit_per_iter: int = 500
    summarize_sample: int = 20
    model: str | None = None
    reasoning_effort: str | None = "high"


@dataclass
class AgentTurn:
    """One pass through the agent's outer loop.

    Created after each LLM call + S2 search round-trip and appended to
    :attr:`SearchAgentResult.transcript`. The fields after ``thinking``
    summarise the result set the agent observed *for this turn*, so
    PB-04's transcript renderer can show the agent's later self what
    each prior query actually returned (year range, unique venues,
    sample titles) without re-quoting the full hit list.
    """

    iteration: int
    thinking: str
    query: dict
    n_results: int
    unique_venues: list[str]
    year_range: tuple[int | None, int | None]
    sample_titles: list[str]
    decision: str
    reasoning: str


@dataclass
class SearchAgentResult:
    """Final report from one ``run_iterative_search`` call.

    ``hits`` is the cumulative dedup'd hit set across all iterations;
    ``transcript`` preserves every turn in iteration order;
    ``final_decision`` records which lifecycle state broke the loop
    (``satisfied`` / ``abort`` / ``max_iterations`` / ``budget``);
    ``tokens_used`` and ``s2_requests_used`` are the deltas this run
    drew from the shared :class:`citeclaw.config.BudgetTracker`.
    """

    hits: list[dict] = field(default_factory=list)
    transcript: list[AgentTurn] = field(default_factory=list)
    final_decision: str = ""
    tokens_used: int = 0
    s2_requests_used: int = 0
