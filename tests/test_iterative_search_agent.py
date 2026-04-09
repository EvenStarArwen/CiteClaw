"""Tests for the iterative meta-LLM search agent (Phase B / PB-05).

Drives :func:`citeclaw.agents.iterative_search.run_iterative_search`
end-to-end with the deterministic offline pieces:

- :class:`StubClient` returns the canned three-state lifecycle
  responses (``initial`` → ``refine`` → ``satisfied``) keyed by the
  count of ``"query":`` substrings in the user prompt — that branch was
  added in PB-02.
- :class:`_BudgetAwareFakeS2` is a thin in-test subclass of
  :class:`FakeS2Client` that bumps :class:`BudgetTracker` per
  ``search_bulk`` call, mirroring the real client's bookkeeping. The
  base fake is intentionally side-effect-free, but PB-05's spec asserts
  on ``budget._s2_api["search"]``, so this wrapper bridges the gap
  without leaking budget bookkeeping into the shared fake.

Phase B is DONE when this file is green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from citeclaw.agents.iterative_search import (
    AgentConfig,
    AgentTurn,
    SearchAgentResult,
    run_iterative_search,
)
from citeclaw.cache import Cache
from citeclaw.clients.llm.base import LLMResponse
from citeclaw.clients.llm.stub import StubClient
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.models import PaperRecord
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _BudgetAwareFakeS2(FakeS2Client):
    """In-test subclass that bumps ``budget.record_s2`` per call.

    The base ``FakeS2Client`` deliberately has no budget reference so
    it stays a pure data fake — but PB-05's contract asserts on the
    ``_s2_api["search"]`` counter, so this wrapper records each search
    in the budget tracker exactly the way the real client would.
    """

    def __init__(self, budget: BudgetTracker) -> None:
        super().__init__()
        self._tracked_budget = budget

    def search_bulk(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._tracked_budget.record_s2("search")
        return super().search_bulk(*args, **kwargs)


def _stub_canned_papers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two distinct result sets for the two queries the stub emits.

    Returned as ``(papers_for_initial_query, papers_for_narrowed_query)``
    so the agent's cumulative-hit dedup logic gets exercised across
    iterations (the third iteration re-runs the narrowed query and
    must NOT add duplicates).
    """
    initial = [
        make_paper("p1", title="First Paper", venue="Nature", year=2020),
        make_paper("p2", title="Second Paper", venue="Science", year=2021),
        make_paper("p3", title="Third Paper", venue="Cell", year=2022),
    ]
    narrowed = [
        make_paper("p4", title="Fourth Paper", venue="Cell", year=2022),
        make_paper("p5", title="Fifth Paper", venue="Nature", year=2023),
    ]
    return initial, narrowed


@pytest.fixture
def budget() -> BudgetTracker:
    return BudgetTracker()


@pytest.fixture
def fake_s2(budget: BudgetTracker) -> _BudgetAwareFakeS2:
    fs = _BudgetAwareFakeS2(budget)
    initial, narrowed = _stub_canned_papers()
    fs.register_search_bulk("test topic", initial)
    fs.register_search_bulk("test topic narrowed", narrowed)
    # Also seed the corpus so enrich_batch can hydrate them for the
    # per-turn observation summary (unique_venues / year_range / titles).
    for p in initial + narrowed:
        fs.add(p)
    return fs


@pytest.fixture
def ctx(
    tmp_path: Path,
    budget: BudgetTracker,
    fake_s2: _BudgetAwareFakeS2,
) -> Context:
    cfg = Settings(data_dir=tmp_path, screening_model="stub")
    cache = Cache(tmp_path / "cache.db")
    return Context(config=cfg, s2=fake_s2, cache=cache, budget=budget)


@pytest.fixture
def llm(ctx: Context) -> StubClient:
    return StubClient(ctx.config, ctx.budget)


def _five_anchors() -> list[PaperRecord]:
    return [
        PaperRecord(
            paper_id=f"SEED{i}",
            title=f"Anchor Paper {i}",
            year=2018 + i,
            venue="Nature",
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Spec assertions
# ---------------------------------------------------------------------------


class TestRunIterativeSearch:
    def test_max_iterations_3_satisfies_with_5_anchors(
        self, ctx: Context, llm: StubClient,
    ):
        """Three rounds with the stub: initial → refine → satisfied.

        Stops on the third iteration's ``satisfied`` decision rather
        than at the loop bound.
        """
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "protein folding", _five_anchors(), llm, ctx, config,
        )
        assert isinstance(result, SearchAgentResult)
        assert len(result.transcript) == 3
        assert result.final_decision == "satisfied"

    def test_max_iterations_1_single_shot(
        self, ctx: Context, llm: StubClient,
    ):
        """Single-iteration runs are valid; the loop terminates via
        ``max_iterations`` because the stub returns ``initial`` (which
        is not a break-state)."""
        config = AgentConfig(max_iterations=1)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert len(result.transcript) == 1
        assert result.final_decision == "max_iterations"

    def test_default_agent_config_has_max_iterations_4(self):
        """Architectural decision: default outer loop is 4 iterations."""
        assert AgentConfig().max_iterations == 4

    def test_empty_anchor_papers_topic_only_fallback(
        self, ctx: Context, llm: StubClient,
    ):
        """Bootstrap mode: the agent must run without anchors,
        rendering the canonical fallback message in place of the
        anchor block."""
        config = AgentConfig(max_iterations=2)
        result = run_iterative_search(
            "topic with no anchors at all", [], llm, ctx, config,
        )
        assert len(result.transcript) >= 1
        # Even without anchors, the LLM scratchpad must round-trip.
        assert all(turn.thinking for turn in result.transcript)

    def test_every_thinking_field_is_non_empty(
        self, ctx: Context, llm: StubClient,
    ):
        """Proves the schema's leading ``thinking`` field round-trips
        all the way from the stub through the JSON parser into the
        ``AgentTurn`` dataclass for every iteration."""
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert all(isinstance(t.thinking, str) and t.thinking
                   for t in result.transcript)
        # The stub's three lifecycle states each emit a distinct
        # thinking string; assert the sequence transitions correctly
        # so a future regression in the iteration counter would
        # surface here as well as in test_llm.py.
        assert result.transcript[0].thinking == "stub: initial exploration"
        assert result.transcript[1].thinking == (
            "stub: prior was too broad, narrowing"
        )
        assert result.transcript[2].thinking == "stub: results saturated"

    def test_iteration_n_plus_1_user_prompt_contains_iteration_n_thinking(
        self, ctx: Context, llm: StubClient,
    ):
        """Level-1 transcript accumulation: when iteration N+1 calls
        the LLM, the user prompt MUST embed iteration N's ``thinking``
        text so the agent can build on its earlier reasoning. We
        intercept ``llm.call`` to capture every user prompt the agent
        constructs and assert each prior thinking string survives into
        the next prompt verbatim.
        """
        captured_users: list[str] = []
        original_call = llm.call

        def spy_call(
            system: str,
            user: str,
            *,
            with_logprobs: bool = False,
            category: str = "other",
            response_schema: dict[str, Any] | None = None,
        ) -> LLMResponse:
            captured_users.append(user)
            return original_call(
                system,
                user,
                with_logprobs=with_logprobs,
                category=category,
                response_schema=response_schema,
            )

        llm.call = spy_call  # type: ignore[method-assign]

        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )

        assert len(captured_users) == 3
        # Iteration 1: no prior turns, so iteration 1's own thinking
        # text MUST NOT yet appear in the very first user prompt.
        assert "stub: initial exploration" not in captured_users[0]
        # Iteration 2: must contain iteration 1's thinking.
        assert "stub: initial exploration" in captured_users[1]
        # Iteration 3: must contain iteration 2's thinking. (And by
        # extension iteration 1's, since the transcript accumulates.)
        assert "stub: prior was too broad, narrowing" in captured_users[2]
        assert "stub: initial exploration" in captured_users[2]
        # Sanity: the loop did break on satisfied.
        assert result.final_decision == "satisfied"

    def test_budget_meta_search_agent_tokens_recorded(
        self, ctx: Context, llm: StubClient,
    ):
        """The agent must spend its LLM tokens under the
        ``meta_search_agent`` category so cost-tracking dashboards can
        attribute the spend correctly."""
        config = AgentConfig(max_iterations=2)
        run_iterative_search("topic", _five_anchors(), llm, ctx, config)
        assert ctx.budget._llm_tokens.get("meta_search_agent", 0) > 0

    def test_budget_s2_api_search_count_equals_iterations(
        self, ctx: Context, llm: StubClient,
    ):
        """One ``search_bulk`` call per iteration. Holds across all
        loop bounds: when the lifecycle stops short (max_iterations=5
        but the stub satisfies at iteration 3), the count tracks the
        actual number of turns, not the configured cap."""
        for max_n in (1, 2, 3, 5):
            ctx.budget._s2_api.clear()
            ctx.budget._llm_tokens.clear()
            config = AgentConfig(max_iterations=max_n)
            result = run_iterative_search(
                "topic", _five_anchors(), llm, ctx, config,
            )
            actual_iterations = len(result.transcript)
            assert ctx.budget._s2_api.get("search", 0) == actual_iterations, (
                f"max_iterations={max_n}: expected "
                f"{actual_iterations} search calls, got "
                f"{ctx.budget._s2_api.get('search', 0)}"
            )


# ---------------------------------------------------------------------------
# A few extra integration assertions to lock in PB-04's behavior beyond the
# strict spec list — these would catch a regression that the per-spec tests
# above might miss but PC-01 (ExpandBySearch) will care about.
# ---------------------------------------------------------------------------


class TestRunIterativeSearchIntegration:
    def test_cumulative_hits_dedup_across_iterations(
        self, ctx: Context, llm: StubClient,
    ):
        """Iterations 2 and 3 of the stub re-issue the same narrowed
        query. The cumulative hit set must NOT contain duplicates."""
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        seen_ids = [h.get("paperId") for h in result.hits]
        assert len(seen_ids) == len(set(seen_ids))
        # 3 papers from iteration 1 + 2 unique papers from iteration 2
        # = 5 unique. Iteration 3 re-fetches the same narrowed set and
        # adds nothing new.
        assert len(result.hits) == 5
        assert set(seen_ids) == {"p1", "p2", "p3", "p4", "p5"}

    def test_search_agent_result_fields_populated(
        self, ctx: Context, llm: StubClient,
    ):
        """All five SearchAgentResult fields are non-default after a
        successful run."""
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert len(result.hits) > 0
        assert len(result.transcript) > 0
        assert result.final_decision == "satisfied"
        assert result.tokens_used > 0
        assert result.s2_requests_used == 3

    def test_agent_turn_observation_summary_populated(
        self, ctx: Context, llm: StubClient,
    ):
        """When the fake's corpus contains the search-result papers,
        the per-turn observation summary should reflect their venues
        and year range — proving the enrich_batch round-trip works."""
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        first_turn = result.transcript[0]
        assert first_turn.n_results == 3  # initial query returned p1/p2/p3
        assert isinstance(first_turn.unique_venues, list)
        assert set(first_turn.unique_venues) == {"Nature", "Science", "Cell"}
        ymin, ymax = first_turn.year_range
        assert ymin == 2020 and ymax == 2022
        assert first_turn.sample_titles  # non-empty

    def test_n_novel_tracks_dedup_across_turns(
        self, ctx: Context, llm: StubClient,
    ):
        """PH-02: each AgentTurn carries n_novel — the count of NEW results
        this turn that weren't already in the cumulative hit set.

        n_novel is the saturation signal the agent reads in its next-turn
        prompt to decide whether to keep refining or mark satisfied —
        without it, the prompt only saw raw n_results which confused the
        agent (n_results stays high even when nothing new is being added).

        Contract:
          - n_novel is always <= n_results
          - On turn 1 (empty cumulative set), n_novel == n_results (every
            paper is brand new)
          - On later turns, n_novel ∈ [0, n_results] depending on overlap
        """
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert len(result.transcript) >= 2
        # Turn 1 fills the cumulative set from empty → all results are novel.
        assert result.transcript[0].n_novel == result.transcript[0].n_results
        # Every later turn must satisfy 0 <= n_novel <= n_results.
        for t in result.transcript[1:]:
            assert 0 <= t.n_novel <= t.n_results
        # The cumulative hit count must equal the sum of n_novel across turns.
        assert len(result.hits) == sum(t.n_novel for t in result.transcript)

    def test_n_novel_is_rendered_into_next_turn_prompt(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """The transcript renderer must include the NEW count in the
        ``Observed:`` line so the agent's next iteration sees it. We
        verify by spying on the LLM client's call() and checking the
        user prompt sent to iteration 2 contains the literal substring."""
        from citeclaw.clients.llm.base import LLMResponse

        captured_prompts: list[str] = []

        original_call = llm.call

        def spy_call(system, user, **kwargs):
            captured_prompts.append(user)
            return original_call(system, user, **kwargs)

        monkeypatch.setattr(llm, "call", spy_call)

        config = AgentConfig(max_iterations=3)
        run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        # Iteration 1's prompt has no transcript yet, but iteration 2's
        # prompt should embed turn 1's observation line including ``NEW``.
        assert len(captured_prompts) >= 2
        iter2_prompt = captured_prompts[1]
        assert "NEW since previous turns" in iter2_prompt

    def test_query_field_round_trips_into_agent_turn(
        self, ctx: Context, llm: StubClient,
    ):
        """The agent must preserve the LLM's emitted query dict
        verbatim in AgentTurn.query, not just its text field."""
        config = AgentConfig(max_iterations=2)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert isinstance(result.transcript[0].query, dict)
        assert result.transcript[0].query.get("text") == "test topic"
        assert result.transcript[1].query.get("text") == "test topic narrowed"

    def test_decision_and_reasoning_round_trip(
        self, ctx: Context, llm: StubClient,
    ):
        config = AgentConfig(max_iterations=3)
        result = run_iterative_search(
            "topic", _five_anchors(), llm, ctx, config,
        )
        assert [t.decision for t in result.transcript] == [
            "initial", "refine", "satisfied",
        ]
        assert [t.reasoning for t in result.transcript] == [
            "stub initial", "stub refine", "stub satisfied",
        ]
