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

import json
from pathlib import Path
from typing import Any

import pytest

from citeclaw.agents.iterative_search import (
    AgentConfig,
    AgentState,
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

    def test_total_in_corpus_is_captured_from_search_payload(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """PH-03: each AgentTurn carries total_in_corpus — the ``total``
        field S2 returned for this query. When total > n_results the
        renderer must surface a PARTIAL warning so the agent knows it's
        only seeing one page of a larger corpus and should narrow with
        filters rather than synonyms."""
        # Patch the fake S2's search_bulk to return a payload with a
        # synthetic ``total`` larger than the data list to simulate a
        # partial fetch.
        def fake_search_bulk(query, *, filters=None, sort=None, token=None, limit=1000):
            return {
                "data": [
                    {"paperId": "p1", "title": "Page-1 hit alpha"},
                    {"paperId": "p2", "title": "Page-1 hit beta"},
                ],
                "total": 5000,
            }

        monkeypatch.setattr(ctx.s2, "search_bulk", fake_search_bulk)
        config = AgentConfig(max_iterations=2)
        result = run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        first = result.transcript[0]
        assert first.n_results == 2
        assert first.total_in_corpus == 5000

    def test_partial_warning_in_next_turn_prompt(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """When the previous turn's S2 search was partial (n_results <
        total_in_corpus), the next iteration's user prompt must contain
        the PARTIAL warning so the agent knows to narrow with filters."""
        def fake_search_bulk(query, *, filters=None, sort=None, token=None, limit=1000):
            return {"data": [{"paperId": "p1", "title": "alpha"}], "total": 9999}

        monkeypatch.setattr(ctx.s2, "search_bulk", fake_search_bulk)

        captured_prompts: list[str] = []
        original_call = llm.call

        def spy_call(system, user, **kwargs):
            captured_prompts.append(user)
            return original_call(system, user, **kwargs)

        monkeypatch.setattr(llm, "call", spy_call)

        config = AgentConfig(max_iterations=2)
        run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        assert len(captured_prompts) >= 2
        iter2_prompt = captured_prompts[1]
        assert "PARTIAL" in iter2_prompt
        assert "9999" in iter2_prompt  # the total figure should be visible

    def test_saturation_guardrail_stops_at_two_zero_novel_turns(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """PH-08: a deterministic safety net for the prompt's MANDATORY
        rule C. If the last two completed turns BOTH had n_results > 0
        AND n_novel == 0, the next iteration's LLM call should be
        skipped and the loop should break with
        final_decision='saturated_guardrail'.

        Mock the LLM with one that NEVER says satisfied (always returns
        refine), so the guardrail is the only thing that can stop the
        loop. Patch search_bulk to always return the SAME single paper:
        T1 has n_novel=1 (fresh), T2 has n_novel=0 (dupe), T3 has
        n_novel=0 (dupe). Without the guardrail the loop would run all
        max_iterations turns; with it, the loop breaks at the start of
        iter 4 with no LLM call for that iteration."""
        from citeclaw.clients.llm.base import LLMResponse

        def fake_search_bulk(query, *, filters=None, sort=None, token=None, limit=1000):
            return {"data": [{"paperId": "fixed_p1", "title": "dupe paper"}], "total": 1}

        monkeypatch.setattr(ctx.s2, "search_bulk", fake_search_bulk)

        # Mock LLM that always returns refine (never satisfied), so the
        # guardrail is the only termination path.
        llm_call_count = [0]
        def always_refine_call(system, user, *, with_logprobs=False, category="other", response_schema=None):
            llm_call_count[0] += 1
            ctx.budget.record_llm(10, 10, category)
            return LLMResponse(
                text=json.dumps({
                    "evaluate": "stub forced refine",
                    "query": {"text": "test query"},
                    "should_stop": False,
                    "reasoning": "always refine",
                }),
            )

        monkeypatch.setattr(llm, "call", always_refine_call)

        config = AgentConfig(max_iterations=10)
        result = run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        # Guardrail must trigger before the full 10 iters complete.
        assert result.final_decision == "saturated_guardrail"
        # T1 has the only novel paper; T2/T3 are zero. Guardrail fires
        # at the START of iter 4 (sees T2=0 and T3=0), so 3 LLM calls
        # were issued, not 10.
        assert llm_call_count[0] == 3
        assert len(result.transcript) == 3

    def test_guardrail_does_not_fire_when_one_turn_recovers(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """The guardrail must NOT fire if a single zero-novel turn is
        followed by even one new paper. Otherwise legitimate pivots
        that recover would be cut short."""
        # Alternate: first call returns unique p1, second returns same p1
        # (n_novel=0), third returns new p2 (n_novel=1).
        responses = [
            {"data": [{"paperId": "p1"}], "total": 1},
            {"data": [{"paperId": "p1"}], "total": 1},  # 0 new
            {"data": [{"paperId": "p2"}], "total": 1},  # 1 new
            {"data": [{"paperId": "p2"}], "total": 1},  # 0 new
        ]
        call_idx = [0]
        def fake_search_bulk(query, *, filters=None, sort=None, token=None, limit=1000):
            r = responses[min(call_idx[0], len(responses) - 1)]
            call_idx[0] += 1
            return r

        monkeypatch.setattr(ctx.s2, "search_bulk", fake_search_bulk)

        config = AgentConfig(max_iterations=4)
        result = run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        # The 0-new T2 was followed by a 1-new T3, so the rule
        # ("BOTH last two turns == 0") does NOT trigger at iter 4.
        # The loop runs all 4 iterations.
        assert result.final_decision != "saturated_guardrail"

    def test_guardrail_does_not_fire_on_two_broken_query_turns(
        self, ctx: Context, llm: StubClient, monkeypatch,
    ):
        """PH-08: the guardrail must distinguish "saturated" (found the
        same papers twice → n_results > 0 but n_novel == 0) from
        "broken query" (got zero results because of bad syntax →
        n_results == 0). Two consecutive broken-query turns must NOT
        fire the guardrail; the agent needs the chance to fix its
        query and recover.

        Empirically observed during PH-08 testing: the agent used an
        invalid fieldsOfStudy abbreviation and got 0 results for T1+T2.
        Without this carve-out the guardrail killed the run before the
        agent could fix the typo."""
        from citeclaw.clients.llm.base import LLMResponse

        # Always return ZERO results (simulating a broken filter that
        # makes every search return nothing).
        def fake_search_bulk(query, *, filters=None, sort=None, token=None, limit=1000):
            return {"data": [], "total": 0}

        monkeypatch.setattr(ctx.s2, "search_bulk", fake_search_bulk)

        llm_call_count = [0]
        def always_refine_call(system, user, *, with_logprobs=False, category="other", response_schema=None):
            llm_call_count[0] += 1
            ctx.budget.record_llm(10, 10, category)
            return LLMResponse(
                text=json.dumps({
                    "evaluate": "broken query, retrying",
                    "query": {"text": "test query"},
                    "should_stop": False,
                    "reasoning": "retry",
                }),
            )

        monkeypatch.setattr(llm, "call", always_refine_call)

        config = AgentConfig(max_iterations=4)
        result = run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        # All 4 iterations should run — the guardrail must NOT fire
        # because n_results == 0 in every turn (broken query, not
        # saturation). The loop ends via max_iterations, not the
        # guardrail.
        assert result.final_decision != "saturated_guardrail"
        assert llm_call_count[0] == 4
        assert len(result.transcript) == 4

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

    def test_agent_state_dedups_hits_across_turns(self):
        """``add_novel_hits`` only counts + appends previously-unseen ids."""
        state = AgentState()
        n1 = state.add_novel_hits([{"paperId": "a"}, {"paperId": "b"}])
        assert n1 == 2
        assert len(state.cumulative_hits) == 2
        n2 = state.add_novel_hits([{"paperId": "b"}, {"paperId": "c"}])
        assert n2 == 1
        assert len(state.cumulative_hits) == 3
        assert state.seen_ids == {"a", "b", "c"}

    def test_agent_state_skips_entries_with_bad_paper_id(self):
        state = AgentState()
        n = state.add_novel_hits([
            {"paperId": "a"},
            {},                      # missing
            {"paperId": ""},         # empty
            {"paperId": 42},         # non-str
            {"paperId": None},
        ])
        assert n == 1
        assert state.cumulative_hits == [{"paperId": "a"}]

    def _turn(self, n_results: int, n_novel: int) -> AgentTurn:
        return AgentTurn(
            iteration=1, thinking="", query={}, n_results=n_results,
            n_novel=n_novel, total_in_corpus=n_results, unique_venues=[],
            year_range=(None, None), sample_titles=[], decision="refine",
            reasoning="",
        )

    def test_agent_state_not_saturated_with_fewer_than_two_turns(self):
        state = AgentState()
        assert not state.is_saturated()
        state.transcript.append(self._turn(n_results=10, n_novel=0))
        assert not state.is_saturated()

    def test_agent_state_saturated_when_last_two_turns_zero_novel(self):
        state = AgentState()
        state.transcript.append(self._turn(n_results=10, n_novel=0))
        state.transcript.append(self._turn(n_results=20, n_novel=0))
        assert state.is_saturated()

    def test_agent_state_not_saturated_when_results_zero(self):
        """A broken query returning 0 results must not count as saturation."""
        state = AgentState()
        state.transcript.append(self._turn(n_results=0, n_novel=0))
        state.transcript.append(self._turn(n_results=0, n_novel=0))
        assert not state.is_saturated()

    def test_agent_state_not_saturated_when_novel_nonzero(self):
        state = AgentState()
        state.transcript.append(self._turn(n_results=10, n_novel=0))
        state.transcript.append(self._turn(n_results=10, n_novel=3))
        assert not state.is_saturated()

    def test_dashboard_receives_per_turn_notes(
        self, ctx: Context, llm: StubClient,
    ):
        """The agent must surface a one-line note + phase update per
        turn so the live panel doesn't sit empty during long LLM/S2
        round-trips."""

        class _RecordingDash:
            is_null = False

            def __init__(self) -> None:
                self.notes: list[str] = []
                self.phases: list[str] = []

            def begin_phase(self, description: str, total: int) -> None:
                self.phases.append(description)

            def tick_inner(self, n: int = 1) -> None: ...

            def note(self, msg: str) -> None:
                self.notes.append(msg)

        dash = _RecordingDash()
        ctx.dashboard = dash
        config = AgentConfig(max_iterations=3)
        run_iterative_search("topic", _five_anchors(), llm, ctx, config)

        # 3 iterations × 3 phases ("designing query", "S2 bulk search",
        # "summarising hits") = 9 begin_phase calls.
        assert len(dash.phases) == 9
        assert all("designing query" in p for p in dash.phases[::3])
        # 3 iterations × 2 notes ("query=...", "N hits ...") + final
        # ExpandBySearch note is added by the wrapping step, not the
        # agent — so 6 notes from the agent itself.
        assert len(dash.notes) == 6
        assert any('query="test topic"' in n for n in dash.notes)
        assert any("→ satisfied" in n for n in dash.notes)
