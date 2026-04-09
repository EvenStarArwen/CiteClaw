"""Tests for :mod:`citeclaw.clients.llm.stub` and :mod:`citeclaw.screening.llm_runner`.

Critically, nothing in this file touches a real LLM provider. All LLM behavior
is routed through :class:`StubClient` — the offline deterministic stub built
into CiteClaw for exactly this use case.
"""

from __future__ import annotations

import json

import pytest

from citeclaw.clients.llm.base import LLMResponse
from citeclaw.clients.llm.factory import build_llm_client, is_stub, supports_logprobs
from citeclaw.clients.llm.stub import StubClient, stub_respond
from citeclaw.config import BudgetTracker, Settings
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.models import BudgetExhaustedError, PaperRecord
from citeclaw.screening import llm_runner
from citeclaw.screening.llm_runner import (
    _client_for,
    _parse,
    _parse_matches,
    _run_one_batch,
    dispatch_batch,
)


# ---------------------------------------------------------------------------
# StubClient
# ---------------------------------------------------------------------------


class TestStubRespond:
    def test_score_template(self):
        text = stub_respond("", 'Return exactly 3 objects with "score"')
        data = json.loads(text)
        assert len(data) == 3
        assert all(d["score"] == 3 for d in data)

    def test_reject_template(self):
        text = stub_respond("", 'Return exactly 2 objects with "reject"')
        data = json.loads(text)
        assert len(data) == 2
        assert all(d["reject"] is False for d in data)

    def test_match_template(self):
        text = stub_respond(
            "",
            'Return exactly 2 objects with "match"\n[{"index":1,"match":true}]',
        )
        data = json.loads(text)
        # Stub now emits the structured-output wrapped shape:
        # ``{"results": [{"index": 1, "match": true}, ...]}``
        assert isinstance(data, dict)
        assert "results" in data
        assert len(data["results"]) == 2
        assert all(d["match"] is True for d in data["results"])

    def test_numbered_list_fallback(self):
        """When no explicit count is declared, we count ``N. ...`` bullets."""
        text = stub_respond(
            "",
            'response should be "match"\n1. a\n2. b\n3. c',
        )
        data = json.loads(text)
        assert len(data["results"]) == 3

    def test_label_prompt(self):
        text = stub_respond("", "Title: The Transformer Paper\nLabel:")
        assert text == "The Transformer"

    def test_default_empty_array(self):
        assert stub_respond("", "something else entirely") == "[]"


class TestStubClient:
    def test_call_records_budget(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, screening_model="stub")
        budget = BudgetTracker()
        client = StubClient(cfg, budget)
        resp = client.call("sys", 'user with "match" field\n1. A', category="title")
        assert isinstance(resp, LLMResponse)
        assert budget.llm_calls == 1
        assert budget.llm_total_tokens > 0

    def test_call_raises_on_exhausted_budget(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, screening_model="stub", max_llm_tokens=1)
        budget = BudgetTracker()
        budget.record_llm(10, 10)  # already exhausted
        client = StubClient(cfg, budget)
        with pytest.raises(BudgetExhaustedError):
            client.call("sys", "usr")


class TestLLMClientFactory:
    def test_is_stub(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, screening_model="stub")
        assert is_stub(cfg) is True

    def test_build_stub_client(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, screening_model="stub")
        client = build_llm_client(cfg, BudgetTracker())
        assert isinstance(client, StubClient)

    def test_supports_logprobs_false_for_stub(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, screening_model="stub")
        client = build_llm_client(cfg, BudgetTracker())
        assert supports_logprobs(client) is False


# ---------------------------------------------------------------------------
# llm_runner._parse and _parse_matches
# ---------------------------------------------------------------------------


class TestParse:
    def test_parse_plain_json(self):
        assert _parse('[{"a": 1}]') == [{"a": 1}]

    def test_parse_fenced_json(self):
        s = "```json\n[{\"a\": 1}]\n```"
        assert _parse(s) == [{"a": 1}]

    def test_parse_unfenced(self):
        assert _parse("```\n[1,2,3]\n```") == [1, 2, 3]


class TestParseMatches:
    def test_normal_case(self):
        raw = '[{"index": 1, "match": true}, {"index": 2, "match": false}]'
        assert _parse_matches(raw, 2) == [True, False]

    def test_wrapped_structured_output(self):
        """Structured output from OpenAI/Gemini comes back as
        ``{"results": [...]}``. Parser must accept both shapes."""
        raw = '{"results": [{"index": 1, "match": true}, {"index": 2, "match": false}]}'
        assert _parse_matches(raw, 2) == [True, False]

    def test_wrapped_structured_output_partial(self):
        raw = '{"results": [{"index": 2, "match": true}]}'
        assert _parse_matches(raw, 3) == [False, True, False]

    def test_out_of_range_index_ignored(self):
        raw = '[{"index": 99, "match": true}]'
        assert _parse_matches(raw, 2) == [False, False]

    def test_fills_missing_with_false(self):
        raw = '[{"index": 1, "match": true}]'
        assert _parse_matches(raw, 3) == [True, False, False]

    def test_non_dict_entries_ignored(self):
        raw = '[1, {"index": 1, "match": true}, null]'
        assert _parse_matches(raw, 1) == [True]

    def test_bad_json_returns_none(self):
        assert _parse_matches("not json", 3) is None

    def test_non_list_returns_none(self):
        """A bare object without ``results`` is not parseable."""
        assert _parse_matches('{"foo": 1}', 3) is None


# ---------------------------------------------------------------------------
# dispatch_batch — title / title_abstract / venue scopes
# ---------------------------------------------------------------------------


class TestDispatchBatch:
    def _papers(self, n, venue=None):
        return [
            PaperRecord(paper_id=f"p{i}", title=f"Title {i}",
                        abstract=f"Abstract {i}", venue=venue)
            for i in range(n)
        ]

    def test_empty_returns_empty(self, ctx):
        lf = LLMFilter(scope="title", prompt="x")
        assert dispatch_batch([], lf, ctx) == {}

    def test_title_scope_stub_all_true(self, ctx):
        lf = LLMFilter(scope="title", prompt="is relevant")
        out = dispatch_batch(self._papers(5), lf, ctx)
        assert set(out.keys()) == {f"p{i}" for i in range(5)}
        assert all(out.values())

    def test_title_abstract_scope(self, ctx):
        lf = LLMFilter(scope="title_abstract", prompt="is novel")
        out = dispatch_batch(self._papers(3), lf, ctx)
        assert all(out.values())

    def test_batching_respects_batch_size(self, ctx, monkeypatch):
        """With batch_size=2 and 5 papers, the runner should dispatch 3 batches
        (2+2+1). Each batch goes through _run_one_batch exactly once."""
        calls: list[int] = []
        original = _run_one_batch

        def spy(client, lf, contents, ids):
            calls.append(len(contents))
            return original(client, lf, contents, ids)

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        ctx.config.llm_batch_size = 2
        ctx.config.llm_concurrency = 1
        lf = LLMFilter(scope="title", prompt="x")
        out = dispatch_batch(self._papers(5), lf, ctx)
        assert sum(calls) == 5
        assert sorted(calls) == [1, 2, 2]
        assert len(out) == 5

    def test_venue_scope_dedups(self, ctx, monkeypatch):
        """Venue scope should dedup identical venue strings before calling the
        LLM, and cache per-venue answers on ``ctx._venue_llm_cache``."""
        papers = [
            PaperRecord(paper_id="p1", venue="Nature"),
            PaperRecord(paper_id="p2", venue="Nature"),  # dup
            PaperRecord(paper_id="p3", venue="Cell"),
            PaperRecord(paper_id="p4", venue=""),         # empty → always False
        ]
        recorded: list[list[str]] = []

        def spy(client, lf, contents, ids):
            recorded.append(list(contents))
            return {c: True for c in contents}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        lf = LLMFilter(scope="venue", prompt="is reputable")
        out = dispatch_batch(papers, lf, ctx)
        all_recorded = {v for batch in recorded for v in batch}
        assert all_recorded == {"Nature", "Cell"}
        assert out["p1"] is True
        assert out["p2"] is True  # served from cache, no extra call
        assert out["p3"] is True
        assert out["p4"] is False
        # Running the same filter again should use the cached venues.
        recorded.clear()
        dispatch_batch(papers, lf, ctx)
        assert recorded == []  # everything served from cache

    def test_venue_cache_isolated_per_filter(self, ctx, monkeypatch):
        recorded: list[str] = []

        def spy(client, lf, contents, ids):
            recorded.append(lf.name)
            return {c: True for c in contents}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        papers = [PaperRecord(paper_id="p1", venue="Nature")]
        f1 = LLMFilter(name="f1", scope="venue", prompt="x1")
        f2 = LLMFilter(name="f2", scope="venue", prompt="x2")
        dispatch_batch(papers, f1, ctx)
        dispatch_batch(papers, f2, ctx)
        assert "f1" in recorded and "f2" in recorded

    def test_llm_exception_defaults_to_false(self, ctx, monkeypatch):
        """If the client raises, every item in that batch defaults to False."""
        class Broken:
            def call(self, *a, **kw):
                raise RuntimeError("down")

        monkeypatch.setitem(llm_runner._LLM_CLIENTS, id(ctx), Broken())
        lf = LLMFilter(scope="title", prompt="x")
        out = dispatch_batch(self._papers(3), lf, ctx)
        assert all(v is False for v in out.values())

    def test_client_for_is_cached(self, ctx):
        a = _client_for(ctx)
        b = _client_for(ctx)
        assert a is b


# ---------------------------------------------------------------------------
# Per-filter model + reasoning_effort overrides
# ---------------------------------------------------------------------------


class TestPerFilterOverrides:
    def test_no_override_reuses_client(self, ctx):
        f1 = LLMFilter(name="a", scope="title", prompt="x")
        f2 = LLMFilter(name="b", scope="title", prompt="y")
        c1 = _client_for(ctx, f1)
        c2 = _client_for(ctx, f2)
        assert c1 is c2  # both key on (None, None)

    def test_different_models_get_different_clients(self, ctx, monkeypatch):
        """Two filters with distinct ``model`` overrides must map to distinct
        cached client instances. Mock ``build_llm_client`` to count builds."""
        build_calls: list[tuple] = []

        class FakeClient:
            def __init__(self, model): self.model = model
            def call(self, *a, **kw): return LLMResponse("[]", [])
            supports_logprobs = False

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            build_calls.append((model, reasoning_effort))
            return FakeClient(model)

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f1 = LLMFilter(name="f1", scope="title", prompt="x", model="gpt-4o")
        f2 = LLMFilter(name="f2", scope="title", prompt="y", model="gemini-2.5-flash")
        c1 = _client_for(ctx, f1)
        c2 = _client_for(ctx, f2)
        assert c1 is not c2
        assert c1.model == "gpt-4o"
        assert c2.model == "gemini-2.5-flash"
        # Two distinct builds happened.
        assert (("gpt-4o", None) in build_calls)
        assert (("gemini-2.5-flash", None) in build_calls)
        # Re-requesting f1 should reuse the cached client, not rebuild.
        before = len(build_calls)
        c1_again = _client_for(ctx, f1)
        assert c1_again is c1
        assert len(build_calls) == before

    def test_model_and_reasoning_both_affect_cache_key(self, ctx, monkeypatch):
        """Same model + different reasoning_effort → separate clients."""
        counts = {"built": 0}

        class FakeClient:
            def __init__(self): pass

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            counts["built"] += 1
            return FakeClient()

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f_low = LLMFilter(
            name="low", scope="title", prompt="x",
            model="gpt-4o", reasoning_effort="low",
        )
        f_high = LLMFilter(
            name="high", scope="title", prompt="x",
            model="gpt-4o", reasoning_effort="high",
        )
        _client_for(ctx, f_low)
        _client_for(ctx, f_high)
        assert counts["built"] == 2

    def test_cross_provider_override_routes_through_factory(self, ctx, monkeypatch):
        """Base config is ``stub``, but a filter overrides to ``gemini-2.5-flash``.
        The factory must dispatch to :class:`GeminiClient` regardless of the
        base model. We mock the Gemini client to avoid needing a real SDK."""
        from citeclaw.clients.llm import gemini as gemini_mod

        built: list[dict] = []

        class FakeGemini:
            supports_logprobs = False

            def __init__(self, config, budget, *, model=None, reasoning_effort=None):
                built.append({"model": model, "reasoning_effort": reasoning_effort})
                self._model = model

            @staticmethod
            def matches(model: str) -> bool:
                return model.startswith("gemini-")

            def call(self, system, user, *, with_logprobs=False, category="other"):
                return LLMResponse(text='[{"index":1,"match":true}]', logprob_tokens=[])

        monkeypatch.setattr(gemini_mod, "GeminiClient", FakeGemini)
        # Also import factory AFTER patching to pick up the replacement? The
        # factory imports GeminiClient at module load, so we also patch the
        # factory's reference.
        from citeclaw.clients.llm import factory as factory_mod
        monkeypatch.setattr(factory_mod, "GeminiClient", FakeGemini)

        # Base is stub — but a filter says model=gemini-2.5-flash.
        assert ctx.config.screening_model == "stub"
        f = LLMFilter(
            name="abstract", scope="title_abstract", prompt="x",
            model="gemini-2.5-flash",
        )
        out = dispatch_batch([PaperRecord(paper_id="p1", title="T", abstract="A")], f, ctx)
        assert out == {"p1": True}
        assert len(built) == 1
        assert built[0]["model"] == "gemini-2.5-flash"

    def test_cache_survives_poison_of_legacy_global(self, ctx):
        """Existing tests that poison ``_LLM_CLIENTS`` still work — the
        legacy path returns the poisoned client when no per-filter override
        is in play."""
        class Poisoned:
            marker = "poisoned"

        llm_runner._LLM_CLIENTS[id(ctx)] = Poisoned()
        try:
            c = _client_for(ctx)
            assert getattr(c, "marker", None) == "poisoned"
        finally:
            llm_runner._LLM_CLIENTS.pop(id(ctx), None)

    def test_dispatch_batch_uses_filter_override(self, ctx, monkeypatch):
        """``dispatch_batch`` should route the LLMFilter's overrides through
        ``_client_for`` and ultimately down into the batched call."""
        calls: list[dict] = []

        class FakeClient:
            supports_logprobs = False
            def __init__(self, **kw): self.kw = kw
            def call(self, system, user, *, with_logprobs=False, category="other"):
                calls.append({"system": system, "user": user})
                return LLMResponse(text='[{"index":1,"match":true}]', logprob_tokens=[])

        # Capture the build kwargs per filter.
        build_args: list[dict] = []

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            build_args.append({"model": model, "reasoning_effort": reasoning_effort})
            return FakeClient()

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f = LLMFilter(
            name="abstract", scope="title", prompt="x",
            model="gpt-4o", reasoning_effort="medium",
        )
        papers = [PaperRecord(paper_id="p1", title="T", abstract="A")]
        dispatch_batch(papers, f, ctx)
        assert build_args == [{"model": "gpt-4o", "reasoning_effort": "medium"}]
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# LLM voting — multiple independent votes per paper with min_accepts threshold
# ---------------------------------------------------------------------------


class TestVoting:
    def _papers(self, n):
        return [
            PaperRecord(paper_id=f"p{i}", title=f"Title {i}", abstract=f"Abs {i}")
            for i in range(n)
        ]

    def test_votes_one_matches_today_behaviour(self, ctx):
        """With votes=1 min_accepts=1 the behaviour is identical to today:
        stub says match=true, so everything passes."""
        f = LLMFilter(scope="title", prompt="x", votes=1, min_accepts=1)
        out = dispatch_batch(self._papers(3), f, ctx)
        assert all(out.values())

    def test_stub_three_votes_two_accepts(self, ctx):
        """Stub is deterministic and always accepts, so 3 votes × 3 papers
        yields all accepts — everyone passes even with min_accepts=2."""
        f = LLMFilter(scope="title", prompt="x", votes=3, min_accepts=2)
        out = dispatch_batch(self._papers(5), f, ctx)
        assert all(out.values())

    def test_vote_aggregation_with_mixed_spy(self, ctx, monkeypatch):
        """Monkey-patch ``_run_one_batch`` to return a deterministic mix of
        accept/reject verdicts per vote index. Verify the tally math:
        papers with accepts >= min_accepts pass, the rest reject."""
        # Each call into the spy counts as one "vote run". We cycle through
        # a pre-baked schedule so paper p0 gets 3/3 accepts, p1 gets 2/3,
        # p2 gets 1/3, p3 gets 0/3.
        schedules = {
            "p0": [True, True, True],
            "p1": [True, True, False],
            "p2": [True, False, False],
            "p3": [False, False, False],
        }
        call_idx = {"n": 0}

        def spy(client, lf, contents, ids):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return {pid: schedules[pid][idx] for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        # All 4 papers fit in one batch so the spy is called exactly 3 times
        # (once per vote). Set batch_size high enough.
        ctx.config.llm_batch_size = 10
        ctx.config.llm_concurrency = 1

        f = LLMFilter(scope="title", prompt="x", votes=3, min_accepts=2)
        out = dispatch_batch(self._papers(4), f, ctx)
        assert out == {
            "p0": True,    # 3 >= 2
            "p1": True,    # 2 >= 2
            "p2": False,   # 1 < 2
            "p3": False,   # 0 < 2
        }
        # The spy must have run exactly V = 3 times (one per vote).
        assert call_idx["n"] == 3

    def test_min_accepts_equal_to_votes_requires_unanimous(self, ctx, monkeypatch):
        """With min_accepts == votes, every vote must accept."""
        schedules = {
            "p0": [True, True, True],
            "p1": [True, True, False],
        }
        call_idx = {"n": 0}

        def spy(client, lf, contents, ids):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return {pid: schedules[pid][idx] for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        ctx.config.llm_batch_size = 10
        ctx.config.llm_concurrency = 1
        f = LLMFilter(scope="title", prompt="x", votes=3, min_accepts=3)
        out = dispatch_batch(self._papers(2), f, ctx)
        assert out == {"p0": True, "p1": False}

    def test_failed_vote_counts_as_reject(self, ctx, monkeypatch):
        """If one of the V votes raises, it counts as a reject. A paper
        with V-1 accepts + 1 failed vote against min_accepts=V gets rejected."""
        call_idx = {"n": 0}

        def spy(client, lf, contents, ids):
            idx = call_idx["n"]
            call_idx["n"] += 1
            if idx == 1:
                # Simulate an exception path: _run_one_batch wraps client.call
                # and returns {pid: False} on exceptions. We model the same here.
                return {pid: False for pid in ids}
            return {pid: True for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        ctx.config.llm_batch_size = 10
        ctx.config.llm_concurrency = 1
        f = LLMFilter(scope="title", prompt="x", votes=3, min_accepts=3)
        out = dispatch_batch(self._papers(2), f, ctx)
        assert out == {"p0": False, "p1": False}

    def test_venue_cache_stores_list_of_votes(self, ctx, monkeypatch):
        """Venue scope caches per-venue ``list[bool]`` of length V."""
        def spy(client, lf, contents, ids):
            return {v: True for v in contents}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        ctx.config.llm_concurrency = 1
        papers = [
            PaperRecord(paper_id="p1", venue="Nature"),
            PaperRecord(paper_id="p2", venue="Nature"),
            PaperRecord(paper_id="p3", venue="Cell"),
        ]
        f = LLMFilter(scope="venue", prompt="is reputable", votes=4, min_accepts=2)
        out = dispatch_batch(papers, f, ctx)
        assert all(out.values())
        cache = ctx._venue_llm_cache["llm"]
        assert isinstance(cache["Nature"], list)
        assert len(cache["Nature"]) == 4
        assert all(cache["Nature"])
        assert len(cache["Cell"]) == 4

    def test_venue_voting_rejects_below_threshold(self, ctx, monkeypatch):
        """If a venue gets 1 accept out of 3 votes against min_accepts=2,
        every paper with that venue should reject."""
        call_idx = {"n": 0}

        def spy(client, lf, contents, ids):
            idx = call_idx["n"]
            call_idx["n"] += 1
            # First vote: accept. Votes 2 and 3: reject.
            return {v: (idx == 0) for v in contents}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        ctx.config.llm_concurrency = 1
        papers = [PaperRecord(paper_id="p1", venue="Nature")]
        f = LLMFilter(scope="venue", prompt="x", votes=3, min_accepts=2)
        out = dispatch_batch(papers, f, ctx)
        assert out == {"p1": False}

    def test_voting_warning_logged_once(self, ctx, caplog):
        """A voting filter should log a budget warning the first time it
        runs, and only once per filter name per context."""
        f = LLMFilter(scope="title", prompt="x", votes=3, min_accepts=2)
        with caplog.at_level("WARNING", logger="citeclaw.llm_runner"):
            dispatch_batch(self._papers(2), f, ctx)
            dispatch_batch(self._papers(2), f, ctx)
        msgs = [r.message for r in caplog.records if "scales" in r.message]
        assert len(msgs) == 1
        assert "3" in msgs[0]

    def test_no_warning_when_votes_is_one(self, ctx, caplog):
        """No warning should fire for the default votes=1 case."""
        f = LLMFilter(scope="title", prompt="x")
        with caplog.at_level("WARNING", logger="citeclaw.llm_runner"):
            dispatch_batch(self._papers(2), f, ctx)
        assert not any("scales" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Structured output: response_format / response_schema plumbing
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Verify that OpenAI and Gemini clients pass structured-output kwargs
    to their SDKs. All SDK calls are mocked — no real network traffic."""

    def test_openai_client_passes_response_format(self, tmp_path, monkeypatch):
        from citeclaw.clients.llm.openai_client import OpenAIClient
        from citeclaw.config import BudgetTracker, Settings

        captured: dict = {}

        class FakeMessage:
            content = '{"results": [{"index": 1, "match": true}]}'

        class FakeChoice:
            message = FakeMessage()
            logprobs = None

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 5
            completion_tokens_details = None

        class FakeResp:
            choices = [FakeChoice()]
            usage = FakeUsage()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeSDK:
            chat = FakeChat()

        # Avoid the real openai.OpenAI constructor.
        monkeypatch.setattr(
            "citeclaw.clients.llm.openai_client._build_openai_sdk",
            lambda cfg: FakeSDK(),
        )

        cfg = Settings(
            data_dir=tmp_path,
            screening_model="gpt-4o",
            openai_api_key="sk-test",
        )
        client = OpenAIClient(cfg, BudgetTracker())
        schema = {"type": "object", "properties": {}}
        resp = client.call("sys", "usr", response_schema=schema, category="t")
        assert "response_format" in captured
        rf = captured["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] is schema
        assert resp.text.startswith("{")

    def test_openai_client_skips_response_format_without_schema(self, tmp_path, monkeypatch):
        from citeclaw.clients.llm.openai_client import OpenAIClient
        from citeclaw.config import BudgetTracker, Settings

        captured: dict = {}

        class FakeResp:
            class _C:
                class _M:
                    content = "{}"
                message = _M()
                logprobs = None
            choices = [_C()]
            class _U:
                prompt_tokens = 1
                completion_tokens = 1
                completion_tokens_details = None
            usage = _U()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResp()

        class FakeSDK:
            chat = type("C", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(
            "citeclaw.clients.llm.openai_client._build_openai_sdk",
            lambda cfg: FakeSDK(),
        )
        cfg = Settings(data_dir=tmp_path, screening_model="gpt-4o", openai_api_key="sk-test")
        client = OpenAIClient(cfg, BudgetTracker())
        client.call("sys", "usr", category="t")  # no response_schema
        assert "response_format" not in captured

    def test_openai_kill_switch_disables_structured_output(self, tmp_path, monkeypatch):
        from citeclaw.clients.llm.openai_client import OpenAIClient
        from citeclaw.config import BudgetTracker, Settings

        captured: dict = {}

        class FakeResp:
            class _C:
                class _M:
                    content = "{}"
                message = _M()
                logprobs = None
            choices = [_C()]
            class _U:
                prompt_tokens = 1
                completion_tokens = 1
                completion_tokens_details = None
            usage = _U()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResp()

        class FakeSDK:
            chat = type("C", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(
            "citeclaw.clients.llm.openai_client._build_openai_sdk",
            lambda cfg: FakeSDK(),
        )
        cfg = Settings(
            data_dir=tmp_path,
            screening_model="gpt-4o",
            openai_api_key="sk-test",
            structured_output_enabled=False,
        )
        client = OpenAIClient(cfg, BudgetTracker())
        client.call("sys", "usr", response_schema={"x": 1}, category="t")
        assert "response_format" not in captured

    def test_dispatch_batch_threads_schema_into_client(self, ctx, monkeypatch):
        """The dispatch wiring should pass ``response_schema`` through to
        ``client.call``. Verified by spying on a fake client's call args."""
        captured: list[dict] = []

        class FakeClient:
            supports_logprobs = False
            def call(self, system, user, *, with_logprobs=False, category="other",
                     response_schema=None):
                captured.append({"response_schema": response_schema, "user": user})
                return LLMResponse(
                    text='{"results":[{"index":1,"match":true}]}',
                    logprob_tokens=[],
                )

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            return FakeClient()

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f = LLMFilter(scope="title", prompt="x")
        papers = [PaperRecord(paper_id="p1", title="t")]
        dispatch_batch(papers, f, ctx)
        assert captured[0]["response_schema"] is not None
        # Must be the canonical citeclaw screening schema.
        schema = captured[0]["response_schema"]
        assert schema.get("type") == "object"
        assert "results" in schema.get("properties", {})

    def test_legacy_client_without_response_schema_kwarg_still_works(self, ctx, monkeypatch):
        """A third-party/test client that doesn't accept ``response_schema``
        should still be dispatched to — the runner falls back to the plain
        signature on ``TypeError``."""
        captured: list[dict] = []

        class LegacyClient:
            supports_logprobs = False
            def call(self, system, user, *, with_logprobs=False, category="other"):
                captured.append({"user": user})
                return LLMResponse(
                    text='{"results":[{"index":1,"match":true}]}',
                    logprob_tokens=[],
                )

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            return LegacyClient()

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f = LLMFilter(scope="title", prompt="x")
        papers = [PaperRecord(paper_id="p1", title="t")]
        out = dispatch_batch(papers, f, ctx)
        assert out == {"p1": True}
        assert len(captured) == 1

    def test_parse_survives_malformed_batch_before_structured_output(self):
        """Document the prior failure mode: a malformed JSON response made
        _parse_matches return None and the whole batch defaulted to False.
        Structured output should make this unreachable in practice, but we
        still want the parser to handle bad text without crashing."""
        bad = '{"results": [{"index": 1, "match": tru'  # truncated
        assert _parse_matches(bad, 3) is None

    def test_schema_module_exports_canonical_shape(self):
        from citeclaw.screening.schemas import (
            SCREENING_SCHEMA_NAME,
            openai_response_format,
            screening_json_schema,
        )
        schema = screening_json_schema()
        assert schema["type"] == "object"
        assert "results" in schema["properties"]
        items = schema["properties"]["results"]["items"]
        assert items["properties"]["index"]["type"] == "integer"
        assert items["properties"]["match"]["type"] == "boolean"
        assert items["required"] == ["index", "match"]

        rf = openai_response_format()
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == SCREENING_SCHEMA_NAME
        assert rf["json_schema"]["strict"] is True


# ---------------------------------------------------------------------------
# Formula-mode LLMFilter — Boolean composition over named sub-queries
# ---------------------------------------------------------------------------


class TestFormulaDispatch:
    def _papers(self, n):
        return [
            PaperRecord(paper_id=f"p{i}", title=f"T{i}", abstract=f"A{i}")
            for i in range(n)
        ]

    def test_formula_runs_each_subquery(self, ctx, monkeypatch):
        """Each named sub-query should become a synthetic LLMFilter whose
        name is ``parent::qname``. Verify all referenced sub-queries run
        exactly once per batch."""
        seen_names: list[str] = []

        def spy(client, lf, contents, ids):
            seen_names.append(lf.name)
            return {pid: True for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        f = LLMFilter(
            name="abs_llm",
            scope="title",
            formula="q1 & q2",
            queries={"q1": "is ml", "q2": "is stats"},
        )
        out = dispatch_batch(self._papers(2), f, ctx)
        assert all(out.values())
        # Both sub-queries should have fired, none of the unused ones
        assert set(seen_names) == {"abs_llm::q1", "abs_llm::q2"}

    def test_formula_evaluation_and_true_false(self, ctx, monkeypatch):
        """Verify the Boolean combination: q1=T q2=F → q1&q2 is False, q1|q2 is True."""
        def spy(client, lf, contents, ids):
            if lf.name.endswith("::q1"):
                return {pid: True for pid in ids}
            return {pid: False for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)

        and_filter = LLMFilter(
            scope="title", formula="q1 & q2",
            queries={"q1": "x", "q2": "y"},
        )
        or_filter = LLMFilter(
            scope="title", formula="q1 | q2",
            queries={"q1": "x", "q2": "y"},
        )
        ps = self._papers(2)
        assert dispatch_batch(ps, and_filter, ctx) == {"p0": False, "p1": False}
        assert dispatch_batch(ps, or_filter, ctx) == {"p0": True, "p1": True}

    def test_formula_negation(self, ctx, monkeypatch):
        """``!q1`` should flip True sub-query verdicts to False at the
        filter level."""
        def spy(client, lf, contents, ids):
            return {pid: True for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        f = LLMFilter(
            scope="title", formula="!q1", queries={"q1": "is survey"},
        )
        out = dispatch_batch(self._papers(1), f, ctx)
        assert out == {"p0": False}

    def test_formula_per_paper_mixed_verdicts(self, ctx, monkeypatch):
        """Different papers should get different final verdicts based on
        how each sub-query scored them. Schedule:
          p0: q1=T, q2=T, q3=F → (q1|q2)&!q3 = True
          p1: q1=T, q2=F, q3=T → (q1|q2)&!q3 = False
          p2: q1=F, q2=F, q3=F → (q1|q2)&!q3 = False
        """
        schedule = {
            "p0": {"q1": True, "q2": True, "q3": False},
            "p1": {"q1": True, "q2": False, "q3": True},
            "p2": {"q1": False, "q2": False, "q3": False},
        }

        def spy(client, lf, contents, ids):
            _, _, qname = lf.name.partition("::")
            return {pid: schedule[pid][qname] for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        f = LLMFilter(
            scope="title",
            formula="(q1 | q2) & !q3",
            queries={"q1": "a", "q2": "b", "q3": "c"},
        )
        out = dispatch_batch(self._papers(3), f, ctx)
        assert out == {"p0": True, "p1": False, "p2": False}

    def test_formula_inherits_voting_settings(self, ctx, monkeypatch):
        """votes and min_accepts on the parent filter should flow into
        each synthetic sub-filter so voting happens per-sub-query."""
        seen_votes: list[int] = []
        seen_min: list[int] = []

        def spy(client, lf, contents, ids):
            seen_votes.append(lf.votes)
            seen_min.append(lf.min_accepts)
            return {pid: True for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        f = LLMFilter(
            scope="title",
            formula="q1",
            queries={"q1": "is ml"},
            votes=3, min_accepts=2,
        )
        dispatch_batch(self._papers(1), f, ctx)
        # The sub-query should have inherited votes=3 → 3 calls, each with min_accepts=2
        assert all(v == 3 for v in seen_votes)
        assert all(m == 2 for m in seen_min)
        assert len(seen_votes) == 3  # three independent voting calls

    def test_formula_inherits_model_override(self, ctx, monkeypatch):
        """The per-filter model/reasoning override should cascade into
        sub-queries so every sub-query uses the same client."""
        builds: list[tuple] = []

        class FakeClient:
            supports_logprobs = False
            def call(self, system, user, *, with_logprobs=False, category="other",
                     response_schema=None):
                return LLMResponse(
                    text='{"results":[{"index":1,"match":true}]}',
                    logprob_tokens=[],
                )

        def spy_build(config, budget, *, model=None, reasoning_effort=None):
            builds.append((model, reasoning_effort))
            return FakeClient()

        monkeypatch.setattr(llm_runner, "build_llm_client", spy_build)
        f = LLMFilter(
            scope="title",
            formula="q1 & q2",
            queries={"q1": "a", "q2": "b"},
            model="gemini-2.5-flash",
            reasoning_effort="low",
        )
        dispatch_batch(self._papers(1), f, ctx)
        # Sub-queries inherit model + reasoning, so they share a single cached
        # client. At most one build call for the (model, reasoning) key.
        assert builds.count(("gemini-2.5-flash", "low")) == 1

    def test_formula_single_prompt_mode_backwards_compatible(self, ctx):
        """A filter without formula/queries still dispatches through the
        simple single-prompt path."""
        f = LLMFilter(scope="title", prompt="is ml")
        out = dispatch_batch(self._papers(2), f, ctx)
        assert all(out.values())  # stub always accepts

    def test_formula_warning_logged_once(self, ctx, monkeypatch, caplog):
        def spy(client, lf, contents, ids):
            return {pid: True for pid in ids}

        monkeypatch.setattr(llm_runner, "_run_one_batch", spy)
        f = LLMFilter(
            scope="title",
            formula="q1 & q2",
            queries={"q1": "a", "q2": "b"},
        )
        with caplog.at_level("WARNING", logger="citeclaw.llm_runner"):
            dispatch_batch(self._papers(1), f, ctx)
            dispatch_batch(self._papers(1), f, ctx)
        formula_warnings = [r for r in caplog.records if "sub-queries" in r.message]
        assert len(formula_warnings) == 1

    def test_formula_via_yaml_builder(self):
        """End-to-end: YAML → build_blocks → LLMFilter with formula mode."""
        from citeclaw.filters.builder import build_blocks

        built = build_blocks({
            "f": {
                "type": "LLMFilter",
                "scope": "title_abstract",
                "formula": "(q_ml | q_stats) & !q_survey",
                "queries": {
                    "q_ml": "is ml",
                    "q_stats": "is stats",
                    "q_survey": "is survey",
                },
            }
        })
        assert built["f"].formula_expr == "(q_ml | q_stats) & !q_survey"
        assert set(built["f"].queries.keys()) == {"q_ml", "q_stats", "q_survey"}


# ---------------------------------------------------------------------------
# PB-02: stub_respond branch for the iterative meta-LLM search agent.
#
# The branch detects the agent prompt via `"agent_decision"` in the user
# string, then advances a deterministic three-state lifecycle by counting
# `"query":` JSON keys in the transcript section. Tests below pin both the
# transition logic and the four-field shape (with non-empty `thinking`)
# that PB-04 / PB-05 will rely on.
# ---------------------------------------------------------------------------


class TestStubAgentDecisionBranch:
    def _agent_user(self, transcript: str = "(no prior turns)") -> str:
        """Render the real PB-01 USER_TEMPLATE so the test catches any
        drift between the prompt module and the stub branch detector."""
        from citeclaw.prompts.search_refine import USER_TEMPLATE
        return USER_TEMPLATE.format(
            topic_description="test topic",
            anchor_papers_block="(none)",
            transcript=transcript,
            iteration=1,
            max_iterations=4,
            target_count=200,
        )

    def test_bare_template_has_zero_query_keys(self):
        """The PB-01 template's static legend must NOT itself contain
        `"query":` — otherwise the iteration counter would start at 1
        and the initial-state branch would be unreachable."""
        user = self._agent_user(transcript="(no prior turns)")
        assert user.count('"query":') == 0
        # Sanity: the branch detector token IS present.
        assert '"agent_decision"' in user

    def test_initial_state_when_no_prior_queries(self):
        user = self._agent_user(transcript="(no prior turns)")
        data = json.loads(stub_respond("", user))
        assert data["agent_decision"] == "initial"
        assert data["thinking"] == "stub: initial exploration"
        assert data["query"] == {"text": "test topic"}
        assert data["reasoning"] == "stub initial"

    def test_refine_state_when_one_prior_query(self):
        # Simulate one prior turn whose raw JSON carries one `"query":`.
        transcript = (
            'Turn 1: {"thinking": "first try", "query": {"text": "q1"}, '
            '"agent_decision": "initial", "reasoning": "r"}'
        )
        user = self._agent_user(transcript=transcript)
        assert user.count('"query":') == 1
        data = json.loads(stub_respond("", user))
        assert data["agent_decision"] == "refine"
        assert data["thinking"] == "stub: prior was too broad, narrowing"
        assert data["query"] == {"text": "test topic narrowed"}

    def test_satisfied_state_when_two_prior_queries(self):
        transcript = (
            'Turn 1: {"thinking": "a", "query": {"text": "q1"}, '
            '"agent_decision": "initial", "reasoning": "r"}\n'
            'Turn 2: {"thinking": "b", "query": {"text": "q2"}, '
            '"agent_decision": "refine", "reasoning": "r"}'
        )
        user = self._agent_user(transcript=transcript)
        assert user.count('"query":') == 2
        data = json.loads(stub_respond("", user))
        assert data["agent_decision"] == "satisfied"
        assert data["thinking"] == "stub: results saturated"

    def test_satisfied_state_for_many_prior_queries(self):
        """Once the agent has refined twice it stays satisfied — the
        ``≥2`` branch must not regress to refine for higher counts."""
        transcript = "\n".join(
            f'Turn {i}: {{"thinking": "t", "query": {{"text": "q{i}"}}, '
            f'"agent_decision": "refine", "reasoning": "r"}}'
            for i in range(5)
        )
        user = self._agent_user(transcript=transcript)
        assert user.count('"query":') == 5
        data = json.loads(stub_respond("", user))
        assert data["agent_decision"] == "satisfied"

    def test_thinking_field_is_non_empty_in_every_state(self):
        """Per PB-02 verify: every response asserts thinking is non-empty.
        This is the contract PB-05's agent tests will lean on."""
        for n_prior in (0, 1, 2, 7):
            transcript = " ".join(['"query":'] * n_prior) or "(no prior turns)"
            user = self._agent_user(transcript=transcript)
            data = json.loads(stub_respond("", user))
            assert isinstance(data["thinking"], str)
            assert data["thinking"], (
                f"empty thinking field for n_prior={n_prior}"
            )

    def test_all_four_fields_present_in_every_state(self):
        for n_prior in (0, 1, 2):
            transcript = " ".join(['"query":'] * n_prior) or "(no prior turns)"
            user = self._agent_user(transcript=transcript)
            data = json.loads(stub_respond("", user))
            assert set(data.keys()) == {
                "thinking", "query", "agent_decision", "reasoning",
            }
            assert isinstance(data["query"], dict)
            assert "text" in data["query"]
            assert data["agent_decision"] in {
                "initial", "refine", "satisfied",
            }

    def test_thinking_serialized_first_for_chain_of_thought_priming(self):
        """RESPONSE_SCHEMA places `thinking` first; the stub's serialized
        JSON should preserve that ordering so capable LLM clients (which
        surface schema fields to the model in declaration order) prime
        the chain of thought before any structured field."""
        user = self._agent_user(transcript="(no prior turns)")
        text = stub_respond("", user)
        i_thinking = text.find('"thinking"')
        i_query = text.find('"query"')
        i_decision = text.find('"agent_decision"')
        i_reasoning = text.find('"reasoning"')
        # All four field markers must be present and in the schema order.
        assert -1 < i_thinking < i_query < i_decision < i_reasoning

    def test_branch_detector_does_not_steal_unrelated_prompts(self):
        """A user prompt that mentions neither agent_decision nor any
        other recognized token must still fall through to the default
        ``[]`` — proving the new branch is truly opt-in."""
        assert stub_respond("", "some unrelated prompt") == "[]"

    def test_call_through_stubclient_with_meta_search_agent_category(
        self, tmp_path,
    ):
        """End-to-end through StubClient with the same category PB-04
        will use, so PB-05's `budget._llm_tokens["meta_search_agent"]`
        assertion is grounded."""
        cfg = Settings(data_dir=tmp_path, screening_model="stub")
        budget = BudgetTracker()
        client = StubClient(cfg, budget)
        user = self._agent_user(transcript="(no prior turns)")
        resp = client.call(
            "sys", user, category="meta_search_agent",
        )
        data = json.loads(resp.text)
        assert data["agent_decision"] == "initial"
        assert data["thinking"]
        assert budget._llm_tokens.get("meta_search_agent", 0) > 0
