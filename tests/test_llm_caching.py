"""Tests for the LLM prompt cache (cache.db ``llm_response_cache`` table
+ :class:`citeclaw.clients.llm.caching.CachingLLMClient`).

The cache is content-addressable: identical prompts (model + system +
user + reasoning_effort + response_schema + with_logprobs) hash to the
same key and serve the response from cache.db without touching the
underlying LLM client. The wrapper is what makes "always cache LLM
calls" work for the user — these tests pin the hit/miss semantics so
nobody silently breaks the savings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citeclaw.cache import Cache
from citeclaw.clients.llm.base import LLMResponse
from citeclaw.clients.llm.caching import CachingLLMClient, make_cache_key
from citeclaw.config import BudgetTracker


class _RecordingClient:
    """LLMClient stand-in that records every call and returns a
    deterministic response. Tracks call counts so we can assert the
    cache wrapper actually skipped the inner client on a hit.
    """

    supports_logprobs = False

    def __init__(self, *, response_text: str = "ok") -> None:
        self._response_text = response_text
        self.calls: list[dict] = []

    def call(
        self,
        system,
        user,
        *,
        with_logprobs=False,
        category="other",
        response_schema=None,
    ):
        self.calls.append({
            "system": system,
            "user": user,
            "category": category,
            "response_schema": response_schema,
            "with_logprobs": with_logprobs,
        })
        return LLMResponse(
            text=self._response_text,
            reasoning_content="thinking trace",
            logprob_tokens=[],
        )


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_identical_inputs_hash_to_same_key(self):
        a = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        b = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        assert a == b

    def test_different_model_changes_key(self):
        a = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        b = make_cache_key(
            model="gemini-2.5-flash", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        assert a != b

    def test_different_reasoning_effort_changes_key(self):
        a = make_cache_key(
            model="gpt-5", reasoning_effort="low",
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        b = make_cache_key(
            model="gpt-5", reasoning_effort="high",
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        assert a != b

    def test_different_user_text_changes_key(self):
        a = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="paper A",
            response_schema=None, with_logprobs=False,
        )
        b = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="paper B",
            response_schema=None, with_logprobs=False,
        )
        assert a != b

    def test_with_logprobs_flag_changes_key(self):
        """A hit without stored logprobs is not equivalent to a fresh
        call that asked for them — different keys must be used."""
        a = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=False,
        )
        b = make_cache_key(
            model="gpt-4o", reasoning_effort=None,
            system="sys", user="usr",
            response_schema=None, with_logprobs=True,
        )
        assert a != b


# ---------------------------------------------------------------------------
# CachingLLMClient — hit/miss/budget bookkeeping
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache.db")


@pytest.fixture
def budget() -> BudgetTracker:
    return BudgetTracker()


class TestCachingLLMClient:
    def test_first_call_misses_then_calls_inner(self, cache, budget):
        inner = _RecordingClient(response_text="answer")
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        resp = wrapper.call("sys", "usr")
        assert resp.text == "answer"
        assert wrapper.cache_hits == 0
        assert wrapper.cache_misses == 1
        assert len(inner.calls) == 1
        # Budget hit counter still 0 (only the inner call was made — and
        # the inner client didn't record any tokens itself in this fake).
        assert budget.llm_cache_hits == 0

    def test_second_identical_call_hits_cache_and_skips_inner(
        self, cache, budget,
    ):
        inner = _RecordingClient(response_text="answer")
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        wrapper.call("sys", "usr")  # miss → store
        resp = wrapper.call("sys", "usr")  # should HIT
        assert resp.text == "answer"
        assert resp.reasoning_content == "thinking trace"
        assert wrapper.cache_hits == 1
        assert wrapper.cache_misses == 1
        # Inner client still only called ONCE — the second call was
        # served from cache.
        assert len(inner.calls) == 1
        # Budget cache hit counter incremented.
        assert budget.llm_cache_hits == 1

    def test_different_user_text_misses_separately(self, cache, budget):
        inner = _RecordingClient(response_text="answer")
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        wrapper.call("sys", "paper A")
        wrapper.call("sys", "paper B")
        assert wrapper.cache_misses == 2
        assert wrapper.cache_hits == 0
        assert len(inner.calls) == 2

    def test_different_model_misses_separately(self, cache, budget):
        inner_a = _RecordingClient(response_text="A's answer")
        inner_b = _RecordingClient(response_text="B's answer")
        wrap_a = CachingLLMClient(
            inner_a, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        wrap_b = CachingLLMClient(
            inner_b, cache, budget, model="gemini-2.5-flash",
            reasoning_effort=None,
        )
        ra = wrap_a.call("sys", "usr")
        rb = wrap_b.call("sys", "usr")
        # Each wrapper saw a miss against its own model key.
        assert ra.text == "A's answer"
        assert rb.text == "B's answer"
        assert len(inner_a.calls) == 1
        assert len(inner_b.calls) == 1

    def test_cache_persists_across_wrapper_instances(self, cache, budget):
        """A second wrapper instance pointing at the same Cache should
        see the cached response from the first wrapper's call. This is
        the across-runs persistence the user asked for."""
        inner_first = _RecordingClient(response_text="first")
        first = CachingLLMClient(
            inner_first, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        first.call("sys", "usr")
        assert len(inner_first.calls) == 1

        # New wrapper, new inner client — should hit the cache and never
        # call the new inner.
        inner_second = _RecordingClient(response_text="second")
        second = CachingLLMClient(
            inner_second, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        resp = second.call("sys", "usr")
        assert resp.text == "first"  # cached value, not the new inner's
        assert second.cache_hits == 1
        assert len(inner_second.calls) == 0

    def test_response_schema_only_forwarded_when_set(self, cache, budget):
        """The wrapper only passes ``response_schema`` through to the
        inner client when non-None, so legacy clients that don't accept
        the kwarg keep working."""
        inner = _RecordingClient(response_text="ok")
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        wrapper.call("sys", "usr")  # response_schema=None
        assert inner.calls[0]["response_schema"] is None  # default

        wrapper.call("sys", "usr2", response_schema={"type": "object"})
        assert inner.calls[1]["response_schema"] == {"type": "object"}

    def test_response_schema_changes_cache_key(self, cache, budget):
        inner = _RecordingClient(response_text="ok")
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        wrapper.call("sys", "usr", response_schema={"type": "object"})
        wrapper.call("sys", "usr", response_schema={"type": "array"})
        # Different schemas → both miss
        assert wrapper.cache_misses == 2
        assert wrapper.cache_hits == 0

    def test_supports_logprobs_proxies_to_inner(self, cache, budget):
        inner = _RecordingClient()
        inner.supports_logprobs = True  # noqa: SLF001 — fake mutation
        wrapper = CachingLLMClient(
            inner, cache, budget, model="gpt-4o", reasoning_effort=None,
        )
        assert wrapper.supports_logprobs is True
