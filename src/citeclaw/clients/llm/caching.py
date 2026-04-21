"""Cache-aware wrapper around any :class:`LLMClient`.

Adds a content-addressable prompt cache (the ``llm_response_cache``
table in cache.db) in front of every concrete LLMClient. The wrapper
intercepts ``call()``, hashes the prompt + model + reasoning_effort,
and either:

  - **HIT**: serves the response from the cache without touching the
    inner client. ``BudgetTracker.record_llm`` is NOT called, so cache
    hits are billed nothing — exactly the savings the user asked for.
    A separate ``llm_cache_hits`` counter on BudgetTracker tracks how
    often this happens.
  - **MISS**: forwards to the inner client (which records its own
    tokens via the existing ``record_llm`` path), then writes the
    response into the cache for next time.

Cache key is sha256 over the JSON-serialised payload of:

  - ``model`` — alias name (so identical prompts to different models
    don't collide).
  - ``reasoning_effort`` — changes thinking budget → different output.
  - ``system`` — full system prompt text.
  - ``user`` — full user prompt text.
  - ``response_schema`` — structured-output schema dict if provided.
  - ``with_logprobs`` — whether the caller asked for logprobs (a hit
    without stored logprobs is not equivalent to a fresh call with
    logprobs requested).

Things intentionally NOT in the key:

  - ``category`` — bookkeeping label only, not response-affecting.
  - temperature / max_tokens — set at client construction; currently
    always 0 / unset for screening. If you ever expose them per-call,
    add them here too.

Stub clients are NOT wrapped — there's no point caching deterministic
fake responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, TYPE_CHECKING

from citeclaw.clients.llm.base import LLMResponse

if TYPE_CHECKING:
    from citeclaw.cache import Cache
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.budget import BudgetTracker

log = logging.getLogger("citeclaw.clients.llm.cache")


def make_cache_key(
    *,
    model: str,
    reasoning_effort: str | None,
    system: str,
    user: str,
    response_schema: dict[str, Any] | None,
    with_logprobs: bool,
) -> str:
    """Compute the sha256 cache key for a single LLM call.

    Stable across runs: identical inputs always hash to the same key.
    """
    payload = {
        "model": model or "",
        "reasoning_effort": reasoning_effort or "",
        "system": system or "",
        "user": user or "",
        "response_schema": response_schema if response_schema else None,
        "with_logprobs": bool(with_logprobs),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class CachingLLMClient:
    """Transparent prompt-cache wrapper around an inner ``LLMClient``.

    The wrapper satisfies the same Protocol as the inner client (it
    has the same ``call`` signature and ``supports_logprobs``
    property), so callers don't need to know whether they're holding
    a cached or raw client.
    """

    def __init__(
        self,
        inner: "LLMClient",
        cache: "Cache",
        budget: "BudgetTracker",
        *,
        model: str,
        reasoning_effort: str | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._budget = budget
        # Store the resolved model name + reasoning_effort so the cache
        # key is computed against the *actual* model the inner client
        # will send to (not the unresolved YAML alias).
        self._model = model
        self._reasoning_effort = reasoning_effort
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @property
    def supports_logprobs(self) -> bool:
        # Defensive ``getattr`` — duck-typed test stand-ins that don't
        # implement the property shouldn't trip an AttributeError on
        # the wrapper.
        return bool(getattr(self._inner, "supports_logprobs", False))

    def call(
        self,
        system: str,
        user: str,
        *,
        with_logprobs: bool = False,
        category: str = "other",
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        key = make_cache_key(
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            system=system,
            user=user,
            response_schema=response_schema,
            with_logprobs=with_logprobs,
        )

        try:
            cached = self._cache.get_llm_response(key)
        except Exception as exc:  # noqa: BLE001 — cache read is best-effort
            log.debug("CachingLLMClient: cache read failed: %s", exc)
            cached = None

        if cached is not None:
            self.cache_hits += 1
            self._budget.record_llm_cache_hit(category)
            return LLMResponse(
                text=cached["text"],
                reasoning_content=cached.get("reasoning_content", ""),
                logprob_tokens=cached.get("logprob_tokens", []),
            )

        self.cache_misses += 1
        # Forward only the kwargs the caller actually used. Some inner
        # clients (older fakes, third-party adapters) don't declare
        # ``response_schema`` in their ``call`` signature, so passing
        # ``response_schema=None`` unconditionally would TypeError. By
        # only forwarding it when non-None we keep the wrapper compatible
        # with any LLMClient that didn't adopt the optional kwarg yet.
        forward_kwargs: dict[str, Any] = {
            "with_logprobs": with_logprobs,
            "category": category,
        }
        if response_schema is not None:
            forward_kwargs["response_schema"] = response_schema
        resp = self._inner.call(system, user, **forward_kwargs)
        try:
            self._cache.put_llm_response(
                key,
                model=self._model,
                text=resp.text,
                reasoning_content=resp.reasoning_content or "",
                logprob_tokens=resp.logprob_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — cache write is best-effort
            # Warn (not debug) — a write failure means every future
            # request for this prompt re-pays for the LLM call. The
            # user should see this in the file log even at default level.
            log.warning("CachingLLMClient: cache write failed: %s", exc)
        return resp
