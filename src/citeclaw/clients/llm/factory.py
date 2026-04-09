"""LLM client factory + provider-detection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from citeclaw.clients.llm.base import LLMClient
from citeclaw.clients.llm.caching import CachingLLMClient
from citeclaw.clients.llm.gemini import GeminiClient
from citeclaw.clients.llm.openai_client import OpenAIClient
from citeclaw.clients.llm.stub import StubClient
from citeclaw.config import BudgetTracker, ModelEndpoint, Settings

if TYPE_CHECKING:
    from citeclaw.cache import Cache


def is_stub(config: Settings) -> bool:
    return (config.screening_model or "").strip().lower() == "stub"


def _build_registry_client(
    config: Settings,
    budget: BudgetTracker,
    *,
    alias: str,
    entry: ModelEndpoint,
    reasoning_effort: str | None,
) -> LLMClient:
    """Construct an OpenAIClient pointed at a registry endpoint.

    The YAML alias becomes the OpenAIClient's ``model`` (so budget /
    pricing keys are stable across the run), while ``served_model_name``
    is what the SDK actually puts in the chat-completions ``model``
    field. The registry's ``api_key_env`` resolves to a real bearer
    token at construction time — the key never sits in YAML or memory
    for longer than the SDK call.
    """
    api_key = entry.resolved_api_key
    return OpenAIClient(
        config,
        budget,
        model=alias,
        reasoning_effort=reasoning_effort,
        endpoint_base_url=entry.base_url,
        endpoint_api_key=api_key or None,
        endpoint_timeout=entry.request_timeout,
        served_model_name=entry.served_model_name or alias,
    )


def build_llm_client(
    config: Settings,
    budget: BudgetTracker,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    cache: "Cache | None" = None,
) -> LLMClient:
    """Build the LLMClient that matches the configured ``screening_model``.

    Priority:
        1. ``stub`` → StubClient.
        2. Registry hit (``Settings.models[alias]``) → OpenAIClient pointed
           at that alias's endpoint with its ``served_model_name``.
        3. Legacy custom endpoint (``llm_base_url``) → OpenAIClient.
        4. Gemini-shaped model name → GeminiClient.
        5. Anything else → OpenAIClient (SaaS OpenAI).

    When ``model`` is provided, all routing is done on the *resolved*
    model (so a base ``screening_model=gpt-4o`` config with a filter-level
    ``model=gemini-2.5-flash`` correctly lands on :class:`GeminiClient`,
    and ``model=gemma-4-31b`` lands on the registry entry).

    When ``cache`` is provided AND the resolved client is not the stub,
    the result is wrapped in a :class:`CachingLLMClient` that hashes
    every prompt against the ``llm_response_cache`` table in cache.db
    so repeat calls skip the model entirely and pay zero tokens.
    """
    effective_model = (model or config.screening_model or "").strip()
    if effective_model.lower() == "stub":
        # Stub responses are deterministic and free; no point caching.
        return StubClient(config, budget, model=model, reasoning_effort=reasoning_effort)

    if effective_model in config.models:
        inner: LLMClient = _build_registry_client(
            config,
            budget,
            alias=effective_model,
            entry=config.models[effective_model],
            reasoning_effort=reasoning_effort,
        )
    elif config.llm_base_url:
        # Legacy: a single global custom endpoint hosts whatever model the
        # user picked. The registry path above supersedes this for new
        # configs but old YAMLs keep working.
        inner = OpenAIClient(
            config, budget, model=model, reasoning_effort=reasoning_effort,
        )
    elif GeminiClient.matches(effective_model):
        inner = GeminiClient(
            config, budget, model=model, reasoning_effort=reasoning_effort,
        )
    else:
        inner = OpenAIClient(
            config, budget, model=model, reasoning_effort=reasoning_effort,
        )

    if cache is None:
        return inner
    return CachingLLMClient(
        inner, cache, budget,
        model=effective_model,
        reasoning_effort=reasoning_effort,
    )


def supports_logprobs(client: LLMClient) -> bool:
    return bool(getattr(client, "supports_logprobs", False))
