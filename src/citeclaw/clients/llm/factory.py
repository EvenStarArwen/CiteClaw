"""LLM client factory + provider-detection helpers."""

from __future__ import annotations

from citeclaw.clients.llm.base import LLMClient
from citeclaw.clients.llm.gemini import GeminiClient
from citeclaw.clients.llm.openai_client import OpenAIClient
from citeclaw.clients.llm.stub import StubClient
from citeclaw.config import BudgetTracker, Settings


def is_stub(config: Settings) -> bool:
    return (config.screening_model or "").strip().lower() == "stub"


def build_llm_client(
    config: Settings,
    budget: BudgetTracker,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMClient:
    """Build the LLMClient that matches the configured ``screening_model``.

    Priority: stub > custom OpenAI-compatible endpoint > Gemini > OpenAI.

    When ``model`` is provided, provider detection routes on the *resolved*
    model (so a base ``screening_model=gpt-4o`` config with a filter-level
    ``model=gemini-2.5-flash`` correctly lands on :class:`GeminiClient`).
    The custom OpenAI-compatible endpoint (``llm_base_url``) always stays
    on the OpenAI SDK path regardless of the model string — that endpoint
    hosts whatever model string the user declares, and switching provider
    families for it doesn't make sense.
    """
    effective_model = (model or config.screening_model or "").strip()
    if effective_model.lower() == "stub":
        return StubClient(config, budget, model=model, reasoning_effort=reasoning_effort)
    if config.llm_base_url:
        # Custom endpoint always takes the OpenAI-SDK path.
        return OpenAIClient(config, budget, model=model, reasoning_effort=reasoning_effort)
    if GeminiClient.matches(effective_model):
        return GeminiClient(config, budget, model=model, reasoning_effort=reasoning_effort)
    return OpenAIClient(config, budget, model=model, reasoning_effort=reasoning_effort)


def supports_logprobs(client: LLMClient) -> bool:
    return bool(getattr(client, "supports_logprobs", False))
