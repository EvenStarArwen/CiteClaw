"""Provider-agnostic token extraction.

Each LLM provider returns its usage object in a different shape:

* OpenAI / OpenAI-compat (vLLM, Grok, Mistral): ``response.usage`` is
  a ``CompletionUsage`` with ``prompt_tokens`` + ``completion_tokens``.
  Reasoning tokens (when present) hang off
  ``response.usage.completion_tokens_details.reasoning_tokens``.
  Note: vLLM-hosted reasoning models (Gemma 4, Qwen3, DeepSeek-R1)
  fold the thinking trace into ``completion_tokens`` and do NOT
  populate ``completion_tokens_details``. The thinking text is
  reachable on ``message.reasoning`` / ``.reasoning_content`` but
  the *count* is not separately reported by the vLLM OpenAI surface.
  Net effect: ``reasoning`` is 0 for vLLM thinking calls; the cost
  estimate still comes out correct because the thinking text is
  already counted in completion_tokens (which is billed at the
  output rate — see budget.py:MODEL_PRICING reasoning convention).

* Gemini: ``response.usage_metadata`` with ``prompt_token_count``,
  ``candidates_token_count``, ``thinking_token_count`` (the last
  populated only for 2.5+ thinking-capable models).

* Stub: no real provider response — caller fakes counts from prompt
  / response length.

The ``TokenUsage`` dataclass is the unified return type: ``prompt``,
``completion``, ``reasoning``. Pass it to
:meth:`citeclaw.budget.BudgetTracker.record_llm` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TokenUsage:
    """Provider-agnostic token counts for one LLM call.

    ``prompt`` / ``completion`` are the standard input/output counts
    every provider reports. ``reasoning`` is the thinking-trace token
    count when the provider exposes it separately (OpenAI o-series,
    Gemini 2.5+); 0 for vLLM thinking models that fold thinking into
    ``completion`` and for non-reasoning models.
    """

    prompt: int
    completion: int
    reasoning: int

    @property
    def is_meaningful(self) -> bool:
        """True when at least one of prompt / completion tokens were spent.

        Used to suppress :meth:`BudgetTracker.record_llm` calls that
        would log a zero-token row — provider responses can omit
        ``usage_metadata`` entirely on certain error paths.
        """
        return bool(self.prompt or self.completion)


def extract_openai_usage(usage: Any) -> TokenUsage:
    """Read ``(prompt, completion, reasoning)`` from an OpenAI-style usage object.

    Covers OpenAI SaaS o-series (reasoning under
    ``completion_tokens_details``) and every OpenAI-compat endpoint
    (vLLM / Grok / Mistral / DeepSeek). Returns zeros when ``usage``
    is None — some streaming / error paths skip the field.
    """
    if usage is None:
        return TokenUsage(0, 0, 0)
    # ``getattr(None, attr, default)`` returns ``default`` so a
    # missing details object collapses through both lookups without a
    # branch.
    details = getattr(usage, "completion_tokens_details", None)
    return TokenUsage(
        prompt=getattr(usage, "prompt_tokens", 0) or 0,
        completion=getattr(usage, "completion_tokens", 0) or 0,
        reasoning=getattr(details, "reasoning_tokens", 0) or 0,
    )


def extract_gemini_usage(usage_metadata: Any) -> TokenUsage:
    """Read ``(prompt, completion, reasoning)`` from a Gemini ``usage_metadata``.

    Field names differ from OpenAI's:
    ``prompt_token_count`` / ``candidates_token_count`` /
    ``thinking_token_count``. The last is populated only for 2.5+
    thinking-capable models.
    """
    if usage_metadata is None:
        return TokenUsage(0, 0, 0)
    return TokenUsage(
        prompt=getattr(usage_metadata, "prompt_token_count", 0) or 0,
        completion=getattr(usage_metadata, "candidates_token_count", 0) or 0,
        reasoning=getattr(usage_metadata, "thinking_token_count", 0) or 0,
    )


def estimate_stub_usage(prompt_text: str, response_text: str) -> TokenUsage:
    """Rough char/4 token estimate for the offline stub client.

    No real model is called; we still feed BudgetTracker something so
    tests that assert non-zero billing don't have to special-case the
    stub. Reasoning is always 0.
    """
    return TokenUsage(
        prompt=len(prompt_text) // _CHARS_PER_TOKEN,
        completion=len(response_text) // _CHARS_PER_TOKEN,
        reasoning=0,
    )
