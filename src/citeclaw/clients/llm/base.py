"""LLMClient Protocol and small response container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Single LLM call result.

    ``text`` is the clean assistant answer (the OpenAI ``message.content``
    field for chat models). ``reasoning_content`` is the model's thinking
    trace, when the provider exposes it as a separate field — vLLM with
    a working reasoning parser populates ``message.reasoning``, OpenAI's
    o-series populates ``completion_tokens_details``, and Gemini 2.5/3
    surfaces it via ``thinking`` parts on the response. Empty string when
    the model didn't think or the provider didn't expose it.

    ``logprob_tokens`` is empty for providers that don't expose logprobs
    (Gemini, reasoning models, stub).
    """

    text: str
    logprob_tokens: list[Any] = field(default_factory=list)
    reasoning_content: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat-style LLM client.

    All concrete clients (OpenAIClient, GeminiClient, StubClient) implement
    this single ``call`` method. The screener / reranker / annotator never
    type-check the provider — they always call ``client.call(...)``.

    ``response_schema`` is an optional JSON Schema dict that the concrete
    client translates into provider-native structured-output constraints
    (OpenAI ``response_format={"type": "json_schema", ...}``; Gemini
    ``response_schema`` + ``response_mime_type="application/json"``). The
    stub ignores it (its output is already well-formed). Clients that
    don't support structured output silently fall back to free-form text.
    """

    def call(
        self,
        system: str,
        user: str,
        *,
        with_logprobs: bool = False,
        category: str = "other",
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    @property
    def supports_logprobs(self) -> bool: ...
