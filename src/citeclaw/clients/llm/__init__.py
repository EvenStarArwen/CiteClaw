"""LLM client package — Protocol + concrete providers (OpenAI, Gemini, Stub)."""

from __future__ import annotations

from citeclaw.clients.llm.base import LLMClient, LLMResponse
from citeclaw.clients.llm.factory import build_llm_client, supports_logprobs
from citeclaw.clients.llm.gemini import GeminiClient
from citeclaw.clients.llm.openai_client import OpenAIClient
from citeclaw.clients.llm.stub import StubClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "GeminiClient",
    "StubClient",
    "build_llm_client",
    "supports_logprobs",
]
