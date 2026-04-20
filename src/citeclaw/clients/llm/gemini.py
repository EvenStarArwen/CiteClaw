"""Gemini LLM client (uses google-genai SDK natively)."""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from citeclaw.clients.llm._reasoning import gemini_thinking_level
from citeclaw.clients.llm._token_extract import extract_gemini_usage
from citeclaw.clients.llm.base import LLMConfigError, LLMResponse
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.models import BudgetExhaustedError

log = logging.getLogger("citeclaw.llm.gemini")


def _strip_additional_properties(schema: Any) -> Any:
    """Recursively remove ``additionalProperties`` from a JSON schema.

    Gemini's ``response_schema`` validator (Google API) does not accept the
    ``additionalProperties`` keyword — it's strict-OpenAI-only — and rejects
    the whole request with HTTP 400 ``INVALID_ARGUMENT`` if it sees one.
    OpenAI uses the same shared schema (in ``citeclaw.screening.schemas``) so
    we strip the field on the way out to Gemini rather than maintaining a
    second copy of every schema.
    """
    if isinstance(schema, dict):
        return {
            k: _strip_additional_properties(v)
            for k, v in schema.items()
            if k not in (
                "additionalProperties",
                "additional_properties",
                # Not a JSON Schema keyword — our provider-routing
                # sentinel for OpenAI strict mode; irrelevant to
                # Gemini and rejected by its schema validator.
                "_strict_openai",
            )
        }
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


class GeminiClient:
    """LLMClient backed by the native google-genai SDK.

    Accepts optional ``model`` / ``reasoning_effort`` overrides so per-filter
    :class:`LLMFilter` settings can pick a different Gemini model or
    thinking level than the global config.
    """

    supports_logprobs = False

    def __init__(
        self,
        config: Settings,
        budget: BudgetTracker,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._model = model or config.screening_model
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else config.reasoning_effort
        )

    @staticmethod
    def matches(model: str) -> bool:
        return model.startswith("gemini-")

    @retry(
        retry=(retry_if_exception_type(Exception) & retry_if_not_exception_type(BudgetExhaustedError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: logging.getLogger("citeclaw.llm.gemini").warning(
            "Gemini call failed (attempt %d): %s — retrying",
            rs.attempt_number,
            (
                f"{type(rs.outcome.exception()).__name__}: "
                f"{str(rs.outcome.exception())[:400]}"
                if rs.outcome and rs.outcome.exception() is not None
                else "unknown"
            ),
        ),
    )
    def call(
        self,
        system: str,
        user: str,
        *,
        with_logprobs: bool = False,
        category: str = "other",
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._budget.is_exhausted(self._config):
            raise BudgetExhaustedError(f"Budget exhausted: {self._budget.summary()}")

        from google import genai  # type: ignore[import-untyped]
        from google.genai import types  # type: ignore[import-untyped]

        api_key = self._config.gemini_api_key
        if not api_key:
            raise LLMConfigError(f"Model '{self._model}' requires gemini_api_key.")

        client = genai.Client(api_key=api_key)
        gen_config: dict[str, Any] = {
            "temperature": 0.0,
            "system_instruction": system,
        }
        thinking_level = gemini_thinking_level(self._reasoning_effort)
        if thinking_level:
            gen_config["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        # Structured output: request ``application/json`` + the given schema
        # so the decoder is constrained to produce conforming JSON. The
        # google-genai SDK accepts a plain dict here as well as typed schema
        # objects — dict is more portable across SDK versions.
        #
        # Gemini does NOT accept ``additionalProperties`` in the schema (it's
        # an OpenAI-strict-only keyword) and 400s the whole request when it
        # sees one. Strip it on the way out so the shared schema in
        # ``citeclaw.screening.schemas`` works for both providers without
        # maintaining two copies.
        #
        # Polymorphic-args gotcha: schemas marked ``_strict_openai: False``
        # (our supervisor + worker response schemas) have a ``tool_args``
        # object with NO declared properties — intentionally polymorphic
        # across 14 tools. Gemini's response-schema enforcer treats
        # "object with no properties" as "object that must stay empty",
        # and silently emits ``"tool_args": {}`` regardless of what the
        # model's reasoning says it should contain (empirically observed on
        # both ``gemini-3.1-flash-lite-preview`` and
        # ``gemini-3.1-pro-preview`` — ~2000 thinking-tokens of correct
        # strategy, then an empty slot). Skip response_schema entirely in
        # that case; the worker/supervisor JSON parsers are lenient and
        # handle free-form JSON plus fenced blocks.
        if response_schema is not None and self._config.structured_output_enabled:
            polymorphic_args = response_schema.get("_strict_openai") is False
            if polymorphic_args:
                # Force JSON output but skip the schema — see the
                # polymorphic-args comment above. Without mime_type the
                # thinking-mode model will emit all 2K+ tokens as
                # ``thought`` parts with nothing in ``text`` (observed on
                # gemini-3.1-pro-preview), collapsing the response to
                # empty string on the client side.
                gen_config["response_mime_type"] = "application/json"
            else:
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = _strip_additional_properties(response_schema)

        resp = client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(**gen_config),
        )
        parts = (resp.candidates[0].content.parts if resp.candidates else []) or []
        text_parts = [p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)]
        text = "\n".join(text_parts) if text_parts else (getattr(resp, "text", "") or "")

        usage = extract_gemini_usage(getattr(resp, "usage_metadata", None))
        if usage.is_meaningful:
            self._budget.record_llm(
                usage.prompt, usage.completion, category,
                reasoning_tokens=usage.reasoning,
                model=self._model,
            )
        return LLMResponse(text=text, logprob_tokens=[])
