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
from citeclaw.clients.llm._schema import strip_for_gemini
from citeclaw.clients.llm._token_extract import extract_gemini_usage
from citeclaw.clients.llm.base import LLMConfigError, LLMResponse
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.models import BudgetExhaustedError

log = logging.getLogger("citeclaw.llm.gemini")


def _log_retry(retry_state: Any) -> None:
    """tenacity ``before_sleep`` hook: warn-log the failure + attempt count."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    detail = (
        f"{type(exc).__name__}: {str(exc)[:400]}"
        if exc is not None
        else "unknown"
    )
    log.warning(
        "Gemini call failed (attempt %d): %s — retrying",
        retry_state.attempt_number,
        detail,
    )


def _extract_text_from_parts(resp: Any) -> str:
    """Join the non-thought ``text`` parts on the first candidate.

    Gemini 2.5/3 thinking-mode responses interleave ``thought=True`` parts
    (the chain-of-reasoning) with ``thought=False`` parts (the user-facing
    answer). We deliberately drop the thought parts so the returned
    ``LLMResponse.text`` is the clean answer; the thinking trace itself
    is currently NOT surfaced via ``reasoning_content`` (a known gap —
    the parts list is walked here but the dropped trace isn't re-attached).
    Falls back to ``resp.text`` when no parts came through.
    """
    parts = (resp.candidates[0].content.parts if resp.candidates else []) or []
    text_parts = [
        p.text for p in parts
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    ]
    if text_parts:
        return "\n".join(text_parts)
    return getattr(resp, "text", "") or ""


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
        """True for the ``gemini-*`` model alias family routed to this client."""
        return model.startswith("gemini-")

    @retry(
        retry=(retry_if_exception_type(Exception) & retry_if_not_exception_type(BudgetExhaustedError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=_log_retry,
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
            # Always set ``application/json`` — without it the thinking-mode
            # model emits all 2K+ tokens as ``thought`` parts with nothing
            # in ``text`` (observed on gemini-3.1-pro-preview), collapsing
            # the response to empty string on the client side. The schema
            # itself is added only when not polymorphic — see the
            # polymorphic-args comment above for why.
            gen_config["response_mime_type"] = "application/json"
            polymorphic_args = response_schema.get("_strict_openai") is False
            if not polymorphic_args:
                gen_config["response_schema"] = strip_for_gemini(response_schema)

        resp = client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(**gen_config),
        )
        text = _extract_text_from_parts(resp)

        usage = extract_gemini_usage(getattr(resp, "usage_metadata", None))
        if usage.is_meaningful:
            self._budget.record_llm(
                usage.prompt, usage.completion, category,
                reasoning_tokens=usage.reasoning,
                model=self._model,
            )
        # ``reasoning_content=""`` is the default but spelled out here to
        # signal the deliberate gap: ``_extract_text_from_parts`` drops the
        # thought parts rather than re-attaching them.
        return LLMResponse(text=text, logprob_tokens=[], reasoning_content="")
