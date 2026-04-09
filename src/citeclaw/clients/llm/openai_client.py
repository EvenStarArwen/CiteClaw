"""OpenAI / OpenAI-compatible LLM client (covers OpenAI SaaS, vLLM, Ollama, ...)."""

from __future__ import annotations

import logging
import re
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from citeclaw.clients.llm.base import LLMResponse
from citeclaw.config import BudgetTracker, Settings
from citeclaw.models import BudgetExhaustedError

log = logging.getLogger("citeclaw.llm.openai")

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove leftover ``<think>...</think>`` blocks from a response."""
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def _custom_endpoint_reasoning_kwargs(reasoning_effort: str) -> dict[str, Any]:
    """Map ``reasoning_effort`` to OSS chat-template kwargs."""
    effort = (reasoning_effort or "").strip().lower()
    if not effort:
        return {}
    if effort in ("off", "none", "false", "disable", "disabled"):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {
        "reasoning_effort": effort,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def _build_openai_sdk(
    config: Settings,
    *,
    endpoint_base_url: str | None = None,
    endpoint_api_key: str | None = None,
    endpoint_timeout: float | None = None,
) -> openai.OpenAI:
    """Construct an ``openai.OpenAI`` SDK client.

    The optional ``endpoint_*`` overrides win over the global
    ``llm_base_url`` / ``llm_api_key`` fields. They are how the
    per-model registry (``Settings.models``) routes a single config
    file at multiple OpenAI-compatible endpoints — one OpenAIClient
    instance per registry alias, each with its own SDK client. The
    legacy ``config.llm_base_url`` path is preserved unchanged when
    no overrides are passed.
    """
    if endpoint_base_url:
        api_key = endpoint_api_key or config.llm_api_key or config.openai_api_key or "none"
        return openai.OpenAI(
            api_key=api_key,
            base_url=endpoint_base_url.rstrip("/"),
            timeout=endpoint_timeout if endpoint_timeout is not None else config.llm_request_timeout,
        )
    if config.llm_base_url:
        api_key = config.llm_api_key or config.openai_api_key or "none"
        return openai.OpenAI(
            api_key=api_key,
            base_url=config.llm_base_url.rstrip("/"),
            timeout=config.llm_request_timeout,
        )
    api_key = config.openai_api_key
    if not api_key:
        raise ValueError(
            f"Model '{config.screening_model}' requires openai_api_key "
            f"(or set llm_base_url for a custom OpenAI-compatible endpoint). "
            f"Set it in config.yaml or OPENAI_API_KEY / CITECLAW_OPENAI_API_KEY env."
        )
    return openai.OpenAI(api_key=api_key, timeout=60.0)


class OpenAIClient:
    """LLMClient for OpenAI SaaS and OpenAI-compatible endpoints (vLLM/Ollama/...).

    Accepts optional ``model`` / ``reasoning_effort`` overrides so per-filter
    settings on ``LLMFilter`` can pick a different model than the global
    ``screening_model``. The underlying ``openai.OpenAI`` SDK client depends
    only on the endpoint (base_url + api_key), not the model — so one client
    instance can serve multiple models if needed. ``_is_reasoning`` is
    derived from the resolved model, not the global config, so overriding
    a base ``gpt-4o`` client to ``o3-mini`` still takes the reasoning path.

    Per-instance endpoint overrides
    -------------------------------
    The four optional ``endpoint_*`` / ``served_model_name`` kwargs are used
    by the registry-routing path in ``factory.build_llm_client``. When the
    YAML alias resolves to an entry in ``Settings.models``, the factory
    constructs one OpenAIClient per alias, each pointed at the alias's own
    OpenAI-compatible endpoint and asked to send the alias's
    ``served_model_name`` over the wire. The ``model`` field on this
    instance stays as the YAML alias (used for budget bookkeeping and
    pricing), while the SDK call uses ``served_model_name``. When the
    overrides are absent, behaviour is identical to the legacy path.
    """

    def __init__(
        self,
        config: Settings,
        budget: BudgetTracker,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
        endpoint_timeout: float | None = None,
        served_model_name: str | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._model = model or config.screening_model
        # ``_served_model_name`` is what we send over the wire as the
        # OpenAI ``model`` field; the YAML alias stays in ``self._model``
        # so the budget tracker / pricing table see a stable name.
        self._served_model_name = served_model_name or self._model
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else config.reasoning_effort
        )
        # Custom-endpoint paths (registry alias OR legacy llm_base_url) skip
        # the OpenAI o-series reasoning detection — those endpoints host OSS
        # models that take ``chat_template_kwargs`` instead of
        # ``reasoning_effort=...``. The registry path is detected by the
        # presence of ``endpoint_base_url``; the legacy path by
        # ``config.llm_base_url``.
        self._is_custom = bool(endpoint_base_url) or bool(config.llm_base_url)
        self._is_reasoning = (not self._is_custom) and any(
            self._model.startswith(p) for p in _OPENAI_REASONING_PREFIXES
        )
        self._sdk = _build_openai_sdk(
            config,
            endpoint_base_url=endpoint_base_url,
            endpoint_api_key=endpoint_api_key,
            endpoint_timeout=endpoint_timeout,
        )

    @property
    def supports_logprobs(self) -> bool:
        # OpenAI reasoning models reject logprobs; everything else (incl. vLLM) supports them.
        return not self._is_reasoning

    @retry(
        retry=(retry_if_exception_type(Exception) & retry_if_not_exception_type(BudgetExhaustedError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: logging.getLogger("citeclaw.llm.openai").warning(
            "LLM call failed (attempt %d), retrying...", rs.attempt_number
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

        kwargs: dict[str, Any] = dict(
            model=self._served_model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
            if with_logprobs:
                kwargs["logprobs"] = True
        if self._is_custom:
            kwargs.update(_custom_endpoint_reasoning_kwargs(self._reasoning_effort))
        elif self._is_reasoning and self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        # Structured output: when a schema is provided *and* the operator
        # hasn't disabled it, pass ``response_format={"type": "json_schema", ...}``
        # so the model is constrained at decode time instead of relying on
        # post-hoc JSON parsing. Applies to both SaaS OpenAI and custom
        # endpoints (vLLM/Ollama/etc.) — the kill switch
        # ``structured_output_enabled`` in Settings lets users opt out of
        # the latter if their endpoint doesn't honor the flag.
        if response_schema is not None and self._config.structured_output_enabled:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "citeclaw_screening_results",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        resp = self._sdk.chat.completions.create(**kwargs)
        usage = resp.usage
        if usage:
            self._budget.record_llm(
                usage.prompt_tokens,
                usage.completion_tokens,
                category,
                reasoning_tokens=_extract_reasoning_tokens(usage),
            )
        text = resp.choices[0].message.content or ""
        if self._is_custom:
            text = _strip_think_tags(text)
        if with_logprobs and not self._is_reasoning:
            lp = resp.choices[0].logprobs
            return LLMResponse(text=text, logprob_tokens=lp.content if lp else [])
        return LLMResponse(text=text, logprob_tokens=[])
