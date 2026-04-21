"""OpenAI / OpenAI-compatible LLM client (covers OpenAI SaaS, vLLM, Ollama, ...)."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from citeclaw.clients.llm._reasoning import (
    custom_endpoint_reasoning_kwargs,
    is_thinking_active,
)
from citeclaw.clients.llm._schema import pop_strict_openai
from citeclaw.clients.llm._token_extract import extract_openai_usage
from citeclaw.clients.llm.base import LLMConfigError, LLMResponse
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.models import BudgetExhaustedError

log = logging.getLogger("citeclaw.llm.openai")

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove leftover ``<think>...</think>`` blocks from a response."""
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_reasoning_content(message: Any) -> str:
    """Pick the non-empty thinking-trace string off a vLLM message.

    Gemma 4's reasoning parser exposes the trace under ``message.reasoning``;
    Qwen3 / DeepSeek-R1 use ``message.reasoning_content``. Returns the
    first non-empty string; ``""`` when neither attribute is present or set.
    """
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


def _build_response_format(
    response_schema: dict[str, Any] | None,
    *,
    structured_output_enabled: bool,
    thinking_active: bool,
) -> dict[str, Any] | None:
    """Build the ``response_format`` kwarg, or ``None`` to omit it.

    Returns ``None`` (caller skips the kwarg) when (a) no schema was
    provided, (b) the user disabled structured output globally, or (c)
    vLLM thinking mode is active — guided decoding's rejected candidates
    count against ``max_completion_tokens`` and exhaust the budget on
    decode overhead. Native-reasoning providers (OpenAI o-series, Grok,
    Mistral) handle structured output during thinking fine.
    """
    if response_schema is None or not structured_output_enabled or thinking_active:
        return None
    schema_clean, strict = pop_strict_openai(response_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "citeclaw_screening_results",
            "strict": strict,
            "schema": schema_clean,
        },
    }


def _log_retry(retry_state: Any) -> None:
    """tenacity ``before_sleep`` hook: warn-log the failure + attempt count."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    detail = (
        f"{type(exc).__name__}: {str(exc)[:400]}"
        if exc is not None
        else "unknown"
    )
    log.warning(
        "LLM call failed (attempt %d): %s — retrying",
        retry_state.attempt_number,
        detail,
    )


def _build_timeout(total: float) -> httpx.Timeout:
    """Build an httpx.Timeout with explicit per-phase bounds.

    Passing a single float to ``openai.OpenAI(timeout=…)`` maps to
    ``httpx.Timeout(total, connect=total, read=total, write=total,
    pool=total)``. In practice httpx's pool / connect phases are fine,
    but the READ phase — how long we wait between response bytes — is
    what actually gates long vLLM reasoning traces. On Modal's
    streaming-via-keepalive, the socket can dribble bytes indefinitely
    without triggering the read timeout. Observed iter-12 and iter-15:
    20+ min stalls on a single call despite ``timeout=300``.

    Setting ``read=total`` but keeping connect/write/pool tight means
    a hung call still reaches a total limit, without breaking happy-
    path long-reasoning responses that legitimately take minutes.
    """
    return httpx.Timeout(
        timeout=total,
        connect=min(30.0, total),
        read=total,
        write=min(30.0, total),
        pool=min(30.0, total),
    )


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
        total = endpoint_timeout if endpoint_timeout is not None else config.llm_request_timeout
        return openai.OpenAI(
            api_key=api_key,
            base_url=endpoint_base_url.rstrip("/"),
            timeout=_build_timeout(total),
        )
    if config.llm_base_url:
        api_key = config.llm_api_key or config.openai_api_key or "none"
        return openai.OpenAI(
            api_key=api_key,
            base_url=config.llm_base_url.rstrip("/"),
            timeout=_build_timeout(config.llm_request_timeout),
        )
    api_key = config.openai_api_key
    if not api_key:
        raise LLMConfigError(
            f"Model '{config.screening_model}' requires openai_api_key "
            f"(or set llm_base_url for a custom OpenAI-compatible endpoint). "
            f"Set it in config.yaml or OPENAI_API_KEY / CITECLAW_OPENAI_API_KEY env."
        )
    return openai.OpenAI(api_key=api_key, timeout=_build_timeout(60.0))


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
        thinking_budget: int = 0,
        reasoning_parser: str = "",
        max_model_len: int = 0,
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
        self._thinking_budget = thinking_budget
        self._reasoning_parser = reasoning_parser or ""
        # Server-side context-window hint from ``ModelEndpoint.max_model_len``.
        # Used by :func:`_custom_endpoint_reasoning_kwargs` to clamp the
        # ``max_completion_tokens`` the SDK sends so vLLM doesn't reject
        # the request with ``max_completion_tokens > max_model_len``.
        self._max_model_len = max_model_len
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
        # Stop on whichever fires first: 6 attempts OR 10 minutes total
        # wall clock. The per-call SDK timeout (``llm_request_timeout``,
        # default 300s) bounds each attempt; without a total-delay cap, a
        # stuck endpoint can eat the full 6 × 300s = 30+ min budget through
        # retries while the worker's turn counter doesn't advance.
        # Observed iter-12 (data_loop_weather worker 2) stalled 20+ min on
        # one turn before the bash-level kill.
        stop=(stop_after_attempt(6) | stop_after_delay(600)),
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
            kwargs.update(custom_endpoint_reasoning_kwargs(
                self._reasoning_effort,
                reasoning_parser=self._reasoning_parser,
                thinking_budget=self._thinking_budget,
                max_model_len=self._max_model_len,
            ))
        elif self._is_reasoning and self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        thinking_active = self._is_custom and is_thinking_active(
            self._reasoning_effort,
            reasoning_parser=self._reasoning_parser,
        )
        # ``pop_strict_openai`` (inside the helper) consumes the routing
        # sentinel — see _schema.py. Polymorphic schemas (tool_args open
        # objects) opt out of OpenAI strict mode, otherwise
        # additionalProperties=false would have to be threaded through
        # every nested object.
        response_format = _build_response_format(
            response_schema,
            structured_output_enabled=self._config.structured_output_enabled,
            thinking_active=thinking_active,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self._sdk.chat.completions.create(**kwargs)
        usage = extract_openai_usage(resp.usage)
        if usage.is_meaningful:
            self._budget.record_llm(
                usage.prompt,
                usage.completion,
                category,
                reasoning_tokens=usage.reasoning,
                model=self._model,
            )
        choice = resp.choices[0]
        text = choice.message.content or ""
        reasoning_content = _extract_reasoning_content(choice.message)
        if self._is_custom:
            # Defensive strip: when the parser worked, content is clean
            # already. The strip is a no-op in that case but still
            # protects against legacy ``<think>...</think>``-style
            # content from older Qwen3 deploys that don't have the
            # reasoning parser configured.
            text = _strip_think_tags(text)
        logprob_tokens: list[Any] = []
        if with_logprobs and not self._is_reasoning:
            lp = choice.logprobs
            logprob_tokens = lp.content if lp else []
        return LLMResponse(
            text=text,
            logprob_tokens=logprob_tokens,
            reasoning_content=reasoning_content,
        )
