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


# Default thinking-token budgets per effort level.  These prevent OSS
# reasoning models (Gemma 4, Qwen3, DeepSeek-R1) from spending 60–100 K
# tokens on a single call while still giving them useful thinking room.
# Commercial models (o3, Claude) handle this internally; the budget is
# only sent to vLLM / custom endpoints via ``thinking_token_budget``.
_EFFORT_THINKING_BUDGET: dict[str, int] = {
    "high": 16384,
    "medium": 8192,
    "low": 4096,
}

# Hard ceiling on total completion (thinking + content).  The soft
# ``thinking_budget`` in chat_template_kwargs is only a hint that the
# model can ignore; ``max_completion_tokens`` is enforced by vLLM's
# sampler and will truncate generation at this limit.  We use 3× the
# thinking budget: empirically Gemma 4 can exceed the soft hint by
# 2–3× on complex prompts, and we need headroom for the actual
# content output (structured JSON).  3× prevents 100K+ runaway
# (observed on 128K context) while rarely truncating useful output.
_COMPLETION_HEADROOM_FACTOR = 3.0


def _custom_endpoint_reasoning_kwargs(
    reasoning_effort: str,
    thinking_budget: int = 0,
) -> dict[str, Any]:
    """Map ``reasoning_effort`` to OSS chat-template kwargs.

    Two interlocked knobs are required for Gemma 4 thinking mode to
    actually surface clean content + separate reasoning:

    1. ``chat_template_kwargs.enable_thinking=True`` tells the chat
       template to inject the ``<|think|>`` capability marker so the
       model is allowed to emit a thinking block.

    2. ``skip_special_tokens=False`` tells vLLM's tokenizer NOT to
       strip the ``<|channel>`` / ``<channel|>`` thinking-block
       delimiters during decode. Without this flag, the markers vanish
       from the response text and vLLM's ``gemma4`` reasoning parser
       can no longer find them — every thinking trace then leaks into
       ``message.content`` as a raw text block starting with
       ``thought\\n`` while ``message.reasoning`` stays None.
       Empirically verified against the live Modal Gemma 4 31B endpoint:
       with the flag, content is the clean answer (e.g. ``"YES"``) and
       reasoning is the full thinking trace; without it, content is the
       polluted blob.

    The flag is harmless when thinking is OFF (the model just doesn't
    emit any special tokens to keep), so we set it whenever the
    reasoning_effort knob is touched at all — including the explicit
    "off" path, since some chat templates inject an empty
    ``<|channel>thought\\n<channel|>`` placeholder there too.

    ``thinking_budget`` (from ``ModelEndpoint.thinking_budget``)
    overrides the default effort-based cap. ``0`` means use the default.
    """
    effort = (reasoning_effort or "").strip().lower()
    if not effort:
        return {}
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
        # See docstring above — must be False to keep the channel markers
        # visible in the response so the reasoning parser can split.
        "skip_special_tokens": False,
    }
    if effort in ("off", "none", "false", "disable", "disabled"):
        return {"extra_body": extra_body}
    extra_body["chat_template_kwargs"]["enable_thinking"] = True
    # Cap reasoning tokens to prevent runaway thinking.  Passed as
    # ``thinking_budget`` inside ``chat_template_kwargs`` — this is a
    # Gemma 4 chat-template feature that limits the thinking trace
    # without requiring vLLM's ``--reasoning-config`` server flag.
    # NOTE: this is a *soft hint* — the model may exceed it.  The hard
    # ceiling is ``max_completion_tokens`` set in the returned dict.
    budget = thinking_budget or _EFFORT_THINKING_BUDGET.get(effort, 16384)
    extra_body["chat_template_kwargs"]["thinking_budget"] = budget
    return {
        "reasoning_effort": effort,
        "extra_body": extra_body,
        # Hard ceiling: thinking budget + headroom for content output.
        "max_completion_tokens": int(budget * _COMPLETION_HEADROOM_FACTOR),
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
        thinking_budget: int = 0,
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
            kwargs.update(_custom_endpoint_reasoning_kwargs(
                self._reasoning_effort, self._thinking_budget,
            ))
        elif self._is_reasoning and self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        # Structured output: when a schema is provided *and* the operator
        # hasn't disabled it, pass ``response_format={"type": "json_schema", ...}``
        # so the model is constrained at decode time instead of relying on
        # post-hoc JSON parsing.
        #
        # EXCEPTION: skip structured output when the custom endpoint has
        # thinking enabled.  vLLM's guided decoding interacts badly with
        # reasoning-mode generation — the guided decoder counts rejected
        # candidate tokens toward ``max_completion_tokens``, causing the
        # model to exhaust its budget on decoding overhead and truncate
        # the actual JSON output.  The model is perfectly capable of
        # producing valid JSON without guided decoding; the JSON salvage
        # code in ``pdf_reference_extractor`` handles edge cases.
        thinking_active = (
            self._is_custom
            and self._reasoning_effort
            and self._reasoning_effort.strip().lower()
            not in ("off", "none", "false", "disable", "disabled", "")
        )
        if (
            response_schema is not None
            and self._config.structured_output_enabled
            and not thinking_active
        ):
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
        # PH-07: vLLM's reasoning parser exposes the thinking trace on
        # the message under either ``reasoning`` (Gemma 4) or
        # ``reasoning_content`` (Qwen3 / DeepSeek-R1) depending on the
        # parser. Read both and surface whichever is non-empty so
        # downstream callers can introspect the model's chain of
        # thought without re-parsing the raw response. The presence of
        # this field is the canonical signal that the parser worked
        # and ``content`` is the clean answer.
        reasoning_content = ""
        msg = resp.choices[0].message
        for attr in ("reasoning", "reasoning_content"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val:
                reasoning_content = val
                break
        if self._is_custom:
            # Defensive strip: when the parser worked, content is clean
            # already. The strip is a no-op in that case but still
            # protects against legacy ``<think>...</think>``-style
            # content from older Qwen3 deploys that don't have the
            # reasoning parser configured.
            text = _strip_think_tags(text)
        if with_logprobs and not self._is_reasoning:
            lp = resp.choices[0].logprobs
            return LLMResponse(
                text=text,
                logprob_tokens=lp.content if lp else [],
                reasoning_content=reasoning_content,
            )
        return LLMResponse(
            text=text,
            logprob_tokens=[],
            reasoning_content=reasoning_content,
        )
