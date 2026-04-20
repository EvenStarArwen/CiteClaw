"""Centralized reasoning-effort dispatch for every LLM provider.

One module owns the mapping from CiteClaw's effort labels (``low`` /
``medium`` / ``high`` / ``minimal``) to the wire shape each provider
expects. Adding a new provider means adding one branch here, not
threading the rules through factory + client + caching layers.

Wire shapes today:

* OpenAI o-series, xAI Grok, Mistral Magistral, DeepSeek-reasoner via
  OpenAI-compat → ``reasoning_effort=<str>`` native kwarg.
* vLLM Gemma 4 / Qwen3 / DeepSeek-R1 → ``extra_body`` with
  ``chat_template_kwargs.{enable_thinking, thinking_budget}`` and
  ``skip_special_tokens=False``, plus ``max_completion_tokens``.
* Gemini 2.5 / 3.x → ``types.ThinkingConfig(thinking_level=<str>)``
  inside the SDK config (the SDK shape can't be expressed as plain
  kwargs, so the Gemini client calls :func:`gemini_thinking_level`
  directly and constructs the typed object itself).
* Models that don't support reasoning at all → empty dict (passing
  no kwargs lets the model run without 400-ing).
"""

from __future__ import annotations

from typing import Any

EFFORT_LEVELS = ("low", "medium", "high", "minimal")
# Explicit-off synonyms collapse to the canonical OFF_LABEL — distinct
# from "user didn't pass anything" (empty string) so the vLLM dispatch
# can still emit ``skip_special_tokens=False`` on the off path while
# truly-unset stays a complete no-op.
OFF_LABEL = "off"
OFF_SYNONYMS = frozenset({"off", "none", "false", "disable", "disabled"})

# Default thinking-token budgets for vLLM-hosted reasoning models. These
# prevent OSS reasoning runs from spending 60-100K tokens per call. The
# 3x headroom factor below leaves room for the actual content output
# (typically structured JSON of 1-3K tokens) on top of the thinking.
VLLM_EFFORT_BUDGETS: dict[str, int] = {
    "high": 16384,
    "medium": 8192,
    "low": 4096,
}
COMPLETION_HEADROOM_FACTOR = 3.0
# vLLM rejects with HTTP 400 if ``max_completion_tokens >= max_model_len``
# (it has to leave room for the prompt).
INPUT_RESERVE_TOKENS = 8192

# ``ModelEndpoint.reasoning_parser`` discriminator buckets.
VLLM_PARSERS = frozenset({"", "vllm", "gemma4", "qwen3", "deepseek_r1", "deepseek-r1"})
NATIVE_REASONING_PARSERS = frozenset({"openai", "grok", "xai", "mistral", "magistral"})
NO_REASONING_PARSERS = frozenset({"none", "off", "disabled"})


def normalize_effort(effort: str | None) -> str:
    """Lower-case + strip. Returns:

    * ``""`` — user passed nothing (or only whitespace).
    * ``OFF_LABEL`` (``"off"``) — user explicitly disabled reasoning.
    * the lowercased label otherwise (``low`` / ``medium`` / ``high`` / ``minimal``).
    """
    s = (effort or "").strip().lower()
    if not s:
        return ""
    if s in OFF_SYNONYMS:
        return OFF_LABEL
    return s


def gemini_thinking_level(effort: str | None) -> str | None:
    """Effort label → Gemini ``thinking_level`` string. ``None`` means don't set."""
    e = normalize_effort(effort)
    return e if e in EFFORT_LEVELS else None


def openai_native_reasoning_kwargs(effort: str | None) -> dict[str, str]:
    """Effort label → ``reasoning_effort=<str>`` for OpenAI o-series + Grok / Mistral.

    Empty dict when effort is unset OR explicitly off, so callers can
    ``kwargs.update(...)`` unconditionally and the model only sees the
    kwarg when it's meaningful.
    """
    e = normalize_effort(effort)
    return {"reasoning_effort": e} if e and e != OFF_LABEL else {}


def vllm_reasoning_kwargs(
    effort: str | None,
    *,
    thinking_budget: int = 0,
    max_model_len: int = 0,
) -> dict[str, Any]:
    """Effort label → vLLM chat-template kwargs for Gemma 4 / Qwen3 / DeepSeek-R1.

    Two interlocked knobs are required for vLLM thinking mode to surface
    clean content + a separately-readable reasoning trace:

    1. ``chat_template_kwargs.enable_thinking=True`` lets the chat
       template inject the thinking-block markers.
    2. ``skip_special_tokens=False`` keeps those markers in the
       decoded response so vLLM's ``reasoning_parser`` can split
       ``content`` from the thinking trace.

    Without (2), every thinking trace leaks into ``message.content``
    and ``message.reasoning`` stays None — empirically observed against
    Modal-hosted Gemma 4 31B. The flag is harmless when thinking is
    off (no special tokens to keep), so it's set whenever
    ``reasoning_effort`` is touched at all.

    ``thinking_budget`` overrides the per-effort default; ``0`` keeps
    the default. ``max_model_len`` clamps ``max_completion_tokens``
    so the request doesn't 400 on prompt + completion overflow.
    """
    e = normalize_effort(effort)
    if not e:
        return {}
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
    }
    if e == OFF_LABEL:
        return {"extra_body": extra_body}
    extra_body["chat_template_kwargs"]["enable_thinking"] = True
    budget = thinking_budget or VLLM_EFFORT_BUDGETS.get(e, 16384)
    desired_completion = int(budget * COMPLETION_HEADROOM_FACTOR)
    if max_model_len > 0:
        safe_ceiling = max(1024, max_model_len - INPUT_RESERVE_TOKENS)
        if desired_completion > safe_ceiling:
            desired_completion = safe_ceiling
            budget = min(budget, int(desired_completion / COMPLETION_HEADROOM_FACTOR))
    extra_body["chat_template_kwargs"]["thinking_budget"] = budget
    return {
        "reasoning_effort": e,
        "extra_body": extra_body,
        "max_completion_tokens": desired_completion,
    }


def custom_endpoint_reasoning_kwargs(
    effort: str | None,
    *,
    reasoning_parser: str = "",
    thinking_budget: int = 0,
    max_model_len: int = 0,
) -> dict[str, Any]:
    """Dispatch over ``reasoning_parser`` to the right wire shape.

    See module docstring for the parser → shape mapping. Returns ``{}``
    when the endpoint explicitly opts out of reasoning so plain
    OpenAI-compatible models (e.g. Together AI Llama-3) don't receive
    kwargs they reject.
    """
    parser = (reasoning_parser or "").strip().lower()
    if parser in NO_REASONING_PARSERS:
        return {}
    if parser in NATIVE_REASONING_PARSERS:
        return openai_native_reasoning_kwargs(effort)
    return vllm_reasoning_kwargs(
        effort, thinking_budget=thinking_budget, max_model_len=max_model_len,
    )


def is_thinking_active(
    effort: str | None,
    *,
    reasoning_parser: str = "",
) -> bool:
    """True when the request will actually send thinking-on kwargs.

    Used by ``OpenAIClient`` to decide whether to skip the OpenAI
    structured-output response_format — vLLM's guided decoder interacts
    badly with thinking mode (rejected candidate tokens count toward
    ``max_completion_tokens``, exhausting the budget on decode overhead).
    Only the vLLM path triggers this; native-reasoning providers handle
    structured output correctly during thinking.
    """
    e = normalize_effort(effort)
    if not e or e == OFF_LABEL:
        return False
    parser = (reasoning_parser or "").strip().lower()
    return parser in VLLM_PARSERS
