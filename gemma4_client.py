"""Reusable Python client for Gemma 4 endpoints (vLLM-served, OpenAI-compatible).

Why this module exists
----------------------
Gemma 4 has several per-request quirks that every client has to handle the
same way. Doing it once here lets any project just
``from gemma4_client import Gemma4Client`` and skip the footguns:

* **Thinking-mode toggle** via ``chat_template_kwargs.enable_thinking``
  (the Qwen3 pattern that vLLM extended to Gemma 4). Without this, the
  model can enter thinking mode for trivial queries and burn 1000s of
  tokens on "let me reconsider…".

* **Thinking-content stripping.** Even when ``--reasoning-parser gemma4``
  is enabled server-side, vLLM issue #38855 means Gemma 4's reasoning
  parser does NOT populate ``message.reasoning_content`` — all thinking
  content leaks into ``content``. We pass ``skip_special_tokens=False``
  so the ``<|channel>thought ... <channel|>`` boundary tokens survive
  into the response text, then strip them with regex. A bare-leak fallback
  (when ``skip_special_tokens=True`` is forced) extracts the last
  paragraph as the final answer.

* **Multi-turn history.** Per Google's Gemma 4 model card, prior-turn
  thinking content MUST NOT be re-sent to the assistant role; otherwise
  the model feedback-loops and thinking grows exponentially. ``ChatSession``
  strips automatically before resending.

* **Recommended sampling.** Google's card specifies
  ``temperature=1.0, top_p=0.95, top_k=64``. Lowering temperature can
  make thinking more repetitive, not less. We default to those values.

* **Modal HTTP 303 redirects.** Modal async-dispatches under load; httpx
  defaults to *not* following redirects on POST. We configure
  ``follow_redirects=True`` so retries hit the right backend.

This file is intentionally STANDALONE — it depends only on ``openai``
and ``httpx``, never imports from CiteClaw. It complements (does not
replace) ``citeclaw.clients.llm.openai_client.OpenAIClient``: the
production path routes through the registry/factory; this client is
for diagnostics, one-offs, and anywhere outside CiteClaw that wants a
clean Gemma 4 client.

Quickstart
----------
Pre-configured factories for the cola-lab Modal endpoints::

    from gemma4_client import Gemma4Client

    client = Gemma4Client.fp4()        # NVFP4 deploy on H200/B200
    print(client.chat("Hello!"))

    client = Gemma4Client.bf16()       # BF16 v2 deploy on H200
    print(client.chat("Hello!"))

Or pass an explicit endpoint::

    client = Gemma4Client(
        base_url="https://my-endpoint/v1",
        api_key="my-key",
        model="google/gemma-4-31B-it",
    )

Single-turn::

    print(client.chat(
        "Solve: 17 * 23",
        system="You are concise.",
        thinking="always",
        thinking_budget=200,
    ))

Multi-turn (history is automatically stripped of thinking)::

    sess = client.session(system="You are concise.", thinking="always")
    print(sess.send("17 * 23?"))
    print(sess.send("Doubled?"))
    print(sess.last_result.usage)        # token counts of the last call

Lower-level (raw messages, full control)::

    result = client.complete(
        messages=[{"role": "user", "content": "Hi"}],
        thinking="never",
        max_tokens=128,
    )
    result.content       # final text, thinking stripped
    result.raw_content   # raw text as returned (for debugging)
    result.reasoning     # message.reasoning_content (None if absent)
    result.usage         # {prompt_tokens, completion_tokens, elapsed_s, ...}

Per-call sampling override::

    client.chat("Hi", temperature=0.5, top_p=0.9, top_k=20, max_tokens=64)

Endpoints
---------
``ENDPOINTS`` exposes the cola-lab Modal deploys keyed by short name. For
other Gemma 4 variants (FP8, AWQ, Gemma 4 26B-A4B), pass an explicit
``base_url`` + ``model`` to ``Gemma4Client(...)``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

try:
    import httpx
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "gemma4_client requires `openai` and `httpx`. "
        "Install with: pip install 'openai>=1.0' httpx"
    ) from e


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_API_KEY = "citeclaw-test-key"

#: Pre-configured cola-lab Modal endpoints. ``Gemma4Client.fp4()`` /
#: ``Gemma4Client.bf16()`` use these.
ENDPOINTS: dict[str, dict[str, str]] = {
    "fp4": {
        "base_url": "https://cola-lab--citeclaw-vllm-gemma-fp4-serve.modal.run/v1",
        "model": "nvidia/Gemma-4-31B-IT-NVFP4",
    },
    "bf16": {
        "base_url": "https://cola-lab--citeclaw-vllm-gemma-v2-serve.modal.run/v1",
        "model": "google/gemma-4-31B-it",
    },
}

#: Google's recommended sampling parameters from the Gemma 4 model card.
GEMMA_DEFAULT_TEMPERATURE = 1.0
GEMMA_DEFAULT_TOP_P = 0.95
GEMMA_DEFAULT_TOP_K = 64

#: System-prompt sentence that bounds thinking length. Plain English; the
#: model honours it loosely but consistently in practice.
THINKING_BUDGET_HINT_TEMPLATE = (
    "Keep your internal reasoning concise — aim for under {n} thinking "
    "tokens before giving the final answer."
)

#: ``thinking`` arg of ``complete`` / ``chat`` / ``session.send``:
#:
#: * ``"always"`` — pass ``enable_thinking=True`` via chat_template_kwargs
#: * ``"never"``  — pass ``enable_thinking=False``
#: * ``"auto"``   — omit the param; let the server template default decide
ThinkingMode = Literal["auto", "always", "never"]


# ============================================================================
# Regex helpers — strip leaked thinking from assistant content
# ============================================================================

# Tagged form: present when the client requests skip_special_tokens=False.
# Matches the official Gemma 4 channel tags AND legacy <thought>/<think>
# variants used by some chat templates.
_TAGGED_THINK_RE = re.compile(
    r"<\|channel\|?>\s*thought.*?<channel\|?>"
    r"|<thought>.*?</thought>"
    r"|<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)

# Bare-leak form: when skip_special_tokens=True (vLLM default) the
# `<|channel>` / `<channel|>` special tokens are stripped silently and the
# message reads ``thought\n* reasoning ...\n* final answer``. We detect the
# leading ``thought`` substring and extract the last sentence as the answer.
_BARE_PREFIX_RE = re.compile(r"^\s*thought\b\s*", re.IGNORECASE)


def strip_thinking(content: str | None) -> str:
    """Remove leaked thinking-channel content from an assistant message.

    Two paths:

    1. **Tagged form** — ``<|channel>thought ... <channel|>`` (or legacy
       ``<thought>...</thought>``). Surgical regex strip.
    2. **Bare-leak form** — message starts with literal ``thought\\n`` and
       no close tag because the special token was stripped server-side.
       We extract the last paragraph's last line as the final answer.

    A clean message (no thinking) passes through unchanged. Returns empty
    string for None or empty input.
    """
    if not content:
        return ""

    cleaned = _TAGGED_THINK_RE.sub("", content)
    if cleaned != content:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    if not _BARE_PREFIX_RE.match(content):
        return content.strip()

    body = _BARE_PREFIX_RE.sub("", content, count=1)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return ""
    last_para = paragraphs[-1]
    last_lines = [ln.strip() for ln in last_para.splitlines() if ln.strip()]
    if not last_lines:
        return ""
    final = last_lines[-1]
    final = re.sub(r"^\*\s+", "", final).strip()
    return final


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CompletionResult:
    """Outcome of one chat-completion call."""

    content: str               # final answer with thinking stripped
    raw_content: str           # raw assistant content as returned
    reasoning: str | None      # value of message.reasoning_content if any
    usage: dict                # {prompt_tokens, completion_tokens, elapsed_s, ...}


@dataclass
class _Turn:
    role: str
    content: str               # for assistant turns, this is the stripped answer
    raw_content: str | None = None
    reasoning: str | None = None


# ============================================================================
# Multi-turn session
# ============================================================================


class ChatSession:
    """Multi-turn chat with automatic thinking-history stripping.

    Each ``send()`` appends a user message, calls the model, and stores
    the assistant's stripped final answer in history. The next turn's
    request only contains stripped final answers — never thinking
    content — so the model can't feedback-loop on past thoughts.

    The most recent ``CompletionResult`` is exposed as ``last_result`` so
    callers can inspect token usage, reasoning_content, etc.
    """

    def __init__(
        self,
        client: "Gemma4Client",
        *,
        system: str | None = None,
        thinking: ThinkingMode | None = None,
        thinking_budget: int | None = None,
    ):
        self._client = client
        self.system = system
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.turns: list[_Turn] = []
        self.last_result: CompletionResult | None = None

    def messages(self) -> list[dict]:
        """Return the messages list as it would be sent to the server."""
        msgs: list[dict] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def append_user(self, content: str) -> None:
        self.turns.append(_Turn(role="user", content=content))

    def append_assistant(self, raw_content: str, reasoning: str | None = None) -> None:
        clean = strip_thinking(raw_content) if not reasoning else raw_content.strip()
        self.turns.append(
            _Turn(
                role="assistant",
                content=clean,
                raw_content=raw_content,
                reasoning=reasoning,
            )
        )

    def send(
        self,
        content: str,
        *,
        thinking: ThinkingMode | None = None,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Append user message, complete, append assistant reply, return final text."""
        self.append_user(content)
        result = self._client.complete(
            messages=self.messages(),
            thinking=thinking or self.thinking,
            thinking_budget=(
                thinking_budget if thinking_budget is not None else self.thinking_budget
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            extra_body=extra_body,
        )
        self.last_result = result
        self.append_assistant(result.raw_content, result.reasoning)
        return result.content

    def reset(self) -> None:
        """Clear all turns. Keeps `system`, `thinking`, `thinking_budget`."""
        self.turns.clear()
        self.last_result = None


# ============================================================================
# Client
# ============================================================================


def _resolve_enable_thinking(mode: ThinkingMode) -> bool | None:
    """Map ThinkingMode to chat_template_kwargs.enable_thinking value."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    return None  # "auto" → omit param, let server default decide


class Gemma4Client:
    """Reusable client for Gemma 4 endpoints (vLLM-served, OpenAI-compatible).

    Construct directly with ``base_url`` + ``api_key`` for any endpoint, or
    use the ``.fp4()`` / ``.bf16()`` factories for the cola-lab Modal
    deploys. All sampling and thinking-mode params have client-level
    defaults that can be overridden per-call.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = DEFAULT_API_KEY,
        model: str | None = None,
        thinking: ThinkingMode = "auto",
        thinking_budget: int | None = None,
        temperature: float = GEMMA_DEFAULT_TEMPERATURE,
        top_p: float = GEMMA_DEFAULT_TOP_P,
        top_k: int = GEMMA_DEFAULT_TOP_K,
        max_tokens: int = 2048,
        request_timeout: float = 900.0,
        skip_special_tokens_when_thinking: bool = True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self._skip_special_when_thinking = skip_special_tokens_when_thinking

        http_client = httpx.Client(
            timeout=request_timeout,
            follow_redirects=True,  # Modal HTTP 303 dispatch
        )
        self._openai = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            http_client=http_client,
        )

    # ----- Pre-configured factories -----

    @classmethod
    def fp4(cls, **overrides: Any) -> "Gemma4Client":
        """Connect to the cola-lab FP4 endpoint (NVFP4 on H200/B200)."""
        cfg = ENDPOINTS["fp4"]
        return cls(
            base_url=overrides.pop("base_url", cfg["base_url"]),
            model=overrides.pop("model", cfg["model"]),
            **overrides,
        )

    @classmethod
    def bf16(cls, **overrides: Any) -> "Gemma4Client":
        """Connect to the cola-lab BF16 v2 endpoint (production, 8x H200)."""
        cfg = ENDPOINTS["bf16"]
        return cls(
            base_url=overrides.pop("base_url", cfg["base_url"]),
            model=overrides.pop("model", cfg["model"]),
            **overrides,
        )

    # ----- Introspection -----

    def list_models(self) -> list[str]:
        """Return the list of models reported by the endpoint's /v1/models."""
        models = self._openai.models.list()
        return [m.id for m in getattr(models, "data", []) or []]

    def health_check(self, *, raise_on_error: bool = False) -> bool:
        """Quick health-check via /v1/models. Returns True on success."""
        try:
            self.list_models()
            return True
        except Exception:
            if raise_on_error:
                raise
            return False

    # ----- Core API -----

    def complete(
        self,
        messages: list[dict],
        *,
        thinking: ThinkingMode | None = None,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        extra_body: dict | None = None,
    ) -> CompletionResult:
        """Send `messages`, return parsed result.

        Args:
            messages:        OpenAI chat-completion messages list.
            thinking:        per-call override of ``self.thinking``.
            thinking_budget: per-call override; if not None and thinking
                             != "never", a budget hint is appended to the
                             system prompt (or a system message is added
                             if none exists).
            max_tokens, temperature, top_p, top_k, seed:
                             per-call sampling overrides.
            extra_body:      arbitrary fields merged into the OpenAI
                             extra_body payload (vLLM-specific knobs).
        """
        thinking_mode: ThinkingMode = thinking or self.thinking
        budget = thinking_budget if thinking_budget is not None else self.thinking_budget
        enable_thinking_value = _resolve_enable_thinking(thinking_mode)

        # Inject budget hint into messages if applicable
        if budget is not None and thinking_mode != "never":
            messages = self._with_budget_hint(messages, budget)

        # Build extra_body
        body: dict[str, Any] = {
            "top_k": top_k if top_k is not None else self.top_k,
        }
        if enable_thinking_value is not None:
            body["chat_template_kwargs"] = {"enable_thinking": enable_thinking_value}
        if enable_thinking_value is True and self._skip_special_when_thinking:
            body["skip_special_tokens"] = False
        if extra_body:
            for k, v in extra_body.items():
                if k == "chat_template_kwargs" and isinstance(v, dict):
                    body.setdefault("chat_template_kwargs", {}).update(v)
                else:
                    body[k] = v

        # Build OpenAI kwargs
        kwargs: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
            "top_p": top_p if top_p is not None else self.top_p,
            "extra_body": body,
        }
        if seed is not None:
            kwargs["seed"] = seed

        t0 = time.time()
        resp = self._openai.chat.completions.create(**kwargs)
        elapsed = time.time() - t0

        msg = resp.choices[0].message
        raw_content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or None
        clean = strip_thinking(raw_content) if not reasoning else raw_content.strip()

        usage_obj = resp.usage
        usage: dict[str, Any] = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "elapsed_s": round(elapsed, 2),
        }
        details = getattr(usage_obj, "completion_tokens_details", None)
        if details is not None:
            rt = getattr(details, "reasoning_tokens", None)
            if rt is not None:
                usage["reasoning_tokens"] = rt

        return CompletionResult(
            content=clean,
            raw_content=raw_content,
            reasoning=reasoning,
            usage=usage,
        )

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        thinking: ThinkingMode | None = None,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Single-turn chat: returns the final assistant content as a string."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(
            messages=messages,
            thinking=thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            extra_body=extra_body,
        ).content

    def session(
        self,
        *,
        system: str | None = None,
        thinking: ThinkingMode | None = None,
        thinking_budget: int | None = None,
    ) -> ChatSession:
        """Open a multi-turn session that auto-strips thinking from history."""
        return ChatSession(
            client=self,
            system=system,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )

    # ----- Internal helpers -----

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        ids = self.list_models()
        if not ids:
            raise RuntimeError(
                "Server reports no models — endpoint may still be cold-starting"
            )
        self.model = ids[0]
        return self.model

    @staticmethod
    def _with_budget_hint(messages: list[dict], budget: int) -> list[dict]:
        hint = THINKING_BUDGET_HINT_TEMPLATE.format(n=budget)
        if messages and messages[0].get("role") == "system":
            existing = (messages[0].get("content") or "").rstrip()
            if hint in existing:
                return list(messages)
            new_sys = f"{existing}\n\n{hint}" if existing else hint
            return [{"role": "system", "content": new_sys}] + list(messages[1:])
        return [{"role": "system", "content": hint}] + list(messages)


# ============================================================================
# Smoke test (`python -m gemma4_client`)
# ============================================================================


def _smoke() -> None:
    """One-shot smoke test against the FP4 endpoint."""
    client = Gemma4Client.fp4()
    print(f"Endpoint: {client.base_url}")
    print(f"Models:   {client.list_models()}")
    reply = client.chat(
        "Say hello in one short sentence.",
        thinking="never",
        max_tokens=64,
    )
    print(f"Reply:    {reply}")


if __name__ == "__main__":
    _smoke()
