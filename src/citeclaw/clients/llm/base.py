"""LLMClient Protocol, LLMResponse value type, and the client error hierarchy.

The pipeline never type-checks individual LLM providers — every screener,
reranker, annotator, and ad-hoc caller goes through the :class:`LLMClient`
Protocol. Concrete clients live in sibling modules
(:mod:`citeclaw.clients.llm.openai_client`, :mod:`.gemini`, :mod:`.stub`)
and the :func:`citeclaw.clients.llm.factory.build_llm_client` factory
picks one based on the configured model alias.

Errors fall into two layers. Configuration / construction problems
(missing API key, unreachable endpoint) raise :class:`LLMConfigError`,
which is catchable as :class:`LLMClientError`. Network / rate-limit
errors propagate as the underlying SDK's exception (httpx /
google.genai / openai) — those are runtime transients
:mod:`tenacity` already handles at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMClientError(Exception):
    """Base for every LLM-client-side error.

    Concrete clients raise subclasses (currently :class:`LLMConfigError`)
    so consumers can ``except LLMClientError`` to catch any client-side
    failure without binding to provider-specific error types.
    """


class LLMConfigError(LLMClientError):
    """Raised when an LLM client cannot reach the requested model.

    Today this means a missing API key for a SaaS provider; in future it
    may cover unreachable custom endpoints, malformed model aliases, etc.
    Surface callers can catch this to fail fast with an actionable
    message rather than letting a network-level error bubble.
    """


@dataclass(frozen=True)
class LLMResponse:
    """One LLM call's output, frozen so providers can return shared instances.

    ``text`` is the clean assistant answer (the OpenAI ``message.content``
    field for chat models — never None; providers normalise to "" on
    refusal / empty completion).

    ``reasoning_content`` carries the model's thinking trace when the
    provider exposes it as a separate field: vLLM with a working
    reasoning parser populates ``message.reasoning``; OpenAI's o-series
    keeps it inside ``completion_tokens_details`` (token count only, not
    surfaced); Gemini 2.5/3 surfaces ``thinking`` parts on the response.
    Empty string when the model didn't think or the provider didn't
    expose it.

    ``logprob_tokens`` is the per-token log-probability list for
    providers that support it (OpenAI chat completions —
    ``logprobs.content``). Empty for providers that don't expose
    logprobs (Gemini, all reasoning models, the stub). The type is
    intentionally loose so each provider can pass through its native
    shape; consumers either iterate looking for ``.token`` / ``.logprob``
    attributes or JSON-serialise the whole list.
    """

    text: str
    logprob_tokens: list[Any] = field(default_factory=list)
    reasoning_content: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat-style LLM client."""

    def call(
        self,
        system: str,
        user: str,
        *,
        with_logprobs: bool = False,
        category: str = "other",
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Run one chat-style completion.

        Parameters
        ----------
        system, user
            Standard chat-completion role contents.
        with_logprobs
            Request per-token logprobs in the response. Silently
            ignored by providers that don't support them — check
            :attr:`supports_logprobs` if the caller needs to gate
            behaviour. The returned :attr:`LLMResponse.logprob_tokens`
            is empty when logprobs are unavailable, regardless of this
            flag.
        category
            Free-form bucket key passed through to
            :class:`citeclaw.budget.BudgetTracker` for per-category
            token / call accounting. The default ``"other"`` lands every
            unattributed call in a single bucket; specialised callers
            (annotation, PDF reference extraction, screening) pass
            their own label so the run summary breaks costs down.
        response_schema
            Optional JSON Schema dict translated into the provider's
            native structured-output constraint (OpenAI
            ``response_format={"type": "json_schema", ...}``; Gemini
            ``response_schema`` + ``response_mime_type="application/json"``).
            Stub clients ignore it. Providers without structured-output
            support fall back to free-form text — the caller is
            responsible for parsing.
        """
        ...

    @property
    def supports_logprobs(self) -> bool:
        """True iff the provider can return per-token logprobs.

        Used by callers (e.g. confidence calibration) to decide whether
        to pass ``with_logprobs=True`` at all rather than silently
        receiving an empty list.
        """
        ...
