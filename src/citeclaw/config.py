"""Configuration: globals + raw blocks/pipeline dicts + budget tracker."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class SeedPaper(BaseModel):
    """One seed paper entry from YAML.

    PC-04 made ``paper_id`` optional so YAML callers can write
    ``{title: "..."}``-only entries that the ``ResolveSeeds`` step
    will look up via ``s2.search_match`` before ``LoadSeeds`` runs.
    Plain ``{paper_id: "..."}`` entries continue to work unchanged.
    """

    paper_id: str = ""
    title: str = ""
    abstract: str | None = None


class ModelEndpoint(BaseModel):
    """One entry in the ``models:`` registry.

    Maps a YAML alias (e.g. ``gemma-4-31b``) to an OpenAI-compatible
    endpoint and the underlying model name vLLM/Ollama serves on it. This
    is what lets a single config file mix Gemini, OpenAI SaaS, and one or
    more self-hosted vLLM endpoints — every block can pick a model alias
    independently and the factory routes each to its own endpoint.

    Fields
    ------
    base_url : str
        The OpenAI-compatible HTTP endpoint, e.g.
        ``https://you--citeclaw-vllm-gemma-serve.modal.run/v1``. The
        trailing ``/v1`` is required (vLLM serves the OpenAI surface
        under that prefix).
    served_model_name : str
        The exact model string the endpoint expects in the chat-completions
        ``model`` field — typically a HuggingFace ID like
        ``google/gemma-4-31B-it``. May differ from the YAML alias.
    api_key_env : str
        Name of the environment variable that holds the bearer token. The
        key itself never appears in YAML; the loader resolves the env var
        at startup. Empty means "use ``CITECLAW_VLLM_API_KEY`` if set,
        otherwise no auth".
    reasoning_parser : str
        Hint for downstream tooling that the endpoint understands a
        specific reasoning parser (``gemma4``, ``qwen3``, ``deepseek_r1``,
        ...). The vLLM server is started with this parser at deploy time;
        the client uses the field as documentation only.
    request_timeout : float | None
        Per-endpoint override for ``llm_request_timeout``. ``None`` falls
        back to the global setting. Useful when one endpoint (e.g. a
        large reasoning model) needs longer timeouts than the rest.
    """

    base_url: str
    served_model_name: str = ""
    api_key_env: str = ""
    reasoning_parser: str = ""
    request_timeout: float | None = None

    @property
    def resolved_api_key(self) -> str:
        """Look up the bearer token from the configured env var.

        Returns ``""`` when neither ``api_key_env`` nor the documented
        fallbacks (``CITECLAW_VLLM_API_KEY``, ``OPENAI_API_KEY``) are set.
        Some self-hosted endpoints accept any non-empty string, in which
        case ``"none"`` is fine; the OpenAI SDK rejects an empty string,
        so the OpenAIClient will substitute ``"none"`` if this returns
        empty.
        """
        if self.api_key_env:
            v = os.environ.get(self.api_key_env, "")
            if v:
                return v
        for fallback in ("CITECLAW_VLLM_API_KEY", "OPENAI_API_KEY"):
            v = os.environ.get(fallback, "")
            if v:
                return v
        return ""


class Settings(BaseSettings):
    # API keys
    openai_api_key: str = ""
    gemini_api_key: str = ""
    s2_api_key: str = ""

    # LLM
    screening_model: str = "stub"
    # PC-06: dedicated model override for the iterative meta-LLM search
    # agent (citeclaw.agents.iterative_search). Empty (the default) means
    # the agent inherits ``screening_model``; set this to a more capable
    # model when you want screening to stay cheap but the search agent
    # to think harder. ``ExpandBySearch`` reads it via the cascade
    # ``self.agent.model or ctx.config.search_model or ctx.config.screening_model``.
    search_model: str = ""
    reasoning_effort: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_request_timeout: float = 300.0
    # Per-model endpoint registry. Each key is a YAML alias the user picks
    # (e.g. ``gemma-4-31b``); each value is a :class:`ModelEndpoint`. When
    # a block sets ``model: gemma-4-31b``, the factory looks up that alias
    # here, builds an OpenAIClient pointed at the alias's ``base_url``, and
    # tells it to send ``served_model_name`` over the wire. This is what
    # lets a single config switch between Gemini, OpenAI SaaS, and one or
    # more self-hosted vLLM endpoints with a single string in the YAML.
    #
    # Aliases that are NOT in the registry fall through to the legacy
    # routing (Gemini detection / global ``llm_base_url`` / OpenAI SaaS),
    # so existing configs keep working unchanged.
    models: dict[str, ModelEndpoint] = Field(default_factory=dict)
    llm_batch_size: int = 20
    llm_concurrency: int = 4
    # Structured-output kill switch. When True (default) the screening
    # runner requests a strict JSON schema from the LLM (OpenAI
    # ``response_format``/Gemini ``response_schema``) so we never silently
    # lose a batch to a malformed JSON close-brace. Set False for
    # OpenAI-compatible endpoints (vLLM/Ollama) that don't honor the flag.
    structured_output_enabled: bool = True

    # Budgets / rate limits
    s2_rps: float = Field(default=0.9, validation_alias="s2_requests_per_second")
    max_llm_tokens: int = 50_000_000
    max_s2_requests: int = 100_000
    max_papers_total: int = 2000

    # Topic + IO
    topic_description: str = ""
    data_dir: Path = Path("data")
    seed_papers: list[SeedPaper] = Field(default_factory=list)

    # Annotate hook (used by annotate.py)
    graph_label_instruction: str = ""

    # Two-section schema
    blocks: dict[str, dict] = Field(default_factory=dict)
    pipeline: list[dict] = Field(default_factory=list)

    # Built lazily after validation
    blocks_built: dict = Field(default_factory=dict, exclude=True)
    pipeline_built: list = Field(default_factory=list, exclude=True)

    model_config = {
        "env_prefix": "CITECLAW_",
        "env_nested_delimiter": "__",
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def _build(self):
        # Defer to avoid circular imports
        if self.blocks:
            from citeclaw.filters.builder import build_blocks
            object.__setattr__(self, "blocks_built", build_blocks(self.blocks))
        if self.pipeline:
            from citeclaw.steps import build_step
            object.__setattr__(
                self, "pipeline_built",
                [build_step(d, self.blocks_built) for d in self.pipeline],
            )
        return self


# API-key fields are intentionally NOT allowed in YAML. They must come from
# environment variables so they never end up committed to a config file.
# ``_normalize_yaml`` raises a loud error if it sees one, naming every
# offending key. We cover both the canonical lowercase form and every legacy
# uppercase alias the loader used to accept.
_FORBIDDEN_YAML_KEYS = {
    "openai_api_key",
    "gemini_api_key",
    "s2_api_key",
    "llm_api_key",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "S2_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "LLM_API_KEY",
}


def _normalize_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    found = [k for k in raw if k in _FORBIDDEN_YAML_KEYS]
    if found:
        raise ValueError(
            "API keys must not be set in config YAML (found: "
            + ", ".join(sorted(found))
            + "). Remove these fields from your config file and export the "
            "corresponding environment variables instead: OPENAI_API_KEY, "
            "GEMINI_API_KEY, S2_API_KEY (or SEMANTIC_SCHOLAR_API_KEY)."
        )
    return dict(raw)


def _env_overrides(values: dict[str, Any]) -> None:
    for env_keys, field in [
        (("CITECLAW_OPENAI_API_KEY", "OPENAI_API_KEY"), "openai_api_key"),
        (("CITECLAW_GEMINI_API_KEY", "GEMINI_API_KEY"), "gemini_api_key"),
        (("CITECLAW_S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"), "s2_api_key"),
    ]:
        for k in env_keys:
            v = os.environ.get(k)
            if v:
                values[field] = v
                break


def load_settings(config_path: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    values: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")
        values.update(_normalize_yaml(data))
    if overrides:
        values.update(overrides)
    _env_overrides(values)
    return Settings(**values)


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------

# Per-million-token prices in USD. Format: (input, output, reasoning).
#
# IMPORTANT: there is no API endpoint that returns the actual USD billed
# for a given call. OpenAI / Gemini / Anthropic all return token *counts*
# in usage metadata, not dollar amounts. The only way to get a real cost
# figure is to (a) maintain a local pricing table — what we do here — and
# multiply, OR (b) query the provider's billing/usage API after the fact
# (which is delayed, requires admin keys, and isn't per-call). The local
# pricing table is the practical option for live in-run estimates.
#
# Reasoning-token billing convention:
#   * Gemini "thinking" tokens are billed at the output rate.
#   * OpenAI o-series (o1/o3/o4) reasoning tokens are billed at the
#     output rate (Anthropic's extended-thinking is similar).
# So the third tuple slot ("reasoning") is set equal to "output" for
# every entry below; it's a knob in case a future provider splits them.
#
# VERIFIED: 2026-01 — when adding a new provider or you suspect rates
# have shifted, re-check the official pricing pages and bump the date.
# Unknown models fall back to ``GENERIC`` (mid-range Gemini Flash) so
# you get a meaningful number rather than $0.

MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # Gemini 3 (current generation)
    "gemini-3-pro":                          (1.25, 10.00, 10.00),
    "gemini-3-flash":                        (0.30,  2.50,  2.50),
    "gemini-3-flash-lite":                   (0.10,  0.40,  0.40),
    "gemini-3.1-flash-lite-preview":         (0.10,  0.40,  0.40),
    # Gemini 2.5 (still widely deployed)
    "gemini-2.5-pro":                        (1.25, 10.00, 10.00),
    "gemini-2.5-flash":                      (0.30,  2.50,  2.50),
    "gemini-2.5-flash-lite":                 (0.10,  0.40,  0.40),
    # OpenAI o-series (reasoning at output rate)
    "o3":                                    (2.00,  8.00,  8.00),
    "o3-mini":                               (1.10,  4.40,  4.40),
    "o4-mini":                               (1.10,  4.40,  4.40),
    # OpenAI gpt
    "gpt-5":                                 (1.25, 10.00, 10.00),
    "gpt-5-mini":                            (0.25,  2.00,  2.00),
    "gpt-4.1":                               (2.00,  8.00,  8.00),
    "gpt-4.1-mini":                          (0.40,  1.60,  1.60),
    # Anthropic (extended-thinking billed at output rate)
    "claude-opus-4-6":                       (15.00, 75.00, 75.00),
    "claude-sonnet-4-6":                     (3.00, 15.00, 15.00),
    "claude-haiku-4-5":                      (1.00,  5.00,  5.00),
    "claude-opus-4":                         (15.00, 75.00, 75.00),
    "claude-sonnet-4":                       (3.00, 15.00, 15.00),
    "claude-haiku-4":                        (0.80,  4.00,  4.00),
    # Stub for tests / dev runs
    "stub":                                  (0.0,   0.0,   0.0),
    # Catch-all for unknown models — set to mid-range Gemini Flash so
    # estimates are reasonable rather than zero.
    "GENERIC":                               (0.30,  2.50,  2.50),
}


def lookup_pricing(model_name: str | None) -> tuple[float, float, float]:
    """Return ``(input_per_M, output_per_M, reasoning_per_M)`` for ``model_name``.

    Falls back to ``GENERIC`` if the exact name is not in ``MODEL_PRICING``;
    also tries a few prefix matches so e.g. ``gemini-3-flash-lite-preview-09``
    matches ``gemini-3-flash-lite``.
    """
    if not model_name:
        return MODEL_PRICING["GENERIC"]
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # Prefix fallbacks: longest-key-first so more specific names win.
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if key in {"GENERIC", "stub"}:
            continue
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    return MODEL_PRICING["GENERIC"]


class BudgetTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm_tokens: dict[str, int] = {}            # input + output, combined
        self._llm_input_tokens: dict[str, int] = {}      # input only (added for cost split)
        self._llm_output_tokens: dict[str, int] = {}     # output only (added for cost split)
        self._llm_calls: dict[str, int] = {}
        self._llm_reasoning_tokens: dict[str, int] = {}
        self._s2_api: dict[str, int] = {}
        self._s2_cache: dict[str, int] = {}

    @property
    def llm_total_tokens(self) -> int:
        return sum(self._llm_tokens.values())

    @property
    def llm_input_tokens(self) -> int:
        return sum(self._llm_input_tokens.values())

    @property
    def llm_output_tokens(self) -> int:
        return sum(self._llm_output_tokens.values())

    @property
    def llm_reasoning_tokens(self) -> int:
        return sum(self._llm_reasoning_tokens.values())

    @property
    def llm_calls(self) -> int:
        return sum(self._llm_calls.values())

    @property
    def s2_requests(self) -> int:
        return sum(self._s2_api.values())

    @property
    def s2_cache_hits(self) -> int:
        return sum(self._s2_cache.values())

    def record_llm(self, input_tokens: int, output_tokens: int, category: str = "other", *, reasoning_tokens: int = 0) -> None:
        with self._lock:
            total = input_tokens + output_tokens
            self._llm_tokens[category] = self._llm_tokens.get(category, 0) + total
            self._llm_input_tokens[category] = self._llm_input_tokens.get(category, 0) + input_tokens
            self._llm_output_tokens[category] = self._llm_output_tokens.get(category, 0) + output_tokens
            self._llm_calls[category] = self._llm_calls.get(category, 0) + 1
            if reasoning_tokens:
                self._llm_reasoning_tokens[category] = self._llm_reasoning_tokens.get(category, 0) + reasoning_tokens

    def record_s2(self, req_type: str = "other", *, cached: bool = False) -> None:
        with self._lock:
            if cached:
                self._s2_cache[req_type] = self._s2_cache.get(req_type, 0) + 1
            else:
                self._s2_api[req_type] = self._s2_api.get(req_type, 0) + 1

    def is_exhausted(self, config: Settings) -> bool:
        if self.llm_total_tokens >= config.max_llm_tokens:
            return True
        if self.s2_requests >= config.max_s2_requests:
            return True
        return False

    def exhausted(self) -> bool:
        return False  # placeholder used by pipeline runner

    def summary(self) -> str:
        return (
            f"LLM: {self.llm_total_tokens / 1_000_000:.3f}M tokens "
            f"({self.llm_calls} calls) | "
            f"S2: {self.s2_requests:,} api + {sum(self._s2_cache.values()):,} cached"
        )

    def cost_estimate(self, model_name: str | None) -> float:
        """Estimate USD cost for the LLM tokens recorded so far.

        Uses :func:`lookup_pricing` to map ``model_name`` to per-million
        prices. Reasoning tokens are billed at the reasoning rate (which for
        most providers is the same as the output rate). Returns 0.0 for the
        stub model so dev runs don't show a fake cost.

        IMPORTANT: this is a *local* estimate computed from token counts +
        the ``MODEL_PRICING`` table. There is no provider API that returns
        the exact USD cost of a single call — the only ground truth is the
        provider's billing dashboard, which is delayed and aggregated. Keep
        ``MODEL_PRICING`` up to date if you need accurate numbers.
        """
        in_per_m, out_per_m, reason_per_m = lookup_pricing(model_name)
        return (
            self.llm_input_tokens     * in_per_m     / 1_000_000
            + self.llm_output_tokens    * out_per_m    / 1_000_000
            + self.llm_reasoning_tokens * reason_per_m / 1_000_000
        )

    def cost_breakdown(self, model_name: str | None) -> dict[str, dict[str, float | int]]:
        """Per-category cost breakdown for the run-end summary.

        Returns ``{category: {"input": tokens, "output": tokens, "reason": tokens, "calls": n, "cost_usd": float}}``
        sorted by cost descending. Useful for spotting which filter is
        eating the budget. Uses one model's pricing for all categories —
        accurate for single-model runs, approximate when filters override
        the model.
        """
        in_per_m, out_per_m, reason_per_m = lookup_pricing(model_name)
        out: dict[str, dict[str, float | int]] = {}
        for cat in self._llm_tokens:
            in_t = self._llm_input_tokens.get(cat, 0)
            out_t = self._llm_output_tokens.get(cat, 0)
            reason_t = self._llm_reasoning_tokens.get(cat, 0)
            calls = self._llm_calls.get(cat, 0)
            cost = (
                in_t     * in_per_m     / 1_000_000
                + out_t    * out_per_m    / 1_000_000
                + reason_t * reason_per_m / 1_000_000
            )
            out[cat] = {
                "input": in_t,
                "output": out_t,
                "reason": reason_t,
                "calls": calls,
                "cost_usd": cost,
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["cost_usd"]))

    def detailed_summary(self) -> str:
        lines = ["Budget breakdown:"]
        lines.append("  LLM usage:")
        for cat in sorted(self._llm_tokens.keys()):
            tokens = self._llm_tokens[cat]
            calls = self._llm_calls.get(cat, 0)
            lines.append(f"    {cat:<20s}  {tokens / 1_000_000:.3f}M tokens  ({calls} calls)")
        lines.append(f"    {'TOTAL':<20s}  {self.llm_total_tokens / 1_000_000:.3f}M tokens  ({self.llm_calls} calls)")
        lines.append("  S2 usage:")
        all_types = sorted(set(self._s2_api.keys()) | set(self._s2_cache.keys()))
        for t in all_types:
            api = self._s2_api.get(t, 0)
            cache = self._s2_cache.get(t, 0)
            lines.append(f"    {t:<20s}  {api:>5} api  {cache:>5} cached")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm": {
                "total_tokens": self.llm_total_tokens,
                "total_tokens_millions": round(self.llm_total_tokens / 1_000_000, 3),
                "total_reasoning_tokens": self.llm_reasoning_tokens,
                "total_calls": self.llm_calls,
                "by_category": {
                    cat: {
                        "tokens": self._llm_tokens[cat],
                        "calls": self._llm_calls.get(cat, 0),
                        "reasoning_tokens": self._llm_reasoning_tokens.get(cat, 0),
                    }
                    for cat in self._llm_tokens
                },
            },
            "s2": {
                "total_api_requests": sum(self._s2_api.values()),
                "total_cache_hits": sum(self._s2_cache.values()),
                "by_type": {
                    t: {"api": self._s2_api.get(t, 0), "cached": self._s2_cache.get(t, 0)}
                    for t in sorted(set(self._s2_api.keys()) | set(self._s2_cache.keys()))
                },
            },
        }
