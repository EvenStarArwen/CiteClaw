"""Configuration: globals + raw blocks/pipeline dicts."""

from __future__ import annotations

import os
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
        Discriminator that selects how ``reasoning_effort`` is sent over
        the wire. This is what lets one OpenAI-compatible client serve
        multiple very different providers correctly.

        Accepted values:

        * ``""`` (default) / ``"vllm"`` / ``"gemma4"`` / ``"qwen3"`` /
          ``"deepseek_r1"``: vLLM chat-template shape —
          ``extra_body={"chat_template_kwargs": {"enable_thinking": ...,
          "thinking_budget": ...}, "skip_special_tokens": False}`` plus
          ``reasoning_effort`` and ``max_completion_tokens``. Use for
          any self-hosted vLLM deployment (Modal Gemma, vLLM Qwen3, ...).
          The vLLM server is started with the matching
          ``--reasoning-parser`` flag at deploy time.
        * ``"openai"`` / ``"grok"`` / ``"xai"`` / ``"mistral"`` /
          ``"magistral"``: native ``reasoning_effort`` top-level kwarg
          only. Use for xAI Grok 3/4, Mistral Magistral, OpenAI-compat
          proxies for OpenAI o-series, or any provider whose API mirrors
          the o-series reasoning-effort surface.
        * ``"none"`` / ``"off"`` / ``"disabled"``: no reasoning kwargs
          are sent. Use for OpenAI-compatible endpoints whose models
          don't support reasoning (e.g. plain Together AI Llama).
    request_timeout : float | None
        Per-endpoint override for ``llm_request_timeout``. ``None`` falls
        back to the global setting. Useful when one endpoint (e.g. a
        large reasoning model) needs longer timeouts than the rest.
    thinking_budget : int
        Maximum reasoning tokens the model may spend per call, passed as
        ``thinking_token_budget`` to vLLM. ``0`` (default) means use the
        effort-based default from :func:`_EFFORT_THINKING_BUDGET`.
        Setting this caps the model's thinking trace without affecting
        content output — prevents Gemma 4 / Qwen3 from burning 100 K
        tokens on a single call.
    max_model_len : int
        Server-side context window (prompt + completion) of the deployed
        model, in tokens. Used to clamp ``max_completion_tokens`` so the
        request cannot exceed the endpoint's context. ``0`` (default)
        means no clamp — behaviour is unchanged for SaaS providers and
        large-context self-hosted endpoints. Set this when your vLLM
        deployment uses ``--max-model-len`` smaller than ``3 ×
        thinking_budget`` (the unclamped completion default). Example:
        Gemma 4 31B on a 32K Modal deployment with ``reasoning_effort:
        high`` would request 49K completion tokens and 400-fail without
        this field set to ``32768``.
    """

    base_url: str
    served_model_name: str = ""
    api_key_env: str = ""
    reasoning_parser: str = ""
    request_timeout: float | None = None
    thinking_budget: int = 0
    max_model_len: int = 0

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
    # OpenAlex — abstract + reference fallback when S2 is incomplete.
    # Polite-pool access is free and unauthenticated; setting an API key
    # unlocks the higher-rate-limit pool. ``openalex_email`` identifies
    # the caller for polite-pool etiquette (OpenAlex prefers this to bare
    # IPs). Absent both, the client still works but is rate-limited.
    openalex_api_key: str = ""
    openalex_email: str = ""

    # LLM
    screening_model: str = "stub"
    # Dedicated model override for the ExpandBySearch supervisor + workers
    # (citeclaw.agents.supervisor / .worker). Empty (the default) means
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
    # Hard ceiling on the number of consecutive S2 calls (across endpoints)
    # that may exhaust their tenacity retry budget before we declare the
    # API down and abort the pipeline. Each `_is_retryable` failure burns
    # 6 attempts × backoff before counting as one "failure" here, so the
    # default 10 means "10 papers in a row that all required a full
    # retry cascade and still didn't make it" — a clean signal of a
    # sustained outage rather than transient hiccups. Set to 0 to
    # disable the auto-abort.
    s2_max_consecutive_failures: int = 10
    # OpenAlex rate limit in requests / second. The polite-pool ceiling
    # is ~10 rps; the API-key pool is ~100 rps. Default of 5 rps keeps
    # the polite pool happy without aggressive bursts.
    openalex_rps: float = 5.0

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
    "openalex_api_key",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "S2_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "LLM_API_KEY",
    "OPENALEX_API_KEY",
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
        (("CITECLAW_S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY"), "s2_api_key"),
        (("CITECLAW_OPENALEX_API_KEY", "OPENALEX_API_KEY"), "openalex_api_key"),
        (("CITECLAW_OPENALEX_EMAIL", "OPENALEX_EMAIL"), "openalex_email"),
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
