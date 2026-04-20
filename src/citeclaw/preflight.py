"""Pre-flight validation: check API keys before the pipeline starts.

The CLI runs this once after building :class:`Settings` so the user gets
a clear, actionable error up-front instead of "invalid API key" buried
in the middle of an LLM call. The check walks the configured pipeline
and filter blocks to figure out which provider keys will actually be
needed at runtime, so a stub-only run never demands an OpenAI key and
a Gemma-via-vLLM run never demands a Gemini key.

Two public helpers:

* :func:`required_keys_for_model` — single-model reverse lookup that
  mirrors :func:`citeclaw.clients.llm.factory.build_llm_client`'s
  routing decisions. Returns ``(env_var, reason)`` or ``None`` for the
  stub.
* :func:`find_missing_api_keys` — walks the whole config and returns a
  human-readable list of every missing required key, deduplicated.

S2 is always required (the pipeline's only retrieval backend), so an
empty ``s2_api_key`` is always a missing key.
"""

from __future__ import annotations

from typing import Iterable

from citeclaw.config import Settings


def required_keys_for_model(
    model: str | None, config: Settings,
) -> tuple[str, str] | None:
    """Return ``(env_var_name, human_reason)`` for the given model alias.

    Mirrors :func:`citeclaw.clients.llm.factory.build_llm_client`'s
    routing precedence:

      1. ``stub`` (case-insensitive) → no key needed.
      2. ``model`` is in ``config.models`` registry → the entry's
         ``api_key_env`` (with documented fallbacks). Only required if
         the entry doesn't already resolve to a non-empty token.
      3. ``config.llm_base_url`` set → ``LLM_API_KEY`` (with
         ``OPENAI_API_KEY`` fallback).
      4. Model name starts with ``gemini-`` → ``GEMINI_API_KEY``.
      5. Anything else (gpt-*, o1/o3/o4, claude-*, etc.) →
         ``OPENAI_API_KEY``.

    Returns ``None`` when no key is required (e.g. stub) OR when the
    routing path resolves a key from the environment without further
    config.
    """
    name = (model or config.screening_model or "").strip()
    if not name or name.lower() == "stub":
        return None

    if name in config.models:
        entry = config.models[name]
        if entry.resolved_api_key:
            return None
        env = entry.api_key_env or "CITECLAW_VLLM_API_KEY"
        return (env, f"model alias {name!r} (registry endpoint {entry.base_url})")

    if config.llm_base_url:
        if config.llm_api_key or config.openai_api_key:
            return None
        return ("LLM_API_KEY", f"custom llm_base_url {config.llm_base_url!r}")

    if name.startswith("gemini-"):
        if config.gemini_api_key:
            return None
        return ("GEMINI_API_KEY", f"Gemini model {name!r}")

    # OpenAI-SaaS fallback (covers gpt-*, o1/o3/o4, claude-*, anything
    # else routed through OpenAIClient with no custom endpoint).
    if config.openai_api_key:
        return None
    return ("OPENAI_API_KEY", f"model {name!r} (default OpenAI client)")


def _walk_blocks(block) -> Iterable[str]:
    """Yield every per-filter ``model:`` override found inside ``block``.

    Recurses through Sequential / Any / Not / Route compositors, and
    emits the ``model`` attribute for each :class:`LLMFilter` leaf when
    one is set. ``model=None`` (the common case) is skipped — those
    filters use ``screening_model``, which is checked separately.
    """
    if block is None:
        return
    model = getattr(block, "model", None)
    if isinstance(model, str) and model:
        yield model
    for child in getattr(block, "layers", None) or ():
        yield from _walk_blocks(child)
    inner = getattr(block, "layer", None)
    if inner is not None:
        yield from _walk_blocks(inner)
    for case in getattr(block, "cases", None) or ():
        yield from _walk_blocks(getattr(case, "target", None))
        yield from _walk_blocks(getattr(case, "predicate", None))


def _walk_pipeline(pipeline) -> Iterable[str]:
    """Yield every step-level model override (ExpandBySearch.agent.model,
    ExpandByPDF.model, ...).

    Recurses through ``Parallel`` branches.
    """
    for step in pipeline or []:
        agent = getattr(step, "agent", None)
        if agent is not None:
            # ``agent`` may be a dataclass (legacy / future agent
            # backend) or a plain dict (current ExpandBySearch shell
            # passes the YAML mapping straight through).
            m = (
                agent.get("model") if isinstance(agent, dict)
                else getattr(agent, "model", None)
            )
            if isinstance(m, str) and m:
                yield m
        m = getattr(step, "model", None)
        if isinstance(m, str) and m:
            yield m
        for branch in getattr(step, "branches", None) or ():
            yield from _walk_pipeline(branch)


def find_missing_api_keys(config: Settings) -> list[str]:
    """Return a list of human-readable error lines for every missing key.

    S2 is always required. LLM keys are inferred from
    ``screening_model``, ``search_model``, every ``LLMFilter.model``
    override in the built blocks, and every step-level ``model`` /
    ``agent.model`` override in the built pipeline.

    Each returned line is suitable for logging directly; an empty list
    means the run is good to go.
    """
    missing: dict[str, str] = {}

    # S2 is mandatory. Its key has no in-YAML fallback path because the
    # config validator forbids API keys in YAML — only env vars.
    if not config.s2_api_key:
        missing["S2_API_KEY"] = (
            "Semantic Scholar (always required for paper metadata + search)"
        )

    # Collect every model alias the run will reach.
    models: list[str] = []
    if config.screening_model:
        models.append(config.screening_model)
    if config.search_model:
        models.append(config.search_model)
    for m in _walk_blocks_in_built(config):
        models.append(m)
    for m in _walk_pipeline(config.pipeline_built):
        models.append(m)

    for model in models:
        result = required_keys_for_model(model, config)
        if result is None:
            continue
        env, reason = result
        # Keep first reason recorded against an env var; downstream
        # users only need one example to fix it.
        missing.setdefault(env, reason)

    return [
        f"missing env var {env!r} — needed for {reason}"
        for env, reason in missing.items()
    ]


def find_optional_unset_keys(config: Settings) -> list[str]:
    """Report optional keys that are unset but unlock better behaviour.

    Unlike :func:`find_missing_api_keys`, nothing in this list blocks a
    run — the caller is expected to log these as warnings rather than
    errors. Currently only :envvar:`OPENALEX_API_KEY`: unset, OpenAlex
    still works via the polite pool, but abstract / reference fallback
    will be rate-limited to ~10 rps.
    """
    out: list[str] = []
    if not config.openalex_api_key:
        out.append(
            "OPENALEX_API_KEY unset — OpenAlex fallback will use the "
            "polite pool (~10 rps). Set the env var for higher limits."
        )
    return out


def _walk_blocks_in_built(config: Settings) -> Iterable[str]:
    """Iterate every model override across all built blocks."""
    for block in (config.blocks_built or {}).values():
        yield from _walk_blocks(block)
