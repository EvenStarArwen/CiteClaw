"""Tests for the pre-flight API key validator.

The validator walks the configured pipeline + filter blocks to figure
out which provider keys will actually be reached at runtime, so a
stub-only run never demands an OpenAI key and a Gemma-via-vLLM run
never demands a Gemini key. S2 is always required.
"""

from __future__ import annotations

import pytest

from citeclaw.config import ModelEndpoint, Settings
from citeclaw.preflight import (
    find_missing_api_keys,
    find_optional_unset_keys,
    required_keys_for_model,
)


def _settings(**overrides) -> Settings:
    base = dict(
        screening_model="stub",
        s2_api_key="dummy",
        seed_papers=[],
        blocks={},
        pipeline=[],
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# required_keys_for_model
# ---------------------------------------------------------------------------


class TestRequiredKeysForModel:
    def test_stub_needs_nothing(self):
        cfg = _settings()
        assert required_keys_for_model("stub", cfg) is None
        assert required_keys_for_model("STUB", cfg) is None

    def test_empty_uses_screening_model(self):
        cfg = _settings(screening_model="stub")
        assert required_keys_for_model("", cfg) is None
        assert required_keys_for_model(None, cfg) is None

    def test_gemini_needs_gemini_key(self):
        cfg = _settings()
        env, _ = required_keys_for_model("gemini-3-pro", cfg)
        assert env == "GEMINI_API_KEY"

    def test_gemini_satisfied_when_set(self):
        cfg = _settings(gemini_api_key="present")
        assert required_keys_for_model("gemini-3-pro", cfg) is None

    def test_openai_default_for_gpt(self):
        cfg = _settings()
        env, _ = required_keys_for_model("gpt-4o", cfg)
        assert env == "OPENAI_API_KEY"

    def test_openai_default_for_o_series(self):
        cfg = _settings()
        env, _ = required_keys_for_model("o3-mini", cfg)
        assert env == "OPENAI_API_KEY"

    def test_openai_satisfied_when_set(self):
        cfg = _settings(openai_api_key="present")
        assert required_keys_for_model("gpt-4o", cfg) is None

    def test_registry_alias_uses_endpoint_env_var(self, monkeypatch):
        monkeypatch.delenv("CITECLAW_VLLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROK_KEY", raising=False)
        cfg = _settings(
            models={
                "grok-4": ModelEndpoint(
                    base_url="https://api.x.ai/v1",
                    served_model_name="grok-4",
                    api_key_env="GROK_KEY",
                    reasoning_parser="grok",
                ),
            },
        )
        env, reason = required_keys_for_model("grok-4", cfg)
        assert env == "GROK_KEY"
        assert "grok-4" in reason

    def test_registry_alias_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("GROK_KEY", "present")
        cfg = _settings(
            models={
                "grok-4": ModelEndpoint(
                    base_url="https://api.x.ai/v1",
                    served_model_name="grok-4",
                    api_key_env="GROK_KEY",
                ),
            },
        )
        assert required_keys_for_model("grok-4", cfg) is None

    def test_legacy_llm_base_url_needs_llm_api_key(self):
        cfg = _settings(llm_base_url="https://example.com/v1")
        env, _ = required_keys_for_model("custom-model", cfg)
        assert env == "LLM_API_KEY"


# ---------------------------------------------------------------------------
# find_missing_api_keys (full walk)
# ---------------------------------------------------------------------------


class TestFindMissingApiKeys:
    def test_stub_run_only_needs_s2(self):
        cfg = _settings(s2_api_key="dummy")
        assert find_missing_api_keys(cfg) == []

    def test_missing_s2_is_reported(self):
        cfg = _settings(s2_api_key="")
        msgs = find_missing_api_keys(cfg)
        assert any("S2_API_KEY" in m for m in msgs)

    def test_gemini_screening_without_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = _settings(screening_model="gemini-3-pro")
        msgs = find_missing_api_keys(cfg)
        assert any("GEMINI_API_KEY" in m for m in msgs)

    def test_search_model_override_picks_up_key_too(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _settings(
            screening_model="stub",
            search_model="gpt-4o",
        )
        msgs = find_missing_api_keys(cfg)
        assert any("OPENAI_API_KEY" in m for m in msgs)

    def test_per_filter_model_override_picked_up(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Build a real Settings with one block whose LLMFilter overrides
        # the model. The block builder runs in Settings.__init__, so
        # blocks_built is populated and the walker can see the override.
        cfg = Settings(
            screening_model="stub",
            s2_api_key="dummy",
            blocks={
                "topic_llm": {
                    "type": "LLMFilter",
                    "scope": "title",
                    "prompt": "the paper is on topic",
                    "model": "gemini-3-pro",
                },
            },
            pipeline=[],
        )
        msgs = find_missing_api_keys(cfg)
        assert any("GEMINI_API_KEY" in m for m in msgs), msgs

    def test_registry_route_validated_via_env(self, monkeypatch):
        monkeypatch.delenv("CITECLAW_VLLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROK_KEY", raising=False)
        cfg = _settings(
            screening_model="grok-4",
            models={
                "grok-4": ModelEndpoint(
                    base_url="https://api.x.ai/v1",
                    served_model_name="grok-4",
                    api_key_env="GROK_KEY",
                    reasoning_parser="grok",
                ),
            },
        )
        msgs = find_missing_api_keys(cfg)
        assert any("GROK_KEY" in m for m in msgs)

    def test_dedup_one_message_per_env_var(self, monkeypatch):
        """Two filters needing the same key produce a single error line."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Settings(
            screening_model="gpt-4o",
            search_model="gpt-4o",
            s2_api_key="dummy",
            blocks={
                "a": {
                    "type": "LLMFilter", "scope": "title",
                    "prompt": "x", "model": "gpt-4o-mini",
                },
                "b": {
                    "type": "LLMFilter", "scope": "title",
                    "prompt": "y", "model": "gpt-4o",
                },
            },
            pipeline=[],
        )
        msgs = find_missing_api_keys(cfg)
        openai_msgs = [m for m in msgs if "OPENAI_API_KEY" in m]
        assert len(openai_msgs) == 1, openai_msgs

    def test_pipeline_step_agent_model_picked_up(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = Settings(
            screening_model="stub",
            s2_api_key="dummy",
            blocks={
                "title_llm": {
                    "type": "LLMFilter", "scope": "title", "prompt": "p",
                },
            },
            pipeline=[
                {"step": "LoadSeeds"},
                {
                    "step": "ExpandBySearch",
                    "screener": "title_llm",
                    "agent": {"model": "gemini-3-pro"},
                },
            ],
        )
        msgs = find_missing_api_keys(cfg)
        assert any("GEMINI_API_KEY" in m for m in msgs), msgs


class TestFindOptionalUnsetKeys:
    def test_openalex_key_warned_when_absent(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        cfg = _settings()
        warnings = find_optional_unset_keys(cfg)
        assert any("OPENALEX_API_KEY" in w for w in warnings)

    def test_openalex_key_set_no_warning(self):
        cfg = _settings(openalex_api_key="present")
        warnings = find_optional_unset_keys(cfg)
        assert not any("OPENALEX_API_KEY" in w for w in warnings)

    def test_never_blocks_a_run(self):
        """The contract: every entry must be a plain string, never a
        blocker. If future code ever returns a tuple / raises, this
        test breaks early."""
        cfg = _settings()
        warnings = find_optional_unset_keys(cfg)
        assert all(isinstance(w, str) for w in warnings)
