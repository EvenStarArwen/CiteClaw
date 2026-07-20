"""Tests for the widened model catalog + OpenAI reasoning detection."""

from __future__ import annotations

from citeclaw.clients.llm.openai_client import _is_openai_reasoning_model
from web.live.backend import models_catalog as mc


class TestCatalogGate:
    def test_all_catalog_models_supported(self):
        for m in mc.catalog():
            assert m["supported"] is True
            assert mc.is_supported(m["id"], "medium")

    def test_alias_still_resolves(self):
        assert mc.resolve_model("gemini-3.1-flash-lite-preview") == "gemini-3.1-flash-lite"
        assert mc.is_supported("gemini-3.1-flash-lite-preview", "minimal")

    def test_unknown_model_refused(self):
        assert not mc.is_supported("gpt-4o", "medium")
        assert "not in the supported catalog" in mc.support_error("gpt-4o", "medium")

    def test_bad_effort_refused_with_specific_error(self):
        assert not mc.is_supported("gpt-5.6-sol", "ultra")
        assert "Reasoning effort" in mc.support_error("gpt-5.6-sol", "ultra")

    def test_stub_gated_by_env(self, monkeypatch):
        monkeypatch.delenv("CITECLAW_WEBUI_ALLOW_STUB", raising=False)
        assert not mc.is_supported("stub", "minimal")
        monkeypatch.setenv("CITECLAW_WEBUI_ALLOW_STUB", "1")
        assert mc.is_supported("stub", "minimal")


class TestIsCatalogModel:
    """``is_catalog_model`` — used to coerce a stale/unsupported stored model
    (a legacy 'stub' session) back to a real one so screening never silently
    runs on the accept-all stub."""

    def test_real_models_and_alias(self):
        assert mc.is_catalog_model("gemini-3.1-flash-lite")
        assert mc.is_catalog_model("gemini-3.1-flash-lite-preview")  # alias
        assert mc.is_catalog_model("gpt-5.6-sol")

    def test_stub_is_not_a_catalog_model_even_when_allowed(self, monkeypatch):
        # is_supported("stub") flips with the env gate, but a stub is NEVER a
        # real catalog model — coercion must always evict it.
        monkeypatch.setenv("CITECLAW_WEBUI_ALLOW_STUB", "1")
        assert mc.is_supported("stub", "minimal")
        assert not mc.is_catalog_model("stub")

    def test_empty_and_unknown_not_catalog(self):
        assert not mc.is_catalog_model("")
        assert not mc.is_catalog_model("made-up-model")


class TestRequiredKey:
    def test_provider_mapping(self):
        assert mc.required_key("gemini-3.5-flash") == "gemini_api_key"
        assert mc.required_key("gpt-5.6-sol") == "openai_api_key"
        assert mc.required_key("gemini-3.1-flash-lite-preview") == "gemini_api_key"
        assert mc.required_key("stub") is None
        assert mc.required_key("made-up") is None


class TestEffortFor:
    def test_reasoning_models_keep_effort(self):
        assert mc.effort_for("gemini-3.5-flash", "high") == "high"
        assert mc.effort_for("gpt-5.4-nano", "low") == "low"

    def test_non_reasoning_models_drop_effort(self):
        assert mc.effort_for("gpt-5.6-chat-latest", "medium") == ""
        # gemini-2.5-flash rejects thinking_level (3.x-only knob)
        assert mc.effort_for("gemini-2.5-flash", "high") == ""

    def test_per_model_effort_clamp(self):
        # gemini-3.1-pro-preview 400s on minimal → nearest accepted (low)
        assert mc.effort_for("gemini-3.1-pro-preview", "minimal") == "low"
        assert mc.effort_for("gemini-3.1-pro-preview", "high") == "high"

    def test_unknown_passthrough(self):
        assert mc.effort_for("stub", "minimal") == "minimal"


class TestOpenAIReasoningDetection:
    def test_families(self):
        assert _is_openai_reasoning_model("o3-mini")
        assert _is_openai_reasoning_model("gpt-5.6-sol")
        assert _is_openai_reasoning_model("gpt-5.5-pro")
        assert _is_openai_reasoning_model("gpt-5.3-codex")

    def test_chat_alias_and_legacy_are_plain(self):
        assert not _is_openai_reasoning_model("gpt-5.6-chat-latest")
        assert not _is_openai_reasoning_model("gpt-4o")
        assert not _is_openai_reasoning_model("")
