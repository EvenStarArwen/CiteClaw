"""Tests for pdfclaw's LLM finder routing through the unified citeclaw client.

The LLM finder used to call ``httpx.post`` directly with a hardcoded
``{base_url, api_key, model}`` tuple per env-var detection branch. These
tests pin the new behaviour: the env-var chain now builds a
:class:`citeclaw.clients.llm.base.LLMClient` via
:func:`citeclaw.clients.llm.build_llm_client`, so any provider the main
CiteClaw registry supports (xAI Grok, Together AI, Mistral, ...) is
automatically available here too.

Covers:
  * Each of the three env-var detection branches returns a client and
    the expected model name.
  * Missing-env-vars path returns ``None``.
  * ``_call_llm`` surfaces the client's text and parses JSON correctly.
"""

from __future__ import annotations

import pytest

from pdfclaw.publishers.llm_finder import _build_llm_client, _call_llm


class TestBuildLlmClient:
    """Env-var priority order: PDFCLAW_LLM_* > GEMINI_API_KEY > CITECLAW_VLLM_*."""

    def _clear_env(self, monkeypatch):
        for k in (
            "PDFCLAW_LLM_BASE_URL",
            "PDFCLAW_LLM_API_KEY",
            "PDFCLAW_LLM_MODEL",
            "GEMINI_API_KEY",
            "CITECLAW_VLLM_API_KEY",
            "CITECLAW_VLLM_BASE_URL",
            "OPENAI_API_KEY",
            "CITECLAW_OPENAI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_returns_none_when_no_env_vars(self, monkeypatch):
        self._clear_env(monkeypatch)
        assert _build_llm_client() is None

    def test_pdfclaw_llm_vars_win_over_others(self, monkeypatch):
        """Explicit PDFCLAW_LLM_* config wins over GEMINI_API_KEY and the
        CITECLAW_VLLM_* fallback, even when all three are set."""
        from citeclaw.clients.llm.openai_client import OpenAIClient

        self._clear_env(monkeypatch)
        monkeypatch.setenv("PDFCLAW_LLM_BASE_URL", "https://api.x.ai/v1")
        monkeypatch.setenv("PDFCLAW_LLM_API_KEY", "fake-grok-key")
        monkeypatch.setenv("PDFCLAW_LLM_MODEL", "grok-4")
        monkeypatch.setenv("GEMINI_API_KEY", "should-be-ignored")
        monkeypatch.setenv("CITECLAW_VLLM_API_KEY", "should-also-be-ignored")
        monkeypatch.setenv("CITECLAW_VLLM_BASE_URL", "https://should-be-ignored/v1")
        built = _build_llm_client()
        assert built is not None
        client, model = built
        assert model == "grok-4"
        # PDFCLAW_LLM_* routes through the legacy llm_base_url path, which
        # always yields OpenAIClient (vLLM / Grok / Together / Mistral are
        # all OpenAI-compatible).
        assert isinstance(client, OpenAIClient)

    def test_gemini_key_routes_to_native_gemini_client(self, monkeypatch):
        """A plain GEMINI_API_KEY (no PDFCLAW_LLM_*) routes through the
        native GeminiClient. This is stricter than the old OpenAI-compat
        shim pdfclaw used to use."""
        from citeclaw.clients.llm.gemini import GeminiClient

        self._clear_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
        built = _build_llm_client()
        assert built is not None
        client, model = built
        assert model.startswith("gemini-")
        assert isinstance(client, GeminiClient)

    def test_vllm_env_vars_last_resort(self, monkeypatch):
        """CITECLAW_VLLM_* fires only when neither PDFCLAW_LLM_* nor
        GEMINI_API_KEY is set — this is the default for fresh CiteClaw
        users who have Modal Gemma configured."""
        from citeclaw.clients.llm.openai_client import OpenAIClient

        self._clear_env(monkeypatch)
        monkeypatch.setenv(
            "CITECLAW_VLLM_BASE_URL",
            "https://fake-modal.example/v1",
        )
        monkeypatch.setenv("CITECLAW_VLLM_API_KEY", "fake-vllm-key")
        built = _build_llm_client()
        assert built is not None
        client, model = built
        assert isinstance(client, OpenAIClient)
        assert "gemma" in model.lower()

    def test_model_env_var_overrides_default(self, monkeypatch):
        """PDFCLAW_LLM_MODEL overrides the per-branch default model name
        — lets a user plug in any model string the endpoint supports."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("PDFCLAW_LLM_BASE_URL", "https://api.together.xyz/v1")
        monkeypatch.setenv("PDFCLAW_LLM_API_KEY", "fake-together-key")
        monkeypatch.setenv(
            "PDFCLAW_LLM_MODEL",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
        built = _build_llm_client()
        assert built is not None
        _, model = built
        assert model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class TestCallLlm:
    """The _call_llm helper now accepts any LLMClient and parses the
    returned text as JSON."""

    class _CannedClient:
        def __init__(self, text: str):
            self._text = text
            self.calls: list[tuple[str, str]] = []

        def call(self, system, user, **kw):
            from citeclaw.clients.llm.base import LLMResponse

            self.calls.append((system, user))
            return LLMResponse(text=self._text)

    def test_extracts_json_object(self):
        c = self._CannedClient('{"action": "fetch", "target": 0}')
        out = _call_llm(c, "sys", "user")
        assert out == {"action": "fetch", "target": 0}

    def test_extracts_json_from_markdown_prose(self):
        """Some models prefix JSON with reasoning — the finder must still
        extract the JSON block."""
        c = self._CannedClient(
            'I think we should click the PDF button. {"action": "click", "target": 2}'
        )
        out = _call_llm(c, "sys", "user")
        assert out == {"action": "click", "target": 2}

    def test_returns_none_on_empty_response(self):
        c = self._CannedClient("")
        assert _call_llm(c, "sys", "user") is None

    def test_returns_none_on_no_json(self):
        c = self._CannedClient("Sorry, I cannot find a PDF on this page.")
        assert _call_llm(c, "sys", "user") is None

    def test_returns_none_on_malformed_json(self):
        c = self._CannedClient('{"action": "click", "target":}')
        assert _call_llm(c, "sys", "user") is None

    def test_propagates_call_through_client(self):
        """The helper uses the injected client's ``call`` method, not any
        hardcoded HTTP shape. This is the whole point of the refactor."""
        c = self._CannedClient('{"action": "give_up"}')
        _call_llm(c, "my system", "my user")
        assert c.calls == [("my system", "my user")]

    def test_returns_none_on_client_exception(self):
        """Client exceptions get logged and turned into None so the agent
        loop can record a history line and continue."""
        class _BoomClient:
            def call(self, *a, **k):
                raise RuntimeError("provider down")

        assert _call_llm(_BoomClient(), "s", "u") is None
