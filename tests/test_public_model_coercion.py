"""The public server coerces a stale/unsupported screening model (e.g. a
legacy 'stub' session persisted in the volume) back to a real catalog model,
so a snowball run can never silently screen with the accept-all stub.
"""

from __future__ import annotations

from web.live.backend import models_catalog as mc
from web.public.backend.server import _effective_model


def test_stub_coerced_to_default():
    assert _effective_model("stub") == mc.SUPPORTED_MODEL


def test_empty_or_none_coerced_to_default():
    assert _effective_model("") == mc.SUPPORTED_MODEL
    assert _effective_model(None) == mc.SUPPORTED_MODEL
    assert _effective_model("   ") == mc.SUPPORTED_MODEL


def test_unknown_coerced_to_default():
    assert _effective_model("made-up-model") == mc.SUPPORTED_MODEL


def test_real_model_kept():
    assert _effective_model("gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"
    assert _effective_model("gpt-5.6-sol") == "gpt-5.6-sol"
    # alias resolves but the stored value is kept verbatim (a real catalog id)
    assert _effective_model("gemini-3.1-flash-lite-preview") == "gemini-3.1-flash-lite-preview"
