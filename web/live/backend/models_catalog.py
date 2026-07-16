"""Model picker catalog (OpenAI + Gemini) with current prices + support guard.

The design's Settings panel lists every current model with its price, but
this first live version only actually *runs* Gemini 3.1 Flash-Lite with
minimal reasoning effort. Selecting any other model raises a clear
"not supported yet" error (enforced in the run endpoint).

Prices are USD per 1M tokens, standard on-demand tier, sourced from the
official OpenAI and Google pricing pages (July 2026). OpenAI legacy
gpt-4o/4.1/o3/o4-mini were dropped from the picker because they are no
longer on the official pricing page and are on retirement paths.

NOTE: the exact id ``gemini-3.1-flash-lite-preview`` (as originally
requested) does not exist — Gemini 3.1 Flash-Lite reached GA, so the live
id is ``gemini-3.1-flash-lite``. We accept the ``-preview`` spelling as an
alias and run it against the GA id so either selection works.
"""

from __future__ import annotations

import os

# The one model this version actually runs, plus the requested alias.
SUPPORTED_MODEL = "gemini-3.1-flash-lite"
_SUPPORTED_ALIASES = {"gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"}
DEFAULT_EFFORT = "minimal"
_ALLOWED_EFFORTS = {"minimal", "low", "medium", "high"}

# Optional dev bypass so the whole live loop can be exercised offline with
# the deterministic StubClient (no API spend). The UI never sends "stub".
def _stub_allowed() -> bool:
    return os.environ.get("CITECLAW_WEBUI_ALLOW_STUB", "") not in ("", "0", "false")


# id, provider, label, input$/Mtok, output$/Mtok, reasoning, note
_CATALOG: list[dict] = [
    # ---- Google Gemini (current) ----
    {"id": "gemini-3.1-flash-lite", "provider": "gemini", "label": "Gemini 3.1 Flash-Lite",
     "input": 0.25, "output": 1.50, "reasoning": True,
     "note": "1M ctx · configurable thinking (minimal/low/medium/high)"},
    {"id": "gemini-3.5-flash", "provider": "gemini", "label": "Gemini 3.5 Flash",
     "input": 1.50, "output": 9.00, "reasoning": True, "note": "thinking model"},
    {"id": "gemini-3.1-pro-preview", "provider": "gemini", "label": "Gemini 3.1 Pro (preview)",
     "input": 2.00, "output": 12.00, "reasoning": True, "note": "≤200k tokens; higher above"},
    {"id": "gemini-2.5-pro", "provider": "gemini", "label": "Gemini 2.5 Pro",
     "input": 1.25, "output": 10.00, "reasoning": True, "note": "≤200k tokens; higher above"},
    {"id": "gemini-2.5-flash", "provider": "gemini", "label": "Gemini 2.5 Flash",
     "input": 0.30, "output": 2.50, "reasoning": True, "note": "thinking model"},
    {"id": "gemini-2.5-flash-lite", "provider": "gemini", "label": "Gemini 2.5 Flash-Lite",
     "input": 0.10, "output": 0.40, "reasoning": False, "note": "budget tier"},
    # ---- OpenAI (current) ----
    {"id": "gpt-5.6-sol", "provider": "openai", "label": "GPT-5.6 Sol",
     "input": 5.00, "output": 30.00, "reasoning": True, "note": "flagship"},
    {"id": "gpt-5.6-terra", "provider": "openai", "label": "GPT-5.6 Terra",
     "input": 2.50, "output": 15.00, "reasoning": True, "note": "balanced"},
    {"id": "gpt-5.6-luna", "provider": "openai", "label": "GPT-5.6 Luna",
     "input": 1.00, "output": 6.00, "reasoning": True, "note": "latency-optimized"},
    {"id": "gpt-5.6-chat-latest", "provider": "openai", "label": "GPT-5.6 Chat (latest)",
     "input": 5.00, "output": 30.00, "reasoning": False, "note": "non-reasoning chat alias"},
    {"id": "gpt-5.5", "provider": "openai", "label": "GPT-5.5",
     "input": 5.00, "output": 30.00, "reasoning": True, "note": "prior flagship"},
    {"id": "gpt-5.5-pro", "provider": "openai", "label": "GPT-5.5 Pro",
     "input": 30.00, "output": 180.00, "reasoning": True, "note": "high-compute"},
    {"id": "gpt-5.4", "provider": "openai", "label": "GPT-5.4",
     "input": 2.50, "output": 15.00, "reasoning": True, "note": ""},
    {"id": "gpt-5.4-mini", "provider": "openai", "label": "GPT-5.4 mini",
     "input": 0.75, "output": 4.50, "reasoning": True, "note": ""},
    {"id": "gpt-5.4-nano", "provider": "openai", "label": "GPT-5.4 nano",
     "input": 0.20, "output": 1.25, "reasoning": True, "note": "cheapest OpenAI"},
    {"id": "gpt-5.4-pro", "provider": "openai", "label": "GPT-5.4 Pro",
     "input": 30.00, "output": 180.00, "reasoning": True, "note": ""},
    {"id": "gpt-5.3-codex", "provider": "openai", "label": "GPT-5.3 Codex",
     "input": 1.75, "output": 14.00, "reasoning": True, "note": "coding-specialized"},
]


def resolve_model(model: str) -> str:
    """Map the requested id to the id we actually send to the provider."""
    m = (model or "").strip()
    if m in _SUPPORTED_ALIASES:
        return SUPPORTED_MODEL
    return m


def is_supported(model: str, effort: str) -> bool:
    m = (model or "").strip()
    e = (effort or DEFAULT_EFFORT).strip().lower()
    if m.lower() == "stub" and _stub_allowed():
        return True
    return m in _SUPPORTED_ALIASES and e in _ALLOWED_EFFORTS


def support_error(model: str, effort: str) -> str:
    return (
        f"Model '{model}' with reasoning effort '{effort}' is not supported in this "
        f"version. This first live release only runs '{SUPPORTED_MODEL}' "
        f"(the model you asked for, 'gemini-3.1-flash-lite-preview', is accepted as an "
        f"alias for it) with reasoning effort one of {sorted(_ALLOWED_EFFORTS)}. "
        f"Pick that model in Settings and try again."
    )


def catalog() -> list[dict]:
    """Return the model list annotated with support + provider-key needs."""
    out = []
    for m in _CATALOG:
        supported = m["id"] in _SUPPORTED_ALIASES
        out.append({
            **m,
            "supported": supported,
            "requires_key": "gemini_api_key" if m["provider"] == "gemini" else "openai_api_key",
        })
    return out
