"""Model picker catalog (OpenAI + Gemini) with current prices + support guard.

Every catalog model is runnable: Gemini ids route through ``GeminiClient``
(thinking level from ``reasoning_effort``), OpenAI ids through
``OpenAIClient`` (native ``reasoning_effort`` for the reasoning models).
Models flagged ``reasoning: False`` (chat aliases / budget tiers) run with
the effort silently dropped — see :func:`effort_for` — so a stored effort
never 400s a non-reasoning model. Ids outside the catalog are refused in
the run endpoint (a typo'd or retired model fails before tokens are spent).

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

# The default the Settings picker preloads.
SUPPORTED_MODEL = "gemini-3.1-flash-lite"
_ALIASES = {"gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite"}
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
    # Rejects thinking level MINIMAL (400, probed 2026-07-19) — "efforts"
    # lists what the API accepts; effort_for clamps minimal → low.
    {"id": "gemini-3.1-pro-preview", "provider": "gemini", "label": "Gemini 3.1 Pro (preview)",
     "input": 2.00, "output": 12.00, "reasoning": True,
     "efforts": ["low", "medium", "high"],
     "note": "≤200k tokens; higher above · minimal thinking auto-raises to low"},
    # reasoning=False: the 2.5 generation rejects ``thinking_level`` (400
    # "Thinking level is not supported for this model" — that config knob is
    # Gemini 3.x-only), so the app sends no thinking config and the model
    # runs with its own default thinking. Probed 2026-07-19.
    {"id": "gemini-2.5-flash", "provider": "gemini", "label": "Gemini 2.5 Flash",
     "input": 0.30, "output": 2.50, "reasoning": False,
     "note": "thinks by default; effort knob not supported"},
    # gemini-2.5-pro and gemini-2.5-flash-lite were dropped 2026-07-19:
    # the API now 404s them for new users ("no longer available").
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


_BY_ID = {m["id"]: m for m in _CATALOG}


def resolve_model(model: str) -> str:
    """Map the requested id to the id we actually send to the provider."""
    m = (model or "").strip()
    return _ALIASES.get(m, m)


def is_supported(model: str, effort: str) -> bool:
    m = resolve_model(model)
    e = (effort or DEFAULT_EFFORT).strip().lower()
    if m.lower() == "stub" and _stub_allowed():
        return True
    return m in _BY_ID and e in _ALLOWED_EFFORTS


def required_key(model: str) -> str | None:
    """Settings-key field this model needs (None for stub/unknown)."""
    m = resolve_model(model)
    meta = _BY_ID.get(m)
    if meta is None:
        return None
    return "gemini_api_key" if meta["provider"] == "gemini" else "openai_api_key"


_EFFORT_ORDER = ("minimal", "low", "medium", "high")


def effort_for(model: str, effort: str) -> str:
    """Effort to actually send.

    * Non-reasoning catalog models get ``""`` — chat aliases and the 2.5
      generation reject (or ignore inconsistently) reasoning parameters,
      so a stored effort must not follow the user onto them.
    * Models with a per-model ``efforts`` list get the nearest accepted
      level (stepping up first): gemini-3.1-pro-preview 400s on
      ``minimal``, so it becomes ``low`` instead of failing the run.
    """
    meta = _BY_ID.get(resolve_model(model))
    if meta is None:
        return effort
    if not meta.get("reasoning", False):
        return ""
    allowed = meta.get("efforts")
    e = (effort or "").strip().lower()
    if not allowed or e in allowed or e not in _EFFORT_ORDER:
        return effort
    i = _EFFORT_ORDER.index(e)
    for j in list(range(i + 1, len(_EFFORT_ORDER))) + list(range(i - 1, -1, -1)):
        if _EFFORT_ORDER[j] in allowed:
            return _EFFORT_ORDER[j]
    return effort


def support_error(model: str, effort: str) -> str:
    e = (effort or "").strip().lower()
    if resolve_model(model) in _BY_ID and e not in _ALLOWED_EFFORTS:
        return (f"Reasoning effort '{effort}' is not recognised — pick one of "
                f"{sorted(_ALLOWED_EFFORTS)} in Settings.")
    return (
        f"Model '{model}' is not in the supported catalog. Pick any model "
        f"from the Settings list (Gemini needs a Gemini API key, OpenAI an "
        f"OpenAI key) and try again."
    )


def catalog() -> list[dict]:
    """Return the model list annotated with support + provider-key needs."""
    out = []
    for m in _CATALOG:
        out.append({
            **m,
            "supported": True,
            "requires_key": "gemini_api_key" if m["provider"] == "gemini" else "openai_api_key",
        })
    return out
