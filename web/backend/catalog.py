"""Model catalog shown by the local Web UI.

Only text-generation models that can plausibly drive CiteClaw's screening
pipeline are included. Prices are standard, paid-tier USD per one million
tokens and were transcribed from the providers' official pricing pages on
2026-07-16. The catalog is informational: the first live release deliberately
allows only ``gemini-3.1-flash-lite-preview`` to start a run.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_MODEL = "gemini-3.1-flash-lite-preview"
SUPPORTED_REASONING_EFFORT = "minimal"
PRICING_UPDATED_AT = "2026-07-16"

PRICING_SOURCES = {
    "openai": "https://developers.openai.com/api/docs/pricing",
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
}


def _model(
    provider: str,
    model_id: str,
    input_usd: float,
    output_usd: float,
    *,
    cached_input_usd: float | None = None,
    long_context: dict[str, float] | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "id": model_id,
        "input_usd_per_m": input_usd,
        "cached_input_usd_per_m": cached_input_usd,
        "output_usd_per_m": output_usd,
        "long_context": long_context,
        "note": note,
        "runnable": model_id == SUPPORTED_MODEL,
    }


MODEL_CATALOG: list[dict[str, Any]] = [
    # OpenAI flagship text models. Short-context standard pricing is shown in
    # the primary columns; threshold pricing, where present, is attached.
    _model(
        "openai",
        "gpt-5.6-sol",
        5.00,
        30.00,
        cached_input_usd=0.50,
        long_context={"input_usd_per_m": 10.00, "cached_input_usd_per_m": 1.00, "output_usd_per_m": 45.00},
    ),
    _model(
        "openai",
        "gpt-5.6-terra",
        2.50,
        15.00,
        cached_input_usd=0.25,
        long_context={"input_usd_per_m": 5.00, "cached_input_usd_per_m": 0.50, "output_usd_per_m": 22.50},
    ),
    _model(
        "openai",
        "gpt-5.6-luna",
        1.00,
        6.00,
        cached_input_usd=0.10,
        long_context={"input_usd_per_m": 2.00, "cached_input_usd_per_m": 0.20, "output_usd_per_m": 9.00},
    ),
    _model(
        "openai",
        "gpt-5.5",
        5.00,
        30.00,
        cached_input_usd=0.50,
        long_context={"input_usd_per_m": 10.00, "cached_input_usd_per_m": 1.00, "output_usd_per_m": 45.00},
    ),
    _model("openai", "gpt-5.5-pro", 30.00, 180.00, long_context={"input_usd_per_m": 60.00, "output_usd_per_m": 270.00}),
    _model(
        "openai",
        "gpt-5.4",
        2.50,
        15.00,
        cached_input_usd=0.25,
        long_context={"input_usd_per_m": 5.00, "cached_input_usd_per_m": 0.50, "output_usd_per_m": 22.50},
    ),
    _model("openai", "gpt-5.4-mini", 0.75, 4.50, cached_input_usd=0.075),
    _model("openai", "gpt-5.4-nano", 0.20, 1.25, cached_input_usd=0.02),
    _model("openai", "gpt-5.4-pro", 30.00, 180.00, long_context={"input_usd_per_m": 60.00, "output_usd_per_m": 270.00}),
    # Gemini general-purpose text models. Audio/image-specific rates and
    # models are intentionally omitted because CiteClaw sends text and expects
    # structured text in return.
    _model("gemini", "gemini-3.5-flash", 1.50, 9.00, cached_input_usd=0.15),
    _model("gemini", "gemini-3.1-flash-lite", 0.25, 1.50, cached_input_usd=0.025),
    _model(
        "gemini",
        SUPPORTED_MODEL,
        0.25,
        1.50,
        cached_input_usd=0.025,
        note="Compatibility preview slug requested for this release; priced as Gemini 3.1 Flash-Lite.",
    ),
    _model(
        "gemini",
        "gemini-3.1-pro-preview",
        2.00,
        12.00,
        cached_input_usd=0.20,
        long_context={
            "threshold_tokens": 200_000,
            "input_usd_per_m": 4.00,
            "cached_input_usd_per_m": 0.40,
            "output_usd_per_m": 18.00,
        },
    ),
    _model("gemini", "gemini-3-flash-preview", 0.50, 3.00, cached_input_usd=0.05),
    _model(
        "gemini",
        "gemini-2.5-pro",
        1.25,
        10.00,
        cached_input_usd=0.125,
        long_context={
            "threshold_tokens": 200_000,
            "input_usd_per_m": 2.50,
            "cached_input_usd_per_m": 0.25,
            "output_usd_per_m": 15.00,
        },
    ),
    _model("gemini", "gemini-2.5-flash", 0.30, 2.50, cached_input_usd=0.03),
    _model("gemini", "gemini-2.5-flash-lite", 0.10, 0.40, cached_input_usd=0.01),
    _model("gemini", "gemini-2.5-flash-lite-preview-09-2025", 0.10, 0.40, cached_input_usd=0.01),
]


def catalog_payload() -> dict[str, Any]:
    return {
        "models": MODEL_CATALOG,
        "supported_model": SUPPORTED_MODEL,
        "supported_reasoning_effort": SUPPORTED_REASONING_EFFORT,
        "pricing_updated_at": PRICING_UPDATED_AT,
        "sources": PRICING_SOURCES,
        "scope": "Text-generation models suitable for CiteClaw's screening pipeline.",
    }
