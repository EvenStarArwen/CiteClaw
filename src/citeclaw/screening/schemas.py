"""Shared JSON Schema definitions for structured LLM screening output.

All screening calls use a single canonical schema: an object with a
``results`` array of ``{index: int, match: bool}`` entries, one per
paper in the batch. The array is wrapped in an object because OpenAI's
``response_format={"type": "json_schema", "strict": True}`` requires
the root to be an object, not a bare array.

Both helpers below build a fresh dict on every call so that downstream
sanitisers (e.g. :func:`citeclaw.clients.llm._schema.strip_for_gemini`)
can safely mutate the result without affecting future calls.
"""

from __future__ import annotations

from typing import Any

SCREENING_SCHEMA_NAME = "citeclaw_screening_results"


def _result_item_schema() -> dict[str, Any]:
    """Inner item schema — one ``{index, match}`` entry."""
    return {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "match": {"type": "boolean"},
        },
        "required": ["index", "match"],
        "additionalProperties": False,
    }


def screening_json_schema() -> dict[str, Any]:
    """JSON Schema for one screening batch's response.

    Shape::

        {"results": [{"index": 1, "match": true}, ...]}
    """
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": _result_item_schema(),
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def openai_response_format() -> dict[str, Any]:
    """OpenAI / vLLM ``response_format`` kwarg for ``chat.completions.create``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": SCREENING_SCHEMA_NAME,
            "strict": True,
            "schema": screening_json_schema(),
        },
    }
