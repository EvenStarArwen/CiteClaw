"""Shared JSON Schema definitions for structured LLM output.

All screening calls use a single canonical schema: an object with a
``results`` array of ``{index: int, match: bool}`` entries. We wrap the
array in an object because OpenAI's ``response_format={"type":
"json_schema"}`` with ``strict: true`` requires the root to be an object.
"""

from __future__ import annotations

from typing import Any

SCREENING_SCHEMA_NAME = "citeclaw_screening_results"


def screening_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for a screening batch response.

    Shape::

        {"results": [{"index": 1, "match": true}, ...]}
    """
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "match": {"type": "boolean"},
                    },
                    "required": ["index", "match"],
                    "additionalProperties": False,
                },
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
