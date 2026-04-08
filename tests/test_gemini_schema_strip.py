"""Tests for Gemini's schema-stripping helper.

Gemini's ``response_schema`` validator (Google API) does not accept the
``additionalProperties`` keyword and rejects the whole request with HTTP
400 ``INVALID_ARGUMENT`` if it sees one. The screening code uses a single
shared schema (``citeclaw.screening.schemas.screening_json_schema``) that
includes ``additionalProperties: false`` for OpenAI strict mode, so the
Gemini client strips it on the way out.
"""

from __future__ import annotations

from citeclaw.clients.llm.gemini import _strip_additional_properties
from citeclaw.screening.schemas import screening_json_schema


def test_strip_top_level():
    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    out = _strip_additional_properties(schema)
    assert "additionalProperties" not in out
    assert out["type"] == "object"


def test_strip_nested():
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"x": {"type": "integer"}},
                },
            },
        },
        "additionalProperties": False,
    }
    out = _strip_additional_properties(schema)
    assert "additionalProperties" not in out
    assert "additionalProperties" not in out["properties"]["results"]["items"]
    # Other fields untouched.
    assert out["properties"]["results"]["items"]["properties"]["x"]["type"] == "integer"


def test_strip_snake_case_variant():
    """Some serializers emit ``additional_properties`` (snake) — strip both."""
    schema = {"type": "object", "additional_properties": False}
    out = _strip_additional_properties(schema)
    assert "additional_properties" not in out


def test_strip_screening_schema():
    """The shared screening schema must come out clean for Gemini."""
    out = _strip_additional_properties(screening_json_schema())
    # Walk recursively and assert no additionalProperties anywhere.
    def _walk(node):
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            assert "additional_properties" not in node
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)
    _walk(out)
    # Structure preserved: still a top-level object with results array.
    assert out["type"] == "object"
    assert out["properties"]["results"]["type"] == "array"
    assert out["properties"]["results"]["items"]["properties"]["index"]["type"] == "integer"


def test_strip_idempotent():
    """Running the stripper twice is a no-op the second time."""
    schema = screening_json_schema()
    once = _strip_additional_properties(schema)
    twice = _strip_additional_properties(once)
    assert once == twice


def test_non_dict_passthrough():
    """Non-dict / non-list values are returned unchanged."""
    assert _strip_additional_properties(42) == 42
    assert _strip_additional_properties("hello") == "hello"
    assert _strip_additional_properties(None) is None
    assert _strip_additional_properties([1, 2, 3]) == [1, 2, 3]
