"""Schema-handling helpers shared across LLM clients.

Both OpenAI and Gemini clients need to:

* Honour the ``_strict_openai`` sentinel that callers may include in a
  schema dict to opt out of OpenAI's strict-mode constraints. The
  marker is provider-routing metadata, not a JSON Schema keyword, and
  must be stripped before the schema reaches either provider's wire.
* Strip provider-incompatible fields. Gemini's ``response_schema``
  validator (Google API) does not accept ``additionalProperties`` and
  rejects the whole request with HTTP 400 ``INVALID_ARGUMENT`` if it
  sees one. OpenAI strict mode requires it, so we keep it in the
  source schema and strip on the way out to Gemini only.

Centralised here so a third structured-output provider doesn't have
to reinvent both behaviours.
"""

from __future__ import annotations

from typing import Any

# JSON-Schema-shaped keys that no provider should receive on the wire.
_DROP_FOR_GEMINI = frozenset({
    "additionalProperties",
    "additional_properties",
    # Internal routing sentinel; meaningless to Gemini and rejected by
    # its validator if forwarded.
    "_strict_openai",
})


def pop_strict_openai(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``(schema_copy_without_marker, strict_flag)``.

    Default ``strict=True`` — opt out by including
    ``_strict_openai: False`` in the schema. The marker is consumed so
    it never reaches OpenAI's wire (OpenAI ignores unknown keys, but
    keeping wire payloads tidy helps debugging).

    The returned schema is a *shallow* copy: the top-level marker is
    removed but nested dicts / lists are shared with the input. Callers
    that need a fully independent schema should ``copy.deepcopy`` first.
    """
    copy = dict(schema)
    strict = bool(copy.pop("_strict_openai", True))
    return copy, strict


def strip_for_gemini(schema: Any) -> Any:
    """Recursively remove keys Gemini's response-schema validator rejects.

    Walks dicts and lists; leaves primitives alone. Returns a fully
    rebuilt structure — the input is not mutated, and the result shares
    no mutable containers with it. Used by
    :class:`citeclaw.clients.llm.gemini.GeminiClient` so the shared
    schema definitions in :mod:`citeclaw.screening.schemas` work for
    both providers without maintaining two copies.
    """
    if isinstance(schema, dict):
        return {
            k: strip_for_gemini(v)
            for k, v in schema.items()
            if k not in _DROP_FOR_GEMINI
        }
    if isinstance(schema, list):
        return [strip_for_gemini(v) for v in schema]
    return schema
