"""Prompts for batched query screening (LLMFilter).

The user prompt mentions the exact ``"match"`` token verbatim so the offline
stub responder recognises the shape and returns a wrapped ``{"results": [...]}``
JSON document. Real providers are additionally constrained at decode time
via ``response_format`` / ``response_schema`` so a close-brace typo can no
longer nuke a whole batch.
"""

from __future__ import annotations

SYSTEM = (
    "For each item below, determine if it matches the given criterion.\n"
    "Answer YES (match) or NO (no match) for each item.\n"
    "Evaluate each item independently.\n"
    "Output only valid JSON."
)

VENUE_SYSTEM = (
    "For each venue name below, determine if it matches the given criterion.\n"
    "Match by identity — treat abbreviations and full names as the same venue.\n"
    "Answer YES (match) or NO (no match) for each.\n"
    "Output only valid JSON."
)

USER_TEMPLATE = (
    'Criterion: "{criterion}"\n\n'
    "{label} ({n} items — return exactly {n} results):\n"
    "{block}\n\n"
    'Respond with JSON of the form '
    '{{"results": [{{"index": 1, "match": true}}, ...]}} '
    "containing exactly {n} items."
)
