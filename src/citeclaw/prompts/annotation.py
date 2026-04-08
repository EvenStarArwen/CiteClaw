"""Prompts for graph node labelling (used by ``citeclaw.annotate``).

The system prompt instructs the model to return ONLY the label text — no
JSON, no quotes, no commentary — because the response is used as a node
attribute directly.
"""

from __future__ import annotations

SYSTEM = (
    "You are labelling a paper in a citation network.\n"
    "Generate a concise label based on the instruction.\n"
    "Reply with ONLY the label text, nothing else — no quotes, no explanation."
)

USER_TEMPLATE = (
    "Instruction: {instruction}\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n\n"
    "Label:"
)
