"""Prompts for LLM-based topic naming.

Used by :mod:`citeclaw.cluster.representation` to label clusters produced by
any clusterer (graph-based or embedding-based). The user prompt is adapted
from BERTopic's default OpenAI representation prompt
(https://github.com/MaartenGr/BERTopic, MIT License, Maarten Grootendorst).

The system prompt mentions the literal token ``"topic_label"`` so the
offline stub responder can recognise the shape and return a wrapped JSON
document with deterministic placeholder values.
"""

from __future__ import annotations

SYSTEM = (
    "You are an expert at labeling research-paper topic clusters concisely.\n"
    "Given a list of keywords and a few representative papers from one cluster,\n"
    "produce a short, descriptive topic_label and a one-sentence summary.\n"
    "Output only valid JSON."
)

USER_TEMPLATE = (
    "I have a topic that is described by the following keywords:\n"
    "{keywords}\n\n"
    "The topic contains the following representative documents:\n"
    "{documents}\n\n"
    "Based on the information above, give a short, descriptive topic_label of\n"
    "at most 5 words for this cluster, followed by a one-sentence summary.\n"
    'Respond with JSON of the form '
    '{{"topic_label": "<short label>", "summary": "<one sentence>"}}.'
)
