"""Prompts for LLM-based topic naming.

Used by :mod:`citeclaw.cluster.representation` to label clusters produced by
any clusterer (graph-based or embedding-based). The single-cluster user
prompt is adapted from BERTopic's default OpenAI representation prompt
(https://github.com/MaartenGr/BERTopic, MIT License, Maarten Grootendorst).

Both the single-cluster and batched forms intentionally mention the
literal token ``"topic_label"`` so the offline stub responder can
recognise the shape and return deterministic placeholder JSON. The
batched form additionally embeds ``cluster_id=`` blocks so the stub
can count clusters and emit one result per input.
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

# ---------------------------------------------------------------------------
# Batched form: one LLM call labels N clusters at once.
# ---------------------------------------------------------------------------

BATCH_SYSTEM = (
    "You are an expert at labeling research-paper topic clusters concisely.\n"
    "You will receive MULTIPLE clusters in one message, each with its own\n"
    "``cluster_id``, a keyword list, and representative papers. Produce one\n"
    "short, descriptive topic_label (at most 5 words) and one-sentence summary\n"
    "per cluster, preserving the given cluster_id exactly.\n"
    'Output only valid JSON of the form '
    '{"results": [{"cluster_id": <int>, "topic_label": "<label>", "summary": "<sentence>"}, ...]}.\n'
    "Return exactly one entry per input cluster."
)

BATCH_USER_TEMPLATE = (
    "I have {n} research-paper topic clusters. Label each one concisely.\n\n"
    "{clusters}\n\n"
    'Return one JSON entry per cluster_id with "topic_label" and "summary", '
    "in the same order as given."
)

CLUSTER_BLOCK_TEMPLATE = (
    "## cluster_id={cid}\n"
    "Keywords: {keywords}\n"
    "Representative papers:\n{documents}\n"
)
