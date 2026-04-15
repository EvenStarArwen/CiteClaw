"""Prompts for graph node labelling (used by ``citeclaw.annotate``).

Two shapes are supported:

* **Single-paper** (``SYSTEM`` / ``USER_TEMPLATE``): returns free-text label
  only. Retained for legacy callers and per-paper fallback.
* **Batched** (``BATCH_SYSTEM`` / ``BATCH_USER_TEMPLATE``): labels N papers
  per LLM call and returns structured JSON
  ``{"results": [{"index": <int>, "label": "<text>"}, ...]}``. The stub
  responder recognises the ``Paper index=`` marker in ``PAPER_BLOCK_TEMPLATE``
  to return deterministic JSON with one entry per input paper.
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

BATCH_SYSTEM = (
    "You are labelling papers in a citation network.\n"
    "For EACH input paper, generate a concise label based on the instruction.\n"
    'Output only valid JSON of the form '
    '{"results": [{"index": <int>, "label": "<text>"}, ...]}.\n'
    "Preserve the given index exactly. Return one entry per input paper."
)

BATCH_USER_TEMPLATE = (
    "Instruction: {instruction}\n\n"
    "Label each of the following {n} papers. "
    "Respond with JSON containing a ``label`` string per ``index``.\n\n"
    "{papers}"
)

PAPER_BLOCK_TEMPLATE = (
    "Paper index={idx}\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n"
)
