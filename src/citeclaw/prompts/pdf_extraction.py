"""Prompts for LLM-based reference extraction from full paper text.

The agent reads a paper's body text and its reference list, then
identifies references that are relevant to a user-specified research
topic.  For each relevant reference it returns:

  - the citation marker (``[23]``, ``(Smith et al., 2020)``, …)
  - the full bibliographic entry from the reference list
  - the *title* extracted from that entry (for S2 resolution)
  - one or more verbatim quotes showing *where* the reference is cited
  - a brief explanation of *why* it is relevant to the topic

The prompt puts the reference list **before** the body text to exploit
primacy bias: the model "loads" the bibliography first, then recognises
markers as it reads through the paper.

The literal ``"relevant_references"`` token ensures the stub responder
can recognise the shape.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are an expert academic literature analyst.  Given the full text "
    "of a research paper and a research topic, identify every reference "
    "cited in the paper that directly matches the topic.\n\n"
    "Rules:\n"
    "1. Be exhaustive over direct matches.  Include EVERY reference in "
    "the bibliography that directly matches the topic — do not cap the "
    "count, do not pick 'favorites', do not skip any that qualify.\n"
    "2. A reference directly matches the topic only when the paper cites "
    "it as the topic entity itself — i.e. the reference IS the thing the "
    "topic asks about (the dataset, the resource, the method, etc.), or "
    "the reference constructs / defines / provides it.  References to "
    "tools, methods, or algorithms that merely *use*, *process*, or "
    "*operate on* the topic entity do NOT match unless the reference "
    "itself IS the topic entity.  When in doubt ask: 'does the cited "
    "quote show the reference IS the topic, or just that it is adjacent "
    "to the topic?'  If the answer is 'adjacent', exclude it.\n"
    "3. Every included reference MUST include at least one verbatim "
    "quote (1–2 sentences) from the paper body that cites it in direct "
    "connection with the topic.  A reference with no such quote must be "
    "discarded.\n"
    "4. Extract the reference *title* exactly as it appears in the "
    "bibliography — do not paraphrase or fabricate.\n"
    "5. If no references directly match, return an empty array.  Return "
    "zero references rather than padding the output with marginal "
    "matches.\n"
    "6. Output only valid JSON matching the provided schema."
)

USER_TEMPLATE = (
    "## Research Topic\n"
    "{topic_description}\n\n"
    "## Reference List\n"
    "{reference_list}\n\n"
    "## Paper\n"
    "Title: {paper_title}\n\n"
    "{body_text}\n\n"
    "## Task\n"
    "Identify all references from the Reference List above that are "
    "relevant to the research topic.  For each, provide the citation "
    "marker (e.g. \"[23]\" for numbered styles, or \"(Smith et al., 2020)\" "
    "for author-year styles), the full reference entry, the title, "
    "verbatim quote(s) showing where it is cited, and a relevance "
    "explanation.\n\n"
    "## Output Format\n"
    "Return a JSON object with exactly this structure (no markdown "
    "fences, no extra keys):\n"
    "```\n"
    "{{\n"
    '  "relevant_references": [\n'
    "    {{\n"
    '      "citation_marker": "[23]",\n'
    '      "reference_text": "Smith, J. et al. Title of paper. Journal 10, 1-5 (2020).",\n'
    '      "title": "Title of paper",\n'
    '      "mentions": [\n'
    '        {{"quote": "verbatim sentence from paper body", '
    '"relevance": "short tag"}}\n'
    "      ],\n"
    '      "relevance_explanation": "one-sentence reason"\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
    "```\n"
    "If no references are relevant, return "
    "`{{\"relevant_references\": []}}`."
)


def pdf_extraction_schema() -> dict[str, Any]:
    """JSON Schema for the PDF reference extraction response.

    Shape::

        {
          "relevant_references": [
            {
              "citation_marker": "[23]",
              "reference_text": "Smith et al. ...",
              "title": "A Great Paper",
              "mentions": [
                {"quote": "...", "relevance": "method comparison"}
              ],
              "relevance_explanation": "..."
            }
          ]
        }
    """
    return {
        "type": "object",
        "properties": {
            "relevant_references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation_marker": {"type": "string"},
                        "reference_text": {"type": "string"},
                        "title": {"type": "string"},
                        "mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "quote": {"type": "string"},
                                    "relevance": {"type": "string"},
                                },
                                "required": ["quote", "relevance"],
                                "additionalProperties": False,
                            },
                        },
                        "relevance_explanation": {"type": "string"},
                    },
                    "required": [
                        "citation_marker",
                        "reference_text",
                        "title",
                        "mentions",
                        "relevance_explanation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["relevant_references"],
        "additionalProperties": False,
    }
