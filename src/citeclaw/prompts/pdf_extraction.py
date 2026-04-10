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
    "of a research paper and a research topic, identify all references "
    "cited in the paper that are relevant to the specified topic.\n\n"
    "Rules:\n"
    "1. Only include references that are genuinely relevant to the topic.\n"
    "2. For each reference provide verbatim quotes from the paper (1-2 "
    "sentences each) that show *how* the reference relates to the topic.\n"
    "3. Extract the reference *title* exactly as it appears in the "
    "bibliography — do not paraphrase or fabricate.\n"
    "4. If the paper has no references relevant to the topic, return an "
    "empty array.\n"
    "5. Output only valid JSON matching the provided schema."
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
    "marker, the full reference entry, the title, verbatim quote(s) "
    "showing where it is cited, and a relevance explanation.\n"
    "Return JSON with the key \"relevant_references\"."
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
