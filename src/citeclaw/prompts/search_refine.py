"""Prompts for the iterative meta-LLM search agent.

Used by :mod:`citeclaw.agents.iterative_search` (Phase B). The agent
designs targeted literature-database queries from a topic description
and a sample of papers already in the collection, then iteratively
refines its query based on what each search returned.

To force a per-call chain of thought *before* any structured decision,
the response schema places a free-text ``thinking`` field FIRST. Capable
models fill that field with their scratchpad before committing to a
query, and each turn's thinking is forwarded into the next iteration's
user prompt so the agent can see (and build on) its earlier reasoning.
This is the "Level-1" inner reasoning loop; native reasoning tokens via
``reasoning_effort="high"`` stack on top.

The user prompt mentions the literal token ``"agent_decision"`` (with
surrounding double quotes, exactly as it would appear in a JSON
document) so the offline stub responder in
:mod:`citeclaw.clients.llm.stub` can recognise the shape and return a
deterministic sequence of canned responses for tests.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You design targeted literature-database queries given a topic and "
    "a sample of papers already in the collection.\n"
    "Before committing to a query, think out loud in the `thinking` "
    "field — note what you already know from the anchor papers, what "
    "is still missing, and what query shape is most likely to surface "
    "the missing work.\n"
    "After each search, inspect the results: if the query was too broad "
    "or too narrow, refine it; if the results saturate the topic, mark "
    "the search satisfied; if the topic is unsearchable from this anchor "
    "set, abort.\n"
    "Output only valid JSON matching the supplied response schema."
)

USER_TEMPLATE = (
    "Topic description:\n"
    "{topic_description}\n\n"
    "Anchor papers already in the collection:\n"
    "{anchor_papers_block}\n\n"
    "Iteration {iteration} of {max_iterations}. Target collection size: "
    "{target_count} papers.\n\n"
    "Transcript so far (most recent turn last):\n"
    "{transcript}\n\n"
    "Design the next literature-database query. Respond with valid JSON "
    "whose fields appear in this exact order:\n"
    '  1. "thinking": free-text scratchpad — fill this BEFORE deciding '
    "the query, so later turns can see the chain of reasoning.\n"
    '  2. "query" — an object with "text" (required), optional "filters", '
    'optional "sort".\n'
    '  3. "agent_decision": one of "initial" / "refine" / "satisfied" '
    '/ "abort".\n'
    '  4. "reasoning": one short sentence justifying the agent_decision.\n'
    "\n"
    "Use `initial` on the first turn, `refine` to narrow or broaden a "
    "previous query, `satisfied` when you have found enough relevant "
    "work, and `abort` if the topic is unsearchable from this anchor set."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    # Property insertion order is meaningful here: capable LLM clients
    # surface fields to the model in the order declared, and the agent's
    # value comes from filling `thinking` BEFORE any structured field.
    "properties": {
        "thinking": {
            "type": "string",
            "description": (
                "Free-text scratchpad. Fill this BEFORE deciding the "
                "query so later iterations can see the chain of "
                "reasoning."
            ),
        },
        "query": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Free-text database query string.",
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional S2 search/bulk filter dict — keys "
                        "like year / venue / fieldsOfStudy / "
                        "minCitationCount / publicationTypes."
                    ),
                },
                "sort": {
                    "type": "string",
                    "description": (
                        "Optional sort key, e.g. citationCount:desc."
                    ),
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "agent_decision": {
            "type": "string",
            "enum": ["initial", "refine", "satisfied", "abort"],
            "description": (
                "Lifecycle state for this turn. `initial` on the first "
                "iteration, `refine` while still iterating, `satisfied` "
                "to break the loop with success, `abort` to break with "
                "failure."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One-sentence justification for the agent_decision "
                "value above."
            ),
        },
    },
    "required": ["thinking", "query", "agent_decision", "reasoning"],
    "additionalProperties": False,
}
