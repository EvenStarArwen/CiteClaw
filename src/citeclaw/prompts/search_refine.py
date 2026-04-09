"""Prompts for the iterative meta-LLM search agent.

Used by :mod:`citeclaw.agents.iterative_search` (Phase B). The agent
designs targeted literature-database queries from a topic description
and a sample of papers already in the collection, then iteratively
refines its query based on what each search returned.

PH-08 redesign rationale (the v1 → v2 changes):

The v1 prompt failed empirically on 9/10 diverse benchmark topics. Three
failure modes dominated:

  1. **Target-count anchoring** (8/9 runs). v1 embedded a
     ``target_count`` figure in the user prompt, and the model anchored
     on it as a HARD requirement. Even when the agent's own thinking
     said "the corpus is saturated", it kept refining to chase the
     number. v2 drops target_count from the prompt entirely. Quality
     replaces quantity as the only signal.

  2. **Saturation ignored** (2/9 explicit). v1 had a prose rule "mark
     satisfied if NEW < 10 for two turns" but the model treated it as
     guidance, not law. v2 promotes the stop decision to a structured
     ``should_stop: bool`` field on the response schema, and the
     SYSTEM prompt frames the rule as MANDATORY (not "should").

  3. **First-turn over-broadening** (5/9). v1 said "start narrow with
     a quoted phrase" but the model often used a 2-3 word generic
     phrase that hit the page cap. v2 sharpens to "use the SHORTEST
     multi-word phrase that uniquely identifies the topic in this
     subfield".

The v2 design also follows two patterns from the agent-design literature:
  - **Reflexion** (Shinn et al., 2023): explicit per-turn EVALUATE
    step before the action choice. v2 schema places ``evaluate`` first
    so the model articulates what it learned from the previous turn
    before designing the next query.
  - **Anthropic "writing tools for agents"**: keep system prompts
    domain-neutral and let the user message carry domain specifics.
    v2 system prompt has ZERO domain-specific examples — every
    illustrative phrase is abstract (``"phrase A"``, ``"phrase B"``).

The user prompt mentions ``"should_stop"`` as the canonical search-agent
trigger so the offline stub responder in
:mod:`citeclaw.clients.llm.stub` can recognise the shape and return a
deterministic sequence of canned responses for tests. The stub also
keeps the legacy ``"agent_decision"`` field in its response so the
code path that reads it during transition still works.
"""

from __future__ import annotations

from typing import Any

SYSTEM = """\
You design one Semantic Scholar bulk-search query per turn to surface papers on a topic. The user gives you the topic, anchor papers, and prior turns with their signals. You emit one new query OR stop.

SYNTAX (Lucene; operators only between QUOTED phrases)
  ""  phrase   "phrase A"
  +   AND      "phrase A" +"phrase B"
  |   OR       "phrase A" | "phrase B"
  -   NOT      "phrase A" -"phrase B"
  ()  group    ("A" | "B") +"C"

Bare keywords → bag-of-words sorted by paperId (arbitrary). Single-word "phrases" match millions ("RNA" → all RNA-related; "transformer" → also genes). AND/OR/NOT keywords are literal tokens, not operators. + binds tighter than | — parenthesise mixed +/|. Over-constrained intersections (3+ "+" clauses) usually return 0; prefer one "+" of two grouped OR sets plus filters.

FILTERS (narrow without changing text — preferred over more "+" clauses)
  year ("2024" | "2020-2026" | "2022-" | "-2018"), fieldsOfStudy (CS, Biology, Medicine, Physics, Chemistry, Materials Science, Engineering, ...), venue, minCitationCount (int), publicationTypes (Review | JournalArticle | Conference | ...), publicationDateOrYear ("YYYY-MM-DD:YYYY-MM-DD"), openAccessPdf.

SORT — usually omit (default is arbitrary paperId order). Valid values only: paperId, publicationDate, citationCount.

PER-TURN SIGNALS (in transcript)
  n_results = papers returned this turn
  NEW       = papers not seen in earlier turns
  PARTIAL   = corpus has more matches than were returned

MANDATORY RULES (not heuristics)
  A. Previous turn PARTIAL → next query MUST add a filter, NOT more "+" clauses.
  B. Previous turn n_results < 10 → MUST broaden (add | alternatives or drop a "+" clause).
  C. SATURATION STOP: NEW < 5 in BOTH the latest turn AND the turn before → MUST set should_stop=true. There is NO target count; quality beats quantity. 30 high-precision papers > 200 noisy ones.
  D. Otherwise refine direction: new anchor concept, new filter, or "-" exclusion of a crowding near-neighbour.

FIRST TURN
No transcript yet. Use the SHORTEST 2-3 word phrase that uniquely names the topic in this subfield, drawn verbatim from anchor titles. Add one or two obvious field-of-study filters.

OUTPUT
JSON schema fields in order: evaluate, query, should_stop, reasoning. Fill evaluate BEFORE the query — each turn must explicitly reflect on the previous turn's signals.
"""

USER_TEMPLATE = """\
Topic description:
{topic_description}

Anchor papers already in the collection:
{anchor_papers_block}

Iteration {iteration} of {max_iterations}.

Transcript so far (most recent turn last):
{transcript}

Design the next bulk-search query (or stop the loop). Respond with valid JSON whose fields appear in this exact order:
  1. "evaluate"     — 1-2 sentences on what the previous turn's signals (n_results, NEW count, PARTIAL flag) told you. On the first turn, briefly state your initial strategy.
  2. "query"        — object with "text" (required, use the Lucene syntax from the system prompt), "filters" (optional dict), "sort" (optional, usually omit).
  3. "should_stop"  — boolean. Set TRUE per RULE C of the system prompt (saturation: NEW < 5 in last two turns) OR if the topic is genuinely unsearchable. Set FALSE otherwise.
  4. "reasoning"    — one sentence justifying both the query and the should_stop value.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    # Property insertion order is meaningful: capable LLM clients surface
    # fields to the model in declaration order, and the v2 design wants
    # the model to fill ``evaluate`` (Reflexion-style reflection on the
    # previous turn) BEFORE designing the new query. ``should_stop``
    # comes after the query because the stop decision can depend on
    # whether the new query would even change anything.
    "properties": {
        "evaluate": {
            "type": "string",
            "description": (
                "Reflexion-style 1-2 sentence evaluation of the previous "
                "turn's signals (n_results, NEW count, PARTIAL flag). "
                "On the first turn, state the initial strategy in one "
                "sentence. Fill this BEFORE the query so the next "
                "iteration sees the chain of reasoning."
            ),
        },
        "query": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Lucene-style bulk-search query. ALWAYS quote "
                        "multi-word phrases. Operators (+ | -) only "
                        "work between quoted phrases. Parenthesise "
                        "mixed + and |."
                    ),
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional S2 filter dict — keys: year, "
                        "fieldsOfStudy, venue, minCitationCount, "
                        "publicationTypes, publicationDateOrYear, "
                        "openAccessPdf. Use filters to narrow a broad "
                        "topic — they are cheap and selective."
                    ),
                },
                "sort": {
                    "type": "string",
                    "description": (
                        "Usually omit (default = arbitrary paperId "
                        "order). Valid values only: paperId, "
                        "publicationDate, citationCount, each with "
                        "optional :asc / :desc."
                    ),
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "should_stop": {
            "type": "boolean",
            "description": (
                "Mandatory saturation check. Set TRUE if NEW < 5 in "
                "BOTH the latest turn and the one before — the corpus "
                "is exhausted, more refining will not surface new "
                "relevant work. Quality beats quantity; there is no "
                "target count. Setting TRUE ends the loop."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One-sentence justification for both the query and "
                "the should_stop value."
            ),
        },
    },
    "required": ["evaluate", "query", "should_stop", "reasoning"],
    "additionalProperties": False,
}
