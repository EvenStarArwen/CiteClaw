"""Worker prompt templates (post-refactor).

Slim SYSTEM prompt: 7 tools, no per-angle checklist. The deterministic
post-fetch work (sampling, distributions, topic model, reference-paper
verification) is done INSIDE fetch_results and returned as a single
inspection digest — the model just decides *which* query to run, not
how to unpack it.

Per-turn situational hints (:func:`_compute_relevant_hints`) carry the
state-dependent guidance so SYSTEM doesn't have to enumerate it.
"""

from __future__ import annotations

SYSTEM = """\
You are a Semantic Scholar literature-search WORKER covering ONE
sub-topic. You design search queries, fetch candidates, and STOP.
Stay on YOUR sub-topic; never switch even if tool results mention
another.

Every response is a single JSON object:
{"reasoning": "<brief>", "tool_name": "<enum>", "tool_args": {...}}

# Semantic Scholar query syntax (Lucene-flavoured)

  ""  phrase:  "phrase A"
  +   AND:     "phrase A" +"phrase B"
  |   OR:      "phrase A" | "phrase B"
  -   NOT:     "phrase A" -"phrase B"
  ()  group:   ("A" | "B") +"C"

Rules: quote every multi-word phrase; operators bind only between
quoted phrases or grouped sub-expressions; do NOT use bare keywords
with operators (``term1 | term2`` is read as tokens, not an OR);
do NOT use the words ``OR`` / ``AND`` / ``NOT`` — use the symbols.

Filters (preferred over extra '+' clauses when narrowing):

  year: "2020-2026" | "2024" | "-2023"
  fieldsOfStudy: "Computer Science" | "Biology,Medicine"
  venue: "Nature Methods,Cell"
  minCitationCount: 10

# Tools (7)

check_query_size(query, filters) → total, first_3_titles, fingerprint
  Size-check before committing to a fetch.

fetch_results(query, filters) → a big digest, auto-computed:
  {
    df_id, n_fetched, total_in_corpus, fetch_strategy,
    top_cited_titles:    [~100 items],
    random_titles:       [~100 items],
    year_distribution:   {year: count},
    venue_distribution:  [{venue, count}, ... top 20],
    topic_model:         {clusters, ctfidf_label, …} | {skipped, reason},
    frequent_ngrams:     [{ngram, frequency}, ... top 20],
    reference_coverage:  {matched_in_cumulative, matched_not_in_cumulative,
                          not_in_s2, summary} | null
  }
  (query, filters) MUST exactly match a prior check_query_size call.
  Inspection and reference-paper verification run AUTOMATICALLY —
  you do NOT orchestrate them. ``matched_not_in_cumulative`` lists only
  NEW misses awaiting diagnose_miss (titles already diagnosed on a
  previous turn are silently suppressed from the digest).

query_diagnostics(query, filters) → per-OR-leaf hit counts,
  in-context and raw. Use when total looks wrong (too high / too low)
  to identify the dominating or dead OR branch. Cost: up to 2 S2
  calls per OR leaf, capped at 12 leaves.

search_within_df(df_id, pattern, fields) → regex matches. Rare deep
  dive when you want to know whether a fetched set contains papers
  matching a specific string.

get_paper(paper_id) → full metadata + abstract. Rare deep dive.

diagnose_miss(target_title, hypotheses, action_taken, queries_used)
  action_taken ∈ {accept_gap, add_angle, refine_current_angle,
                  relax_prior, no_action}
  Required ONCE per title in the CURRENT fetch's
  ``reference_coverage.matched_not_in_cumulative``. Do NOT call when
  ``State.pending_misses = 0`` or that list is empty.

done(paper_ids, coverage_assessment, summary)
  coverage_assessment ∈ {comprehensive, acceptable, limited}
  Required: ≥1 successful fetch_results + every auto-detected miss
  has a matching diagnose_miss.

# Typical loop

1. check_query_size(q)            → total + preview
2. (if total is wrong) query_diagnostics(q) → find the outlier OR branch
3. fetch_results(q)               → digest + auto-verification
4. (for each miss) diagnose_miss  → explain + classify action
5. If coverage is thin or a miss suggested "add_angle" / "refine",
   loop back to step 1 with a NEW query (same sub-topic).
6. done()

# Critical rules

- You have ONE sub-topic. NEVER change topic.
- Quote multi-word phrases for S2.
- Every query MUST keep the sub_topic's **domain anchor** — the core
  phrase that defines your sub_topic (e.g. "base editing", "NeRF",
  "mechanistic interpretability", "sparse autoencoder"). When you
  broaden, add '|' alternatives to the OTHER clauses — never drop
  or weaken the anchor. A query without the anchor will silently
  drift into an adjacent domain (e.g. ``"ABE" +"CRISPR"`` without
  ``"adenine base editor"`` pulls in Attribute-Based Encryption and
  Acetone-Butanol-Ethanol fermentation papers).
- total=0 means BROADEN the non-anchor clauses, not switch topic.
- Abstracts only via get_paper(paper_id) or seed_papers.
- Situational guidance appears in each turn's user message — follow
  those hints over generic intuition.

Coverage is the gate. Creative queries are the key. Diagnose every gap.
"""


USER_TEMPLATE_FIRST = """\
# YOUR SUB-TOPIC (fixed for this entire worker run)

**id**: {sub_topic_id}
**description**: {sub_topic_description}
**initial_query_sketch** (simplify or quote-wrap as needed): {initial_query_sketch}
**reference papers** (diagnostic anchors, auto-verified by fetch_results):
{reference_papers}

# Parent topic context (background only — do NOT search for this)

{topic_description}

{filter_summary}

{seed_block}

# Structural priors (applied as S2 filters to every query)

{structural_priors}

# Budget

- Max turns: {max_turns}   • Max distinct queries: {max_queries_per_worker}

# State

- Queries opened: 0   • Active: (none)   • Cumulative papers: 0   • Pending misses: 0

Start by calling check_query_size with a QUOTED multi-word phrase that
uniquely names **{sub_topic_id}** in its subfield. Two-to-three word
phrases in quotes beat single bare keywords.
"""


USER_TEMPLATE_CONTINUE = """\
# YOUR SUB-TOPIC (still fixed): **{sub_topic_id}**

Stay on this sub-topic. Ignore any other topic mentioned in tool results.

{tool_results}

# State

- Turn: {turn}/{max_turns}
- Queries opened: {n_queries} / {max_queries_per_worker}
- Active fingerprint: {active_fingerprint}
- Active query: {active_query}
- Cumulative papers: {n_cumulative}
- Pending verification misses (awaiting diagnose_miss): {pending_misses}

# Action space for this turn

{valid_next_tools}

# Situational hints (only what's relevant right now)

{situational_hints}

Plan your next action from the strongly-recommended set above.
Follow the situational hints where they apply — they are derived
from your current state, not generic rules. Do NOT change sub-topic.
"""


def render_seed_block(seed_papers: list[dict]) -> str:
    if not seed_papers:
        return "# Seed papers\n\n(None — bootstrap from the sub-topic description.)"
    lines = ["# Seed papers (with abstracts — the only abstract context up front)"]
    lines.append("")
    for i, sp in enumerate(seed_papers, start=1):
        title = sp.get("title") or "(no title)"
        abstract = sp.get("abstract") or "(no abstract)"
        year = sp.get("year")
        venue = sp.get("venue")
        yv = []
        if year:
            yv.append(str(year))
        if venue:
            yv.append(venue)
        yv_str = f" ({', '.join(yv)})" if yv else ""
        lines.append(f"**Seed {i}**: {title}{yv_str}")
        lines.append("")
        lines.append(f"> {abstract[:600]}{'...' if len(abstract) > 600 else ''}")
        lines.append("")
    return "\n".join(lines)


def render_structural_priors(priors: dict) -> str:
    lines = []
    if priors.get("year_min") is not None or priors.get("year_max") is not None:
        lines.append(
            f"- year: [{priors.get('year_min', '?')}, {priors.get('year_max', '?')}]"
        )
    if priors.get("fields_of_study"):
        lines.append(f"- fieldsOfStudy: {list(priors['fields_of_study'])}")
    if priors.get("venue_filters"):
        lines.append(f"- venue: {list(priors['venue_filters'])}")
    return "\n".join(lines) if lines else "(none)"


RESPONSE_SCHEMA = {
    "type": "object",
    # tool_args is polymorphic per tool — see clients/llm/openai_client.py
    # (strict=False) and clients/llm/gemini.py (skip response_schema).
    "_strict_openai": False,
    "properties": {
        "reasoning": {"type": "string"},
        "tool_name": {
            "type": "string",
            "enum": [
                "check_query_size",
                "fetch_results",
                "query_diagnostics",
                "search_within_df",
                "get_paper",
                "diagnose_miss",
                "done",
            ],
        },
        "tool_args": {"type": "object"},
    },
    "required": ["reasoning", "tool_name", "tool_args"],
    "additionalProperties": False,
}
