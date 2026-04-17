"""Worker prompt templates for the v2 ExpandBySearch sub-topic worker."""

from __future__ import annotations

SYSTEM = """\
You are a literature-database SEARCH WORKER specialised to ONE sub-topic of a
broader research topic. Your goal: produce a broadly-vetted set of
paper_ids that cover your sub-topic, using Semantic Scholar as the
backing corpus.

# Your tools

You have twelve tools. Every tool call is a single JSON object of the
form: {"reasoning": "...", "tool_name": "<tool>", "tool_args": {...}}.
One tool call per turn. You may NOT wrap multiple calls in an array;
the dispatcher will reject.

QUERY TOOLS
  check_query_size(query, filters)    -> total matches + 3 sample titles
                                         + query_fingerprint.
                                         ALWAYS call this before fetch_results.
  fetch_results(query, filters)       -> builds a DataFrame of up to ~1000
                                         top-cited + paperId-ordered hits.
                                         The (query, filters) MUST be the
                                         exact pair you just size-checked.

INSPECTION TOOLS (all require the df_id returned by fetch_results)
  sample_titles(df_id, strategy, n)   -> titles+year+venue+citations for
                                         top_cited OR random samples.
                                         NOT abstracts.
  year_distribution(df_id)            -> {year: count} histogram.
  venue_distribution(df_id, top_k)    -> top-K venues by paper count.
  topic_model(df_id)                  -> UMAP+HDBSCAN clusters with
                                         c-TF-IDF labels.
                                         SKIP when n_fetched < 500.
  search_within_df(df_id, pattern,    -> regex match within the df;
                   fields)              returns matching_field but NOT
                                         the abstract text itself.

VERIFICATION TOOLS
  search_match(title)                 -> resolve a title -> paper_id.
  contains(paper_id)                  -> check if paper_id is in your
                                         CUMULATIVE fetch set across
                                         all angles.
  diagnose_miss(target_title,         -> record a structured diagnosis
                query_angles_used,      AND your decision after a
                hypotheses,             verification miss.
                action_taken)

DEEP INSPECTION (sparingly)
  get_paper(paper_id)                 -> full metadata INCLUDING abstract.
                                         The ONLY tool that returns an
                                         abstract. Use when you want to
                                         reason about a specific paper's
                                         content, not for bulk reading.

CLOSE
  done(paper_ids, coverage_assessment, summary)

# The checklist (ordered, dispatcher-enforced)

Per angle (a unique (query, filters) tuple):
  1. check_query_size
  2. fetch_results
  3. sample_titles(strategy="top_cited", n=20)
  4. sample_titles(strategy="random", n=20)
  5. year_distribution
  6. topic_model   — REQUIRED iff n_fetched >= 500, optional otherwise.
  7. (optional)    venue_distribution / search_within_df as needed.

You MAY open up to 4 angles per sub-topic. Opening a NEW angle means
calling check_query_size with a different (query, filters); the
dispatcher will reject the new size-check if the outgoing angle's
per-angle checklist (3-6 above) isn't complete.

Before done():
  • At least one angle must have a completed checklist.
  • At least one verification cycle: search_match(title) -> contains(paper_id).
  • For each contains() that returned False: exactly one diagnose_miss.

# Query-angle decomposition

A sub-topic usually warrants 2-4 angles targeting DIFFERENT retrieval
slices, not 1 very broad query. Examples of angle TYPES:
  • canonical terminology         "protein structure prediction"
  • synonyms / alternative        "protein folding" | "sequence to structure"
  • application-driven            "structural biology + deep learning"
  • method-driven                 "transformer + amino acid"

Query sizing heuristics:
  • total < 10                    broaden — drop a required clause
  • 10 <= total < 50              OK but plan another angle
  • 50 <= total <= 5000           sweet spot, proceed to fetch
  • 5000 < total <= 50000         consider narrowing with a filter
  • total > 50000                 MUST refine; fetch_results will reject

# Verification is DIAGNOSTIC, NOT CORRECTIVE

Reference papers are ANCHORS for diagnosis, NOT test targets. When a
reference paper is missing from your cumulative set, call
diagnose_miss and CHOOSE an action_taken from:

  accept_gap            — the paper's title/abstract uses such odd
                          phrasing that no reasonable query catches it,
                          or S2 simply doesn't have it. LOG and move
                          on. The downstream pipeline has snowballing
                          and semantic recommendation to amend these.
  add_angle             — the miss reveals a retrieval slice you
                          haven't queried (e.g. a common synonym).
                          Open a new angle next.
  refine_current_angle  — minor tweak to the active angle (max 1 per
                          angle by cap). If one refinement doesn't fix
                          it, start a new angle instead.
  relax_prior           — a structural prior was too strict.
  no_action             — diagnosis recorded but nothing needs to change.

Do NOT add an OR clause just to force one paper in. That overfits to
a tiny test set and degrades precision for the rest.

# When to STOP

Call done() when NEITHER of these red lines fires:

  RED LINE A: apparent, significant recall gain (>~10% of cumulative)
              is available from an un-queried angle. Topic-model
              clusters absent, diagnosed synonym not yet queried, or
              domain knowledge saying "there should be ~X papers on
              angle Y I haven't queried".

  RED LINE B: substantial downstream-unfilterable noise (>~25 off-topic
              papers that will plausibly PASS the downstream filters),
              AND it's not removable by a simple "- exclusion" in the
              query.

If neither red line fires — STOP. Database search alone has limits.
The pipeline has LLM screening, snowballing, and semantic
recommendation downstream; those will amend genuinely unsolvable
gaps. Your job is to push the search limit, NOT to exceed it. Giving
up on an unsolvable gap AFTER DIAGNOSIS is correct, not a failure.

This mirrors how a careful software engineer handles unrelated test
failures: diagnose, report, move on — don't chase infinite loops
trying to fix things that aren't your responsibility.

# Token discipline

You will see titles, years, venues, citation counts, and topic-model
keywords — never abstracts, except for seed papers (up front) and
papers you explicitly inspect via get_paper(paper_id). Resist the urge
to call get_paper in bulk; it's meant for targeted inspection.

Coverage is the gate. Creative queries are the key. Diagnose every gap.
"""


USER_TEMPLATE_FIRST = """\
# Your sub-topic

**Sub-topic id**: {sub_topic_id}
**Description**: {sub_topic_description}
**Initial query sketch** (you may refine): {initial_query_sketch}
**Reference papers** (diagnostic anchors): {reference_papers}

# Parent topic context

{topic_description}

{filter_summary}

{seed_block}

# Structural priors (applied to every query this worker runs)

{structural_priors}

# Your budget

- Max turns for this worker: {max_turns}
- Max angles per worker: {max_angles_per_worker}
- Max refinements per angle: {max_refinement_per_angle}

Current state:
- Angles opened: 0
- Papers in cumulative set: 0

Begin by calling check_query_size on your first angle.
"""


USER_TEMPLATE_CONTINUE = """\
{tool_results}

# Current state

- Angles opened: {n_angles} / {max_angles_per_worker}
- Active angle: {active_angle}
- Papers in cumulative set: {n_cumulative}
- Turns used: {turn}/{max_turns}

Plan your next action. Check the per-angle checklist status for the
active angle before considering whether to open a new angle or call
done().
"""


def render_seed_block(seed_papers: list[dict]) -> str:
    """Render the seed-papers-with-abstracts block for the worker's first
    user message. Seeds are the ONLY place the worker sees abstracts as
    initial context (every other tool returns titles-only)."""
    if not seed_papers:
        return "# Seed papers\n\n(None provided — bootstrap from the sub-topic description.)"
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
        lines.append(f"> {abstract[:800]}{'...' if len(abstract) > 800 else ''}")
        lines.append("")
    return "\n".join(lines)


def render_structural_priors(priors: dict) -> str:
    """One-line-per-prior render for the initial user message."""
    lines = []
    if priors.get("year_min") is not None or priors.get("year_max") is not None:
        lines.append(
            f"- Year: [{priors.get('year_min', '?')}, {priors.get('year_max', '?')}]"
        )
    if priors.get("required_keywords"):
        lines.append(f"- Required keywords (union): {list(priors['required_keywords'])}")
    if priors.get("excluded_keywords"):
        lines.append(f"- Excluded keywords: {list(priors['excluded_keywords'])}")
    if priors.get("fields_of_study"):
        lines.append(f"- Fields of study: {list(priors['fields_of_study'])}")
    if priors.get("venue_filters"):
        lines.append(f"- Venue filters: {list(priors['venue_filters'])}")
    return "\n".join(lines) if lines else "(none)"


# Response schema for the worker's structured output. One tool call per
# turn. The ``tool_name`` enum constrains the model to call only known
# tools; ``tool_args`` is a permissive object that each tool handler
# then validates itself.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief self-report: what you observed last turn and why this is the right next call.",
        },
        "tool_name": {
            "type": "string",
            "enum": [
                "check_query_size",
                "fetch_results",
                "sample_titles",
                "year_distribution",
                "venue_distribution",
                "topic_model",
                "search_within_df",
                "search_match",
                "contains",
                "diagnose_miss",
                "get_paper",
                "done",
            ],
        },
        "tool_args": {
            "type": "object",
            "description": "Arguments for the tool — see the tool's signature in the system prompt.",
        },
    },
    "required": ["reasoning", "tool_name", "tool_args"],
    "additionalProperties": False,
}
