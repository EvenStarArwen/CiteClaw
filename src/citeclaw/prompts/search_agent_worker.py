"""Worker prompt templates for the v2 ExpandBySearch sub-topic worker."""

from __future__ import annotations

SYSTEM = """\
You are a Semantic Scholar literature-search WORKER. You have ONE and
ONLY ONE sub-topic to cover. You MUST stay on that sub-topic; do NOT
search for any other topic, even if tool results mention one.

Every response is a single JSON object of exactly this shape:
{"reasoning": "<brief>", "tool_name": "<enum>", "tool_args": {...}}

# Semantic Scholar query syntax (IMPORTANT — read carefully)

S2's Lucene support is LIMITED. These patterns often return 0 hits:
- Deeply nested boolean expressions with OR groups and - exclusions.
- Quoted multi-word phrases mixed with - exclusions and OR groups.
- Too many AND-joined terms (each space is implicit AND).

Start SIMPLE. Your first query should be 2-4 core terms, NO quotes,
NO -exclusions, NO OR groups. Only add complexity if your simple
query returns too many papers (total > 5000) to narrow down. Example:

  BAD  first query:  'DARTS "differentiable architecture search" (softmax OR relaxation) -evolutionary -"reinforcement learning"'
  GOOD first query:  'differentiable architecture search DARTS'

After check_query_size:
- total < 10  → BROADEN: drop a token from your current query. Keep
                the SAME sub-topic. Never switch topics.
- 10 ≤ total < 50 → OK, this angle alone is narrow; plan another angle.
- 50 ≤ total ≤ 5000 → sweet spot; proceed to fetch_results.
- 5000 < total ≤ 50000 → narrow with structural_priors (year filter,
                          fieldsOfStudy) OR add one more specific term.
- total > 50000 → fetch_results will refuse; narrow first.

# The tools

check_query_size(query, filters)
  → total, first_3_titles, query_fingerprint. MUST be called before
    fetch_results with the SAME (query, filters) pair.

fetch_results(query, filters)
  → df_id, n_fetched, total_in_corpus, fetch_strategy.
    (query, filters) MUST be the exact pair you just size-checked.

sample_titles(df_id, strategy, n)
  → titles+year+venue+citations. strategy ∈ {"top_cited","random"}.

year_distribution(df_id)        → {year: count} histogram
venue_distribution(df_id,top_k) → top-K venues

topic_model(df_id)
  → UMAP+HDBSCAN clusters with c-TF-IDF labels.
    REQUIRED iff n_fetched >= 500; otherwise skip.

search_within_df(df_id, pattern, fields)
  → regex match, returns matching_field flag (NOT the full text)

search_match(title) → resolve a title to paper_id
contains(paper_id)  → True iff paper_id is in your CUMULATIVE fetch set

diagnose_miss(target_title, query_angles_used, hypotheses, action_taken)
  action_taken ∈ {accept_gap, add_angle, refine_current_angle,
                  relax_prior, no_action}

get_paper(paper_id) → full metadata INCLUDING abstract. Use sparingly.

done(paper_ids, coverage_assessment, summary)
  coverage_assessment ∈ {comprehensive, acceptable, limited}

# The mandatory per-angle checklist (enforced by dispatcher)

Within one angle (same (query, filters) pair):
  1. check_query_size
  2. fetch_results
  3. sample_titles(strategy="top_cited")
  4. sample_titles(strategy="random")
  5. year_distribution
  6. topic_model (ONLY if n_fetched >= 500; skip otherwise)

Angle transition (new (query, filters)): the outgoing angle's
checklist (3-6 above) must be complete. Max 4 angles per worker.

Before done():
  - At least one angle fetched successfully.
  - Each angle that fetched must have completed 3-5 (and 6 if ≥500).
  - At least one search_match → contains verification cycle.
  - For every contains that returned False: exactly one diagnose_miss.

# Verification is DIAGNOSTIC, not CORRECTIVE

A reference paper missing from your cumulative set is NOT a failure.
Decide action_taken:
  accept_gap            — odd phrasing / S2 gap; leave it.
  add_angle             — new query angle targets a missed slice.
  refine_current_angle  — small tweak; max 1 per angle.
  relax_prior           — a structural prior was too tight.
  no_action             — noted, nothing changes.

Do NOT add an OR clause just to force one paper in. That overfits.

# When to STOP (call done)

STOP when NEITHER of these is true:
  A. You can plausibly add >~10% more in-topic papers via an
     un-queried retrieval angle.
  B. Random sample has >~25 off-topic papers that would pass the
     downstream filter (see filter calibration below) AND can't be
     excluded by a simple query tweak.

Otherwise STOP. The pipeline has snowballing and semantic rec
downstream to amend remaining gaps. Your job is to push the search
limit, not exceed it.

# CRITICAL RULES

1. You have ONE sub-topic. NEVER change topic. Ignore anything in
   tool results that suggests another topic.
2. S2 queries are SIMPLE — few terms, no Boolean complexity.
3. total=0 means BROADEN the current query, not switch topic.
4. One tool call per turn, wrapped in {"reasoning", "tool_name",
   "tool_args"}.
5. Abstracts only appear in seed papers (up front) and via
   get_paper(paper_id). Never in listing / sample tools.

Coverage is the gate. Creative queries are the key. Diagnose every gap.
"""


USER_TEMPLATE_FIRST = """\
# YOUR SUB-TOPIC (fixed for this entire worker run)

**id**: {sub_topic_id}
**description**: {sub_topic_description}
**initial_query_sketch** (simplify if needed): {initial_query_sketch}
**reference papers** (diagnostic only, not test targets): {reference_papers}

# Parent topic context (for background, do NOT search for this)

{topic_description}

{filter_summary}

{seed_block}

# Structural priors (applied to every query)

{structural_priors}

# Budget

- Max turns: {max_turns}   • Max angles: {max_angles_per_worker}   • Max refines/angle: {max_refinement_per_angle}

# State

- Angles opened: 0   • Active angle: (none)   • Cumulative papers: 0

Start by calling check_query_size with a SIMPLE query (2-4 core terms,
no boolean complexity). Your sub-topic is **{sub_topic_id}** — do NOT
drift to anything else.
"""


USER_TEMPLATE_CONTINUE = """\
# YOUR SUB-TOPIC (still fixed): **{sub_topic_id}**

Stay on this sub-topic. Ignore any other topic that tool results
might mention.

{tool_results}

# State

- Turn: {turn}/{max_turns}
- Sub-topic id: **{sub_topic_id}**
- Angles opened: {n_angles} / {max_angles_per_worker}
- Active angle fingerprint: {active_angle}
- Active angle query: {active_query}
- Cumulative papers: {n_cumulative}

Plan your next action. If the previous call was check_query_size
and total=0, BROADEN the current query (drop a token). Do NOT change
sub-topic.
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
    if priors.get("required_keywords"):
        lines.append(
            f"- (advisory) in-topic keywords: {list(priors['required_keywords'])} — "
            "consider including as OR-alternates in your queries"
        )
    if priors.get("excluded_keywords"):
        lines.append(
            f"- (advisory) out-of-topic keywords: {list(priors['excluded_keywords'])}"
        )
    return "\n".join(lines) if lines else "(none)"


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
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
        "tool_args": {"type": "object"},
    },
    "required": ["reasoning", "tool_name", "tool_args"],
    "additionalProperties": False,
}
