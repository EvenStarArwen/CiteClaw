"""Prompt templates for V3 ExpandBySearch.

V4.1 revision:

- Supervisor goes back to plan-only. It dispatches sub-topics with
  just ``{id, description}`` — no facet skeleton. The anchor-shared
  decomposition judgement is still the supervisor's job, but the
  output is a count, not a skeleton.
- Anchor-discovery agent runs between supervisor and worker, ungated
  by any skeleton — it asks the description alone for a precise
  query, confirms 5-15 canonical papers.
- Worker's first act is to READ the anchor papers and design its own
  facets from real author vocabulary, then emit an initial
  :class:`QueryPlan`. No amendment turn — the worker owns the facet
  set and can ``add_facet`` / ``remove_facet`` in later iterations.
- Refinement iterations still pick up to two transformations from a
  closed set (now eight ops, adding ``remove_facet``); the plan is
  mutated in place, never retyped.
- Anchor coverage is the refinement signal; total count + the query
  tree are informational context inside ``diagnose_plan``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Worker prompts
# ---------------------------------------------------------------------------


WORKER_SYSTEM = """\
You are designing Semantic Scholar search queries for ONE assigned
sub-topic of a broader literature survey. You work one step at a time —
each turn I show you one piece of information and ask one focused
question. You answer. Over multiple turns we refine your query together.

# Your sub-topic

{description}

# How refinement works

Each iteration you get to apply up to two transformations to the
current query: typically one to fill coverage gaps (add missing
vocabulary, loosen an over-tight match), and one to kill noise
(prune a leaky term, tighten a match). The balance between the two
depends on what the previous iteration's clusters and anchor
coverage revealed — there is no fixed direction.

The primary signals you react to:
- **Topical cleanliness** — are the clusters and the top-cited
  titles all on-topic for the sub-topic?
- **Anchor coverage** — do the canonical papers you (and the
  anchor-discovery step) flagged actually appear in the fetched set?

Total count and the query tree are context, not signals on their own.

# Semantic Scholar query syntax

Use natural-language Boolean operators:
- `AND` (must contain)
- `OR`
- `NOT` (must not contain)
- `"..."` exact phrase
- `(...)` grouping

Syntax rules:
- Always write `AND` / `OR` / `NOT` as WORDS (the translator will
  convert them to the symbolic operators S2 requires).
- Wrap every OR group in parentheses: `("A" OR "B") AND "C"`.

# Term-strictness guidance (used by tighten/loosen transformations)

For multi-word terms, three matching modes from loosest to strictest:
- `A AND B` — both words anywhere (AND).
- `"A B"~N` — both words within N positions (proximity).
- `"A B"` — exact contiguous phrase.

Exact-phrase matching is brittle: it misses documents that use the
same concept with different word order or intervening modifiers.

Two hard rules on term choice:
- Tokens shorter than 4 characters — especially pure-letter
  acronyms like `PE`, `RT`, `LM` — collide with so many unrelated
  fields (preeclampsia, reverse transcriptase, linear model) that
  no AND-gate cleans them up. Omit such bare acronyms, or pair
  them with their full expansion in the same OR group so the
  expansion carries the recall.
- If one phrase is a substring of another, drop the longer one.
  Every document matching `"A B C"` also matches `"A B"`, so
  `("A B" OR "A B C")` collapses to `"A B"` alone.

S2's default matching folds plurals and common inflections together
(`edit` catches `edits` / `editing`; `protein` catches `proteins`;
`pegRNA` catches `pegRNAs`), so you don't need to list both.

When you pass a term into any transformation or propose-first shape,
supply it WITHOUT surrounding double-quotes. Strictness is a
separate field — the `raw` field is just the text. Writing
`"prime editing"` as the raw would double-wrap it on render.

Output format: each turn, respond with a single JSON object shaped
to the current question — I'll tell you the shape.
"""


WORKER_PROPOSE_FIRST = """\
Here's everything you have to work from for the initial query:

# Sub-topic

{description}

# Anchor papers (real canonical work for this sub-topic)

{anchors_block}

# Task

Design the facets for this sub-topic by reading the anchor papers'
titles and abstracts. A facet is one AND'd concept whose
co-occurrence with the other facets is a near-guarantee of on-topic.
Typical counts: 2-4 facets. Fewer AND'd facets = higher recall;
more = higher precision. Err toward the minimum set that captures
the sub-topic.

For each facet, write an OR group of the terms a domain expert
would use. Mine the anchor abstracts for real vocabulary rather
than listing the theoretically neat labels; authors' wording is
what S2 indexed.

For each term, pick a strictness:
- `and_words` — multi-word term matched as AND'd words (loosest).
- `proximity` — `"A B"~N` within N positions.
- `phrase` — exact contiguous quoted phrase (strictest; default).

Single-word terms are always rendered bare regardless.

Respond with JSON:
```
{{"facets": [
    {{"id": "technology", "concept": "prime editing technique",
      "terms": [
        "prime editing",
        {{"raw": "pegRNA", "strictness": "phrase"}},
        {{"raw": "twin prime editing", "strictness": "proximity", "slop": 2}}
      ]}},
    ...
  ],
  "exclusions": [],
  "reasoning": "one short sentence on why these facets capture the sub-topic"
}}
```

`id` must be a unique slug — later transformations reference it.
Exclusions are rare — leave empty unless a concrete noise source
is already obvious from the anchors.
"""


WORKER_SELECT_TRANSFORMATIONS = """\
# Current query plan

{plan_tree}

Rendered query: `{query}`

# What you've seen this iteration

- Total count: {total} (new +{diff_new}, seen {diff_seen})
- Anchor coverage:
{anchor_coverage}
- Top-cluster cleanliness (one-sentence recap): {ans_clusters}
- Top-100 cleanliness (one-sentence recap): {ans_top100}
- Per-noise-item diagnoses:
{noise_diagnoses}

# Prior transformations this worker has already applied

{transformation_history}

# Diagnose + plan

diagnosis:       {diagnosis}
intended change: {intended_change}

# Task

Pick up to 2 transformations to apply. The system applies them
mechanically — you never retype the query. Available ops:

- `add_or_alternative(facet_id, terms)` — pearl-grow within a
  facet. `terms` is a list of strings or `{{"raw": ..., "strictness": ...}}`.
- `remove_or_alternative(facet_id, term)` — drop a leaky OR
  alternative the in-cluster tree showed pulling noise.
- `tighten_term(term, to)` — `to` is `proximity_N` or
  `exact_phrase`. Use when that term's loose form is the noise
  source.
- `loosen_term(term, to)` — `to` is `and_words` or `proximity_N`.
  Use when an anchor is absent and an exact-phrase term is the
  obvious reason.
- `swap_operator(to, ...)` — rare. `to: "OR"` merges facet_id + b
  into one OR group. `to: "AND"` splits a facet_id on a
  `split_on` term list into two AND'd facets.
- `add_facet(facet_id, concept, terms)` — add a whole new AND
  dimension. Use when a real dimension is missing.
- `remove_facet(facet_id)` — drop an AND dimension that isn't
  topic-defining. Use when diagnosis shows one facet is subsumed
  by another or was a bad call from iter 0.
- `add_exclusion(term)` — NOT clause. Last resort; use only when
  the noise source shares one unambiguous characteristic term
  with no collateral damage.

Typical iteration: one widening op + one narrowing op. If coverage
and cleanliness already look acceptable, `transformations` can be
empty and `satisfied: true` closes the worker.

Respond with JSON:
```
{{"transformations": [
    {{"type": "add_or_alternative", "facet_id": "technology", "terms": ["twinPE"]}},
    {{"type": "remove_or_alternative", "facet_id": "application", "term": "neoplasm"}}
  ],
  "reasoning": "one short sentence",
  "satisfied": false
}}
```
"""


WORKER_NOISE_TOPIC_DIAG = """\
You flagged cluster {cluster_id} as unrelated to the sub-topic
(`{description}`). I've pulled the cluster's in-cluster breakdown for
you — it shows exactly which OR alternatives across your facets are
responsible for these papers landing here.

{inspect_output}

# Question (DIAGNOSIS ONLY — do NOT propose a fix)

In ONE short sentence, name the specific mechanism in your current
query that is letting this cluster in. Identify the culprit OR
alternative(s) or term(s). Don't describe a fix — just the mechanism.

Respond with JSON: `{{"reason": "..."}}`.
"""


WORKER_NOISE_PAPER_DIAG = """\
You flagged the following paper as off-topic:

  title: {title}

Here is what S2 knows about it (pulled to save you a tool call):

{inspect_output}

# Question (DIAGNOSIS ONLY — do NOT propose a fix)

In ONE short sentence, name the specific part of your current query
that let this paper through. Don't describe a fix.

Respond with JSON: `{{"reason": "..."}}`.
"""


WORKER_SYNTAX_ERROR = """\
Your proposed query has Semantic Scholar syntax problems that would
cause the API to reject it (or silently mis-interpret it). Fix them
and resubmit.

Your query:
  {query}

Issues found:

{issues}

Rewrite the query to fix every ERROR. Warnings (lower-case `warn`)
are advisory — address them if the advice matches your intent.
Respond with JSON: `{{"query": "..."}}`.
"""


WORKER_CHECK_CLUSTERS = """\
Your current query: `{query}`

# Topic clusters (UMAP + HDBSCAN on SPECTER2, c-TF-IDF keywords)

{clusters}

# Question

Look at each cluster. Are any of them **obviously unrelated** to the
sub-topic (`{description}`)? Also: are there major sub-areas of the
sub-topic that you'd expect to see but that DO NOT appear as a cluster?

Respond with JSON:
`{{"unrelated_clusters": [cluster_ids], "missing_areas": ["..."], "reasoning": "..."}}`.
Keep reasoning to ONE short sentence.
"""


WORKER_CHECK_TOP100 = """\
Your current query: `{query}`

# Top-100 papers by citation count (titles only)

{titles}

# Question

Scan for red flags: any paper clearly **off-topic** that shouldn't be
here? You do NOT need to list every one. Just flag what's notable.

Respond with JSON:
`{{"off_topic_titles": ["..."], "reasoning": "..."}}`. Keep reasoning
to ONE short sentence.
"""


WORKER_DIAGNOSE_PLAN = """\
Your current query: `{query}`

# What you answered this iteration

- check_clusters: {ans_clusters}
- check_top100:   {ans_top100}

# Anchor coverage this iteration

{anchor_coverage}

# Per-noise-item diagnoses (from forced Phase 4b / 5b questions)

{noise_diagnoses}

# Context (informational — not a signal on its own)

- Total papers matching on S2: **{total}**
- New vs prior queries this iteration: **+{diff_new} new**, **{diff_seen} seen before**

Query tree (per-clause counts):
{query_tree}

# Raw analysis (full data — use this to synthesise)

## top-10 cited papers (title + abstract)
{top10_block}

## 10 random papers (title + abstract)
{random10_block}

## topic clusters (c-TF-IDF keywords + representative titles)
{clusters}

# Synthesise — ONE plan

Each refinement round targets two success conditions:
(a) the clusters and the top-cited slice are topical, and
(b) the auto-injected anchors are ``present`` in the fetched set.
When both hold the query is good enough; further iteration risks
over-optimising.

Refinement modes you can combine (one of each is a common pattern):
- **Pearl growing** — add an OR alternative with vocabulary you
  saw in the top-cited / random titles or in the anchor papers but
  didn't anticipate.
- **Term tightening** — promote a term from AND-words to proximity
  or to exact phrase when that term's looseness is the noise source.
- **Term loosening** — demote in the reverse direction when an
  anchor is absent and an over-strict match is the obvious reason.
- **AND-gate a leaky acronym** — replace a bare `PE`-type
  alternative with a facet pair like `("prime editing" OR ("PE"
  AND pegRNA))`.
- **Pruning** — drop OR alternatives the in-cluster tree shows add
  no unique relevant papers.
- **Facet add / remove** — add a new AND dimension when a real
  one is missing, or remove one whose presence is clearly not
  topic-defining.
- **Exclusion (NOT)** — last resort; only when a noise cluster
  shares one unambiguous characteristic term with no collateral
  damage.

You now have per-noise-item diagnoses that name the specific
mechanisms letting bad papers in. Combine those with the overall
picture and commit to ONE plan. Multiple noise sources may point
to conflicting fixes (tighten here, loosen there) — pick the single
coherent plan that wins the trade-off, don't list both.

Respond with JSON:
`{{"diagnosis": "...", "intended_change": "...", "coverage_ok": true|false}}`.

Keep `diagnosis` and `intended_change` SHORT — one sentence each.
`coverage_ok` is true when anchor coverage is strong enough AND
cleanliness looks acceptable that another refinement round would
be over-optimising.
"""


WORKER_FINAL_SUMMARY = """\
Budget exhausted. Across all iterations you fetched a deduped set of
{n_unique} papers.

Top clusters in the final union:
{clusters}

Anchor coverage on the final query:
{anchor_coverage}

In ONE sentence, describe the coverage this sub-topic ended with.

Respond with JSON: `{{"summary": "..."}}`.
"""


# ---------------------------------------------------------------------------
# Anchor discovery (pre-worker stage)
# ---------------------------------------------------------------------------


ANCHOR_DISCOVERY_SYSTEM = """\
You are finding anchor papers for a sub-topic: the handful of
canonical works every worker must retrieve to claim coverage. Your
output grounds the downstream worker's synonym expansion in real
domain vocabulary.

The job is two turns:
1. Write a PRECISE query using only terms you are 100% sure are
   correct for this sub-topic. No speculative synonyms. The query
   must be specific enough that its top-cited matches are obvious
   anchors, not a broad recall sweep.
2. I run your query on S2, rank by citation count, and show you the
   top 15 candidates with title + abstract. You mark each
   `on_topic` / `off_topic` / `uncertain` and I keep only the
   on-topic ones.

Same natural-language Boolean syntax as the worker: AND / OR / NOT
as words, wrap OR groups in parentheses, `"..."` for exact phrase.

Respond with JSON per the shape I request each turn.
"""


ANCHOR_DISCOVERY_QUERY = """\
# Sub-topic

{description}

# Task

Write a precise query. Include only terms you are 100% sure authors
use for these exact concepts. It is fine (expected) for this query
to be narrow — we're looking for the canonical papers, not full
coverage.

Respond with JSON: `{{"query": "...", "reasoning": "one short sentence"}}`.
"""


ANCHOR_DISCOVERY_CONFIRM = """\
# Sub-topic

{description}

# Top candidates (by citation count)

{candidates}

# Task

For each candidate, decide whether it is clearly ON-TOPIC for the
sub-topic. Read the abstract, not just the title. Reject papers
whose topic overlaps only in surface vocabulary (e.g. "PE" as
preeclampsia when the sub-topic is prime editing). Keep the bar
high — one off-topic paper in the anchor set will mislead the
worker's vocabulary mining.

Respond with JSON:
`{{"decisions": [{{"index": 1, "verdict": "on_topic"|"off_topic"|"uncertain", "reason": "..."}}, ...]}}`.
Use the `[N]` indices from the candidate list. Keep each `reason`
to a short phrase.
"""


# ---------------------------------------------------------------------------
# Supervisor — plan-only, anchor-shared decomposition
# ---------------------------------------------------------------------------


SUPERVISOR_SYSTEM = """\
You are a literature-search SUPERVISOR. You analyse the parent topic
and decide whether it needs sub-topic decomposition. The system
dispatches workers for you — you never call dispatch yourself.

# Decomposition decision

Each worker designs its own facets after reading anchor papers; your
job is to decide how many workers the parent topic needs.

Before picking a count, sketch in your head the facet set (the AND'd
concepts whose co-occurrence guarantees on-topic) of each candidate
sub-topic. Then apply this test:

- If every candidate would use the SAME facets — differing only in
  synonym variants within each facet — ONE sub-topic suffices. The
  worker's single query captures all sub-areas.
- If candidates need STRUCTURALLY different facets (at least one
  facet that differs in kind, not just in vocabulary), decompose.

Abstract illustration:
- Shared facet set: all sub-areas captured by `(X) AND (Y)` → one
  sub-topic.
- Distinct facet sets: sub-area 1 `(X) AND (P)`, sub-area 2 `(X)
  AND (Q)`, sub-area 3 `(X) AND (R)`, where P, Q, R are genuinely
  different concepts → three sub-topics.

Default toward fewer. Respect the parent topic's explicit scope —
any `NOT X` phrase in the description excludes X as a sub-topic axis.

# Signals between turns

After each worker returns, you see a pairwise paper-ID overlap
matrix. High overlap (>50%) means the decomposition was redundant —
treat it as a strong signal against adding more sub-topics.

# Turn flow

1. Turn 1: `set_strategy` with your initial decomposition (1 to
   {max_subtopics} sub-topics).
2. System runs anchor-discovery and dispatches sub-topics one at a
   time in queue order. After each worker returns, you get a turn:
   - `add_sub_topics` — add a new sub-topic if the returned clusters
     reveal a genuinely new retrieval axis not covered by any
     existing sub-topic AND still in the parent's scope.
   - `continue` — no change, let the system dispatch the next queued
     sub-topic.
   - `done(summary)` — close early.
3. When the queue empties and you have nothing to add, call
   `done(summary)`.

# Tools

Every reply is `{{"reasoning": "...", "tool_name": "<tool>", "tool_args": {{...}}}}`.
Keep `reasoning` to one short sentence.

- `set_strategy(sub_topics: [{{id, description}}])` — call once at
  turn 1. `description` is 1-2 English sentences naming the
  sub-area. No query syntax; no facets.
- `add_sub_topics(sub_topics: [{{id, description}}])` — add only
  when the new sub-area needs a structurally different facet set
  from every existing sub-topic AND is in the parent's scope.
  Never to cover a narrow gap within an existing sub-topic's
  coverage — that's the worker's job.
- `continue` — no arguments. Proceed to the next queued worker.
- `done(summary)` — close the run.
"""


SUPERVISOR_USER_FIRST = """\
# Parent topic

{topic_description}

# Your budget

- Max sub-topics across the whole run: {max_subtopics}
- Max supervisor turns: {supervisor_max_turns}
- Each worker will run {max_iter} query iterations.

Call `set_strategy` now. If every candidate sub-topic would use the
same facet set (synonym variation only), emit one sub-topic. If
structurally different facet sets are needed, emit one per
sub-topic. Respect the parent's `NOT` scope.
"""


SUPERVISOR_USER_CONTINUE = """\
# Parent topic (re-shown every turn — respect its scope)

{topic_description}

{tool_results}

# Dispatched so far

{dispatched_block}

# Remaining to dispatch

{remaining_block}

# Overlap matrix (pairwise % paper-ID overlap between dispatched workers)

{overlap_matrix}

Plan your next action.
"""
