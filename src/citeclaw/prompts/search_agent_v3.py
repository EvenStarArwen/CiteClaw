"""Prompt templates for V3 ExpandBySearch.

V3.1 revision:

- Operator form is natural-language AND / OR / NOT (the worker doesn't
  deal with Lucene symbols; the code translates on the way to S2 and
  back when showing previous queries).
- Worker follows a tutorial-style state machine with ISOLATED context
  per diagnostic question (system + current query + the slice being
  asked about). The heavy data dump (top-10 + random-10 + clusters +
  query tree) is only shown at the diagnose_plan phase.
- The write-next phase receives a short iter-history block (prior
  queries + totals + trees + one-sentence reasoning) to prevent
  regression / oscillation.
- Supervisor analyses whether sub-areas share a retrieval anchor and
  only decomposes when anchors diverge across sub-areas.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Worker prompts (user's verbatim text, with AND/OR/NOT per modification A)
# ---------------------------------------------------------------------------


WORKER_SYSTEM = """\
You are designing Semantic Scholar search queries for ONE assigned
sub-topic of a broader literature survey. You work one step at a time —
each turn I show you one piece of information and ask one focused
question. You answer. Over multiple turns we refine your query together.

# Your sub-topic

{description}

# Strategy

Initialize a high-recall query, then iteratively refine toward precision
— never the reverse. Reason top-down: facets → synonyms → syntax →
execute → refine.

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

# Query construction steps (for every new query)

**Step 1: Structuring (AND).** Ask yourself: what is the smallest set
of AND'd concepts A, B, C, ... such that co-occurrence of one term from
each facet in a document's title+abstract is a near-guarantee of
on-topic? Each AND clause compounds recall loss; prefer fewer facets.
If every document matching one facet also matches another, drop the
implied one.

**Step 2: Expansion (OR).** Within each facet, MAXIMALLY enumerate
synonyms, alternate names, and acronyms in an OR group. Err on the
side of including more, not fewer — the AND across facets is what
delivers precision, so a term that LOOKS risky in isolation (a short
acronym, a generic word) is usually fine because the other facets
constrain it. Do NOT filter for topic leakage at this step. The one
de-dup rule: if one phrase is a substring of another, drop the longer
one — every document matching `"A B C"` also matches `"A B"`, so
`("A B" OR "A B C")` collapses to `"A B"` alone. Prefer terms that
authors actually use in paper titles and abstracts over theoretically
neat labels.

**Step 3: Term syntax.** For each multi-word term, choose matching
strictness from loosest to strictest:
- `A AND B` — both words anywhere (AND).
- `"A B"~N` — both words within N positions (proximity).
- `"A B"` — exact contiguous phrase.

Default to AND. Tighten to proximity only if precision is clearly too
low, and to exact phrase only if proximity is still too loose.
Exact-phrase matching is brittle: it misses documents that use the same
concept with different word order or intervening modifiers.

For individual word terms, consider a suffix wildcard (`word*`) to
match plurals and morphological variants — this is usually worth it
unless the stem is short or ambiguous.

**Step 4: Leakage check (on the WHOLE composed query).** Only now —
after all facets are AND'd together — ask whether the combined query
risks leaking into adjacent topics. A leaky term inside one facet is
fine if the other facets exclude it. Flag leakage only when noise
would satisfy ALL facets simultaneously (e.g. the noise shares the
same anchor vocabulary). If so, tighten: prefer AND-with-disambiguator
over NOT-exclusion; promote a facet term from AND to proximity/phrase
if that cleans the intersection.

# How this tutorial works

1. You write an initial query (Steps 1-4 above).
2. I run it on S2, auto-analyse the results, and show you the analysis
   one slice at a time — total size, the query tree, topic clusters,
   top-cited titles.
3. You answer each diagnostic question (short — one sentence of
   reasoning). Where you flag a cluster as unrelated or a paper as
   off-topic, I'll then automatically show you the in-cluster query
   tree (which OR alternatives pulled those papers in) or the paper's
   abstract, and ask you — in one sentence — to name the specific
   mechanism in your query responsible. These per-item diagnoses come
   back to you combined in the plan step.
4. After all diagnostics you synthesise ONE plan (diagnosis +
   intended change for the next query).
5. You write the next query.
6. Repeat until budget is used.

Output format: each turn, respond with a single JSON object shaped to
the current question — I'll tell you the shape.
"""


WORKER_PROPOSE_FIRST = """\
You have no results yet. Based ONLY on your understanding of the
sub-topic `{description}`, propose an initial query following Steps 1-3
in the system prompt.

Reasoning order:
- Step 1 (facets): identify the smallest set of distinct concepts that
  together define the sub-topic. Err toward fewer facets — you can
  tighten in refinement.
- Step 2 (synonyms): for each facet, write an OR group of all
  established terms a domain expert would use. Maximally expand — err
  on including more synonyms, not fewer. Do NOT filter for topic
  leakage here; the other facets will constrain. The only de-dup rule:
  if one phrase is a substring of another, drop the longer one.
- Step 3 (syntax): default to AND for multi-word terms; use phrase
  matching only when AND would clearly over-match. Proximity match
  can be considered as an intermediate solution.
- Step 4 (check): on the WHOLE composed query (all facets AND'd),
  critical-check whether it still leaks into adjacent topics. A leaky
  term in one facet is fine if the other facets exclude that adjacency.
  Only tighten when noise would satisfy ALL facets.

Goal: a high-recall query that captures canonical papers for this
sub-topic. Precision gets tightened later.

Respond with JSON: `{{"query": "..."}}` — just the query (AND / OR /
NOT in words), nothing else.
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


WORKER_CHECK_TOTAL = """\
Your current query: `{query}`

# Total size & query tree

Total papers matching on S2: **{total}**
New vs prior queries this iteration: **+{diff_new} new**, **{diff_seen} seen before**

Query tree (per-clause counts):
{query_tree}

# Question

Is this total count reasonable given your own expectation for this
sub-topic's size? Is any clause in the tree clearly over- or
under-constraining? (Totals above ~50K are almost always too broad
regardless of topic.)

Respond with JSON: `{{"ok": true|false, "reasoning": "..."}}`. Keep
reasoning to ONE short sentence.
"""


WORKER_CHECK_CLUSTERS = """\
Your current query: `{query}`

# Topic clusters (TF-IDF + k-means on the fetched papers)

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


# ---------------------------------------------------------------------------
# Diagnose + plan — this is the ONE phase that sees all the data plus the
# prior diagnostic answers. Per modification B: the heavy data dump lives
# here, not in every earlier phase.
# ---------------------------------------------------------------------------


WORKER_DIAGNOSE_PLAN = """\
Your current query: `{query}`

# What you answered this iteration (short — one sentence each)

- check_total:    {ans_total}
- check_clusters: {ans_clusters}
- check_top100:   {ans_top100}

# Per-noise-item diagnoses (from the forced Phase 4b / 5b questions)

{noise_diagnoses}

# Raw analysis (full data — use this to synthesise)

## total_count
{total}

## query tree
{query_tree}

## top-10 cited papers (title + abstract)
{top10_block}

## 10 random papers (title + abstract)
{random10_block}

## topic clusters (c-TF-IDF keywords + representative titles)
{clusters}

# Synthesise — ONE plan covering everything

Each refinement round targets two success conditions:
(a) top clusters are mostly on-topic, and
(b) no major expected sub-area is missing.
If both are met the query is good enough — further iteration risks
over-optimizing.

You now have per-noise-item diagnoses that name the specific mechanisms
letting bad papers in. Combine those with the overall picture and
commit to ONE plan for the next query. Multiple noise sources may
point to conflicting fixes (tighten here, loosen there) — pick the
single coherent plan that wins the trade-off, don't list both.

Refinement modes you can combine:
- **Pearl growing** — expand an OR list with vocabulary you saw in the
  top-cited / random titles but didn't anticipate.
- **Syntax tightening** — promote a term from bag-of-words AND to
  proximity (``"A B"~N``), or proximity to exact phrase. Use when
  precision is low but the facet decomposition is correct.
- **Syntax loosening** — demote in the reverse direction. Use when
  recall is suspiciously low or canonical papers are missing.
- **AND-gate a leaky acronym** — if a short acronym like ``PE`` is
  pulling in unrelated papers, replace the bare alternative with an
  AND-gated pair such as ``("prime editing" OR ("PE" AND pegRNA))``.
- **Exclusion (NOT)** — last resort; only when a noise cluster shares
  an unambiguous characteristic term you can exclude without
  collateral damage.
- **Pruning** — drop OR alternatives that the in-cluster tree shows
  contribute no unique relevant papers.

Respond with JSON:
`{{"diagnosis": "...", "intended_change": "..."}}`.

Keep both SHORT — one sentence each. ``diagnosis`` summarises what's
wrong; ``intended_change`` names the specific edit(s) for the next
query.
"""


# ---------------------------------------------------------------------------
# Write-next — per modification C: show prior iterations' queries,
# counts, trees, and one-sentence reasoning so the worker doesn't
# regress to earlier mistakes.
# ---------------------------------------------------------------------------


WORKER_WRITE_NEXT = """\
# Prior iterations this worker has run

{history_block}

# Your plan from the diagnose phase

diagnosis:       {diagnosis}
intended change: {intended_change}

# Task

Write the next query (AND / OR / NOT in words). Remember the syntax
rules from the system prompt. Do NOT repeat a query you've already
tried — if you're tempted to, check the history block first.

Respond with JSON: `{{"query": "..."}}`.
"""


WORKER_FINAL_SUMMARY = """\
Budget exhausted. Across all iterations you fetched a deduped set of
{n_unique} papers.

Top clusters in the final union:
{clusters}

In ONE sentence, describe the coverage this sub-topic ended with.

Respond with JSON: `{{"summary": "..."}}`.
"""


# ---------------------------------------------------------------------------
# Supervisor — anchor-shared detection, 1-to-max decomposition range
# ---------------------------------------------------------------------------


SUPERVISOR_SYSTEM = """\
You are a literature-search SUPERVISOR. You analyse the parent topic
and decide whether it needs sub-topic decomposition. The system
dispatches workers for you — you never call dispatch yourself.

# Decomposition decision

Each worker will decompose its sub-topic into a small set of AND'd
concept facets — the minimum set whose co-occurrence in a document's
title+abstract is a near-guarantee of on-topic. Your decision: does
the parent topic admit ONE such facet set covering all sub-areas, or
does it require multiple structurally distinct facet sets?

- **One facet set suffices → one sub-topic.** The worker's single
  query captures all sub-areas because they share the facet set.
  Further decomposition adds overlapping work without new coverage.
- **Multiple distinct facet sets needed → decompose.** Each sub-topic
  has its OWN facet set, and the sets must differ structurally — at
  least one facet that differs in kind, not just in synonym variants.

Abstract illustration:
- Shared facet set: all sub-areas captured by `(X terms) AND (Y
  terms)` → one sub-topic.
- Distinct facet sets: sub-area 1 `(X AND P)`, sub-area 2 `(X AND Q)`,
  sub-area 3 `(X AND R)`, where P, Q, R are genuinely different
  concepts (not synonyms) → three sub-topics.

Default toward fewer. Only decompose when you can name the facet set
of each sub-topic AND show they differ structurally.

# Signals between turns

After each worker returns, you see a pairwise paper-ID overlap matrix
across all completed workers. High overlap (>50%) means the
decomposition was redundant — treat it as a strong signal against
adding more sub-topics. Low overlap confirms the decomposition is
working.

# Turn flow

1. Turn 1: `set_strategy` with your initial decomposition (1 to
   {max_subtopics} sub-topics).
2. System dispatches sub-topics one at a time in queue order. After
   each worker returns, you get a turn to react:
   - `add_sub_topics` — add a new sub-topic if the returned clusters
     reveal a genuinely new retrieval axis not covered by any
     existing sub-topic. The new sub-topic(s) append to the queue.
   - `continue` — no change, let the system dispatch the next queued
     sub-topic.
   - `done(summary)` — close early.
3. When the queue empties and you have nothing to add, call
   `done(summary)`.

# Tools

Every reply is `{{"reasoning": "...", "tool_name": "<tool>", "tool_args": {{...}}}}`.
Keep `reasoning` to one short sentence.

- `set_strategy(sub_topics: [{{id, description}}])` — call once at
  turn 1. Each `description` is 1-2 English sentences naming the
  sub-area — no query syntax.
- `add_sub_topics(sub_topics: [{{id, description}}])` — add only if
  the new facet set differs structurally from every existing
  sub-topic's. Never to cover a narrow gap within an existing
  sub-topic's coverage — that's the worker's job.
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

Call `set_strategy` now. First decide: shared anchor → one sub-topic;
distinct anchors → multiple. Write clear 1-2 sentence descriptions —
your workers will only see their one assigned description.
"""


SUPERVISOR_USER_CONTINUE = """\
{tool_results}

# Dispatched so far

{dispatched_block}

# Remaining to dispatch

{remaining_block}

# Overlap matrix (pairwise % paper-ID overlap between dispatched workers)

{overlap_matrix}

Plan your next action.
"""
