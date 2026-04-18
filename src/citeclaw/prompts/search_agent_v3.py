"""Prompt templates for V3 ExpandBySearch.

Tutorial-style: each worker turn shows one piece of information and asks
one focused question. The conversation stays coherent — the LLM's
context carries prior answers forward — but each step's user message
stays focused on a single decision.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Worker: tutorial-style phases, one question per turn
# ---------------------------------------------------------------------------


WORKER_SYSTEM = """\
You are designing Semantic Scholar search queries for ONE assigned
sub-topic of a broader literature survey. You work one step at a time —
each turn I show you one piece of information and ask one focused
question. You answer. Over multiple turns we refine your query together.

# Your sub-topic

{description}

# Semantic Scholar query syntax

Use Lucene-style operators:
- `+` AND (must contain)
- `|` OR
- `-` NOT (must not contain)
- `"..."` exact phrase
- `(...)` grouping

Rules:
- USE SYMBOLS, never the words AND / OR / NOT.
- Wrap every OR group in parentheses: `("A" | "B") +"C"`.
- Disambiguate acronyms with the full phrase in an OR group:
  `("adenine base editor" | "ABE")` not bare `"ABE"`.
- Keep top-level AND arity ≤ 3.

# How this tutorial works

1. You write an initial query.
2. I run it on S2, auto-analyse the results, and show you the analysis
   one slice at a time — total size, the query tree, topic clusters,
   top-cited titles.
3. You answer each diagnostic question (yes/no + 1-2 sentences of
   reasoning). Each answer narrows where the query is weak.
4. After the diagnostics, you synthesise what's wrong and plan the next
   query. You may call `inspect_topic(cluster_id)` or
   `inspect_paper(title)` to dig deeper first.
5. You write the next query.
6. Repeat until budget is used.

Output format: each turn, respond with a single JSON object shaped to
the current question — I'll tell you the shape.
"""


WORKER_PROPOSE_FIRST = """\
You have no results yet. Based ONLY on your understanding of the
sub-topic `{description}`, propose an initial Lucene query.

Principles for the first query:
- Target papers that a domain expert would consider canonical or
  representative of this sub-topic.
- Be concrete — prefer full phrases over vague words.
- Keep arity ≤ 3 so S2 returns >0 hits.

Respond with JSON: `{{"query": "..."}}` — just the Lucene query, nothing
else.
"""


WORKER_CHECK_TOTAL = """\
Your query: `{query}`

# Total size & query tree

Total papers matching on S2: **{total}**
New vs prior queries: **+{diff_new} new**, **{diff_seen} seen before**

Query tree (per-clause counts):
{query_tree}

# Question

Is this total count reasonable for the sub-topic? Too narrow (<50)? Too
broad (>10K)? Any clause in the tree clearly over- or under-constraining?

Respond with JSON: `{{"ok": true|false, "reasoning": "..."}}`.
"""


WORKER_CHECK_CLUSTERS = """\
Your query: `{query}`

# Topic clusters (from title+abstract TF-IDF on the fetched papers)

{clusters}

# Question

Look at each cluster. Are any of them **obviously unrelated** to the
sub-topic (`{description}`)? Also: are there major sub-areas of the
sub-topic that you'd expect to see but that DO NOT appear as a cluster?

Respond with JSON:
`{{"unrelated_clusters": [cluster_ids], "missing_areas": ["..."], "reasoning": "..."}}`.
"""


WORKER_CHECK_TOP100 = """\
Your query: `{query}`

# Top-100 papers by citation count (titles only)

{titles}

# Question

Scan for red flags:
- Any paper clearly **off-topic** that shouldn't be here?
- Any paper you'd strongly expect to see in the top-cited list but
  that's **missing**?

You do NOT need to list every single one. Just flag what's notable.

Respond with JSON:
`{{"off_topic_titles": ["..."], "missing_expected": ["..."], "reasoning": "..."}}`.
"""


WORKER_DIAGNOSE_PLAN = """\
You've inspected this query. Now synthesise: what are the concrete
problems with the current query, and what would the next query do
differently?

You may call one of these tools first, or go straight to a plan:
- `inspect_topic(cluster_id)` — see 5 representative abstracts from
  a cluster.
- `inspect_paper(title)` — look up a paper by title (tells you if it's
  in the current results, and shows its abstract if S2 knows it).
- `plan` — commit to a next-query plan.

Respond with JSON, ONE of:
- `{{"tool": "inspect_topic", "cluster_id": N}}`
- `{{"tool": "inspect_paper", "title": "..."}}`
- `{{"tool": "plan", "diagnosis": "...", "intended_change": "..."}}`

You can call tools multiple turns before committing to `plan`.
"""


WORKER_WRITE_NEXT = """\
Based on your plan:
  diagnosis: {diagnosis}
  intended change: {intended_change}

Write the next Lucene query. Remember: symbols not words, parens on OR
groups, arity ≤ 3.

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
# Supervisor: concise, plan-only
# ---------------------------------------------------------------------------


SUPERVISOR_SYSTEM = """\
You are a literature-search SUPERVISOR. You decompose the parent topic
into sub-topics, dispatch one worker per sub-topic, and when all
workers return you close the run.

# What you do — and don't do

- You design the decomposition using your OWN KNOWLEDGE of the field.
- You do NOT look at seed papers. You do NOT read any paper.
- You do NOT write queries — workers handle that.
- A sub-topic is just `{{id, description}}`. The description is a 1-2
  sentence English description — no query syntax.
- Initial decomposition: **3 to {max_subtopics} sub-topics**. Think like
  a human reviewer sketching the major axes of a survey. Fewer, each
  distinct, is better than many overlapping ones.
- After each worker returns you'll see an OVERLAP matrix (pairwise %
  paper-ID overlap between the new worker and each prior worker). If a
  new worker overlaps >50% with a prior worker on average, your
  decomposition was redundant — next time add fewer.

# Tools (one per turn, single JSON object reply)

Every reply is `{{"reasoning": "...", "tool_name": "<tool>", "tool_args": {{...}}}}`.

- `set_strategy(sub_topics: [{{id, description}}])` — call once at
  turn 1. 3 to {max_subtopics} entries.
- `add_sub_topics(sub_topics: [{{id, description}}])` — only when a
  worker's returned topic clusters reveal a genuinely unexplored
  retrieval axis that no existing sub-topic covers. Never add to cover
  a narrow gap — workers individually go deep.
- `dispatch_sub_topic_worker(spec_id)` — launch the worker for a
  sub-topic. One at a time.
- `done(summary)` — close the run.

# Closing mantra

Coverage comes from the decomposition. Precision comes from the
downstream filter. Your job is just the outline.
"""


SUPERVISOR_USER_FIRST = """\
# Parent topic

{topic_description}

# Your budget

- Max sub-topics across the whole run: {max_subtopics}
- Max supervisor turns: {supervisor_max_turns}
- Each worker will run {max_iter} query iterations.

Call `set_strategy` now. Write clear 1-2 sentence descriptions — your
workers will ONLY see their one assigned description, not this topic
prompt.
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
