"""ExpandBySearch V3 — tutorial-style worker + clean supervisor.

Design principles (differs from V2):

1. Supervisor keeps a clean brain. It proposes 3–6 sub-topics using
   only its pre-training knowledge — no seed papers, no topic_description
   pollution down to workers, no initial_query_sketch. Sub-topic spec
   is just {id, description}.

2. Worker runs a Bayesian-optimization-style loop with max_iter steps.
   Each iter: propose a query (acquisition), system auto-fetches and
   analyses (function evaluation), agent answers one-question-at-a-time
   checks (tutorial-style), then plans and rewrites.

3. Per-query analysis includes:
     - total_count + query_tree (per-clause counts)
     - diff vs previous query (new N, seen M)
     - topic modelling clusters (TF-IDF + k-means)
     - top-100 cited titles

4. No self-stop. Workers run to max_iter. Precision concerns
   (off-topic clusters) go into per-phase instructions, not stopping
   criteria.

5. Supervisor sees per-subagent overlap matrix + topic clusters +
   top-cited titles. No single-paper visibility.
"""
