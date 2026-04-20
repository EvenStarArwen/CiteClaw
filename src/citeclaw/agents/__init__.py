"""Reserved for the next ExpandBySearch agent backend.

The v2 supervisor/worker implementation that previously lived here was
cleared on 2026-04-20 pending a rewrite. The pipeline-side wiring
(``citeclaw.steps.expand_by_search.ExpandBySearch``) is intact and
exposes a ``_screen_and_finalize`` helper that the new agent should
hand its aggregate paper_ids to.
"""
