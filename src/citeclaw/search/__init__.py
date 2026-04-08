"""Local search utilities for the CiteClaw expansion family.

This package holds in-process predicates over already-fetched
``PaperRecord`` lists. It deliberately has zero S2 dependency: the
intent is post-fetch trimming when the S2 API can't express the
predicate (regex, abstract text search, arbitrary unions). The
``ExpandBy*`` family invokes :func:`apply_local_query` after enriching
candidates from the network to drop anything the user wants out before
LLM screening.

Submodules:
  - :mod:`citeclaw.search.query_engine` — pure ``apply_local_query``.
"""

from citeclaw.search.query_engine import apply_local_query

__all__ = ["apply_local_query"]
