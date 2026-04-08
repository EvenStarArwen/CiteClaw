"""Centralized LLM prompts for CiteClaw.

Every LLM use case in the project lives in its own submodule under this
package. The convention is one submodule per *task type*, with module-level
constants named ``SYSTEM`` (the system prompt) and ``USER_TEMPLATE`` (a
``str.format``-style template) plus any task-specific helpers.

Submodules:
  - :mod:`citeclaw.prompts.screening`     — used by ``LLMFilter`` query screening
  - :mod:`citeclaw.prompts.annotation`    — used by ``annotate.py`` for graph node labels
  - :mod:`citeclaw.prompts.topic_naming`  — used by ``cluster.representation`` for
    LLM-based cluster naming

Adding a new LLM use case? Create a new submodule here, point your call site
at it via ``from citeclaw.prompts import <new_module>``, and route the actual
LLM call through :class:`citeclaw.clients.llm.base.LLMClient` so it goes through
the same factory + budget tracking as everyone else.
"""
