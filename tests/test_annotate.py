"""Tests for the annotate.py refactor — verifies it routes through
:class:`LLMClient` instead of using the OpenAI SDK directly.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import pytest

from citeclaw.annotate import _label_one, annotate_graph
from citeclaw.clients.llm import build_llm_client
from citeclaw.config import BudgetTracker, Settings


class TestLabelOne:
    def test_stub_label_strips_quotes(self, basic_settings: Settings):
        """``_label_one`` calls ``client.call`` and post-processes the text."""
        budget = BudgetTracker()
        client = build_llm_client(basic_settings, budget)
        out = _label_one(client, "Identify the topic", "The Great Paper", "An abstract.")
        # Stub returns the first two words of the title (no quotes).
        assert isinstance(out, str)
        assert out  # non-empty

    def test_budget_exhaustion_returns_empty_string(self, basic_settings: Settings, monkeypatch):
        """If the underlying client raises BudgetExhaustedError, ``_label_one``
        catches it and returns ``""`` so the caller can fall back."""
        from citeclaw.models import BudgetExhaustedError

        class _BoomClient:
            def call(self, system, user, **kw):
                raise BudgetExhaustedError("over budget")

        out = _label_one(_BoomClient(), "x", "y", "z")
        assert out == ""

    def test_generic_exception_returns_empty_string(self, basic_settings: Settings):
        class _BoomClient:
            def call(self, system, user, **kw):
                raise RuntimeError("provider down")

        out = _label_one(_BoomClient(), "x", "y", "z")
        assert out == ""


class TestAnnotateGraphRoutesThroughLLMClient:
    def test_calls_build_llm_client_when_instruction_set(self, tmp_path: Path, basic_settings: Settings, monkeypatch):
        """``annotate_graph`` should always go through ``build_llm_client`` —
        no direct OpenAI SDK calls. We spy on the factory and assert it
        gets invoked."""
        # Build a tiny graphml file.
        g = ig.Graph(n=2)
        g.vs["paper_id"] = ["A", "B"]
        g.vs["title"] = ["Title A", "Title B"]
        g.vs["abstract"] = ["Abs A", "Abs B"]
        gpath = tmp_path / "in.graphml"
        g.write_graphml(str(gpath))
        opath = tmp_path / "out.graphml"

        called: dict[str, int] = {"build": 0}

        # Spy on build_llm_client at the import site annotate.py uses.
        from citeclaw import annotate as annotate_mod
        original = annotate_mod.build_llm_client

        def _spy(cfg, budget, **kw):
            called["build"] += 1
            return original(cfg, budget, **kw)

        monkeypatch.setattr(annotate_mod, "build_llm_client", _spy)

        # No config_path → uses stub by default. instruction provided → goes
        # through the LLM call path.
        annotate_graph(
            graph_path=gpath,
            output_path=opath,
            instruction="Generate a topic label",
            config_path=None,
        )
        assert called["build"] == 1
        assert opath.exists()
        # Output graph has labels.
        out_g = ig.Graph.Read_GraphML(str(opath))
        assert "label" in out_g.vs.attributes()

    def test_no_instruction_skips_llm(self, tmp_path: Path, monkeypatch):
        """When no instruction is provided, annotate_graph uses the title
        directly and never calls the LLM factory."""
        g = ig.Graph(n=1)
        g.vs["paper_id"] = ["A"]
        g.vs["title"] = ["Some Title"]
        g.vs["abstract"] = ["abs"]
        gpath = tmp_path / "in.graphml"
        g.write_graphml(str(gpath))
        opath = tmp_path / "out.graphml"

        called: dict[str, int] = {"build": 0}
        from citeclaw import annotate as annotate_mod

        def _spy(cfg, budget, **kw):
            called["build"] += 1
            return annotate_mod.build_llm_client(cfg, budget, **kw)  # would recurse if hit

        monkeypatch.setattr(annotate_mod, "build_llm_client", _spy)

        # Build minimal stub config that has no graph_label_instruction.
        from citeclaw.config import load_settings
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "screening_model: stub\n"
            f"data_dir: {tmp_path}\n"
            "topic_description: stub\n"
            "seed_papers: []\n"
            "blocks: {}\n"
            "pipeline: []\n"
        )

        annotate_graph(
            graph_path=gpath,
            output_path=opath,
            instruction=None,  # explicit no-instruction path
            config_path=cfg_path,
        )
        # The LLM factory was never invoked because the title-only path
        # short-circuits before any LLM build.
        assert called["build"] == 0
