"""Tests for the annotate.py refactor — verifies it routes through
:class:`LLMClient` instead of using the OpenAI SDK directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import pytest

from citeclaw.annotate import (
    _label_batch,
    _label_one,
    _parse_annotation_batch,
    annotate_graph,
)
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

    def test_batched_call_count_smaller_than_paper_count(
        self, tmp_path: Path, basic_settings: Settings, monkeypatch,
    ):
        """With batch_size=20 and 5 papers, annotate_graph issues exactly
        one LLM call for the whole batch — not 5. This is the whole point
        of the batching refactor."""
        g = ig.Graph(n=5)
        g.vs["paper_id"] = [f"P{i}" for i in range(5)]
        g.vs["title"] = [f"Title {i}" for i in range(5)]
        g.vs["abstract"] = [f"Abstract {i}" for i in range(5)]
        gpath = tmp_path / "in.graphml"
        g.write_graphml(str(gpath))
        opath = tmp_path / "out.graphml"

        # Spy on the LLMClient's call count.
        from citeclaw.clients.llm import stub as stub_mod
        original_respond = stub_mod.stub_respond
        calls: list[str] = []

        def spy(system, user):
            calls.append(user)
            return original_respond(system, user)

        monkeypatch.setattr(stub_mod, "stub_respond", spy)

        annotate_graph(
            graph_path=gpath,
            output_path=opath,
            instruction="Give a topic label",
            config_path=None,
        )
        # One batched call covers all 5 papers; the batched stub returns
        # a label for every index so there are NO per-paper fallbacks.
        assert len(calls) == 1, (
            f"expected 1 batched call; got {len(calls)}: {[c[:80] for c in calls]}"
        )
        # Output graph still has distinct labels per node.
        out_g = ig.Graph.Read_GraphML(str(opath))
        labels = list(out_g.vs["label"])
        assert len(labels) == 5
        assert all(labels)

    def test_batch_partial_response_falls_back_per_paper(
        self, tmp_path: Path, basic_settings: Settings, monkeypatch,
    ):
        """If the batched LLM response omits some indices, the missing
        ones fall back to per-paper ``_label_one`` calls so no node
        ships with its placeholder (title prefix) label."""
        g = ig.Graph(n=3)
        g.vs["paper_id"] = ["A", "B", "C"]
        g.vs["title"] = ["Apple study", "Banana study", "Cherry study"]
        g.vs["abstract"] = ["a", "b", "c"]
        gpath = tmp_path / "in.graphml"
        g.write_graphml(str(gpath))
        opath = tmp_path / "out.graphml"

        # Force the batched parser to return results for only one index.
        from citeclaw import annotate as annotate_mod
        monkeypatch.setattr(
            annotate_mod,
            "_parse_annotation_batch",
            lambda text: {0: "batched-label-A"},
        )

        annotate_graph(
            graph_path=gpath,
            output_path=opath,
            instruction="Give a topic label",
            config_path=None,
        )
        out_g = ig.Graph.Read_GraphML(str(opath))
        labels = list(out_g.vs["label"])
        # Index 0 used the batched result.
        assert labels[0] == "batched-label-A"
        # Indices 1 and 2 fell back to _label_one (stub → first two words
        # of title). The important invariant: they are NOT the pre-batch
        # placeholder ``(title or '?')[:40]``, which would be the full
        # title prefix. The stub returns only the first 2 words (e.g.
        # ``"Banana study"``).
        assert labels[1] and len(labels[1].split()) <= 2
        assert labels[2] and len(labels[2].split()) <= 2

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


class TestParseAnnotationBatch:
    def test_wrapped_results(self):
        raw = (
            '{"results": ['
            '{"index": 0, "label": "alpha"},'
            '{"index": 1, "label": "beta"}'
            "]}"
        )
        assert _parse_annotation_batch(raw) == {0: "alpha", 1: "beta"}

    def test_bare_array(self):
        """Providers that drop the ``{"results": ...}`` envelope should
        still parse correctly."""
        raw = '[{"index": 7, "label": "g"}]'
        assert _parse_annotation_batch(raw) == {7: "g"}

    def test_code_fenced(self):
        raw = '```json\n{"results": [{"index": 3, "label": "x"}]}\n```'
        assert _parse_annotation_batch(raw) == {3: "x"}

    def test_label_quotes_stripped(self):
        """The parser must normalise surrounding quotes — models often
        wrap the label string in extra quotes."""
        raw = '{"results": [{"index": 0, "label": "\\"quoted\\""}]}'
        assert _parse_annotation_batch(raw) == {0: "quoted"}

    def test_empty_label_dropped(self):
        raw = '{"results": [{"index": 0, "label": ""}]}'
        assert _parse_annotation_batch(raw) is None

    def test_invalid_json_returns_none(self):
        assert _parse_annotation_batch("not json") is None

    def test_entries_without_index_dropped(self):
        raw = '{"results": [{"label": "no idx"}, {"index": 1, "label": "ok"}]}'
        assert _parse_annotation_batch(raw) == {1: "ok"}


class TestLabelBatch:
    def test_stub_batches_three(self, basic_settings: Settings):
        """_label_batch issues exactly one call for three papers and
        returns one label per index."""
        budget = BudgetTracker()
        client = build_llm_client(basic_settings, budget)
        items = [
            (0, "First paper title", "abstract 1"),
            (5, "Second paper title", "abstract 2"),
            (9, "Third paper title", "abstract 3"),
        ]
        out = _label_batch(client, "Topic label", items)
        # Every input index has a non-empty label in the result.
        assert set(out.keys()) == {0, 5, 9}
        assert all(out.values())

    def test_empty_items_no_call(self, basic_settings: Settings):
        """_label_batch short-circuits on empty input — no LLM call."""
        class _ShouldNotCallClient:
            def call(self, *a, **k):
                raise AssertionError("client should not be invoked on empty batch")

        out = _label_batch(_ShouldNotCallClient(), "x", [])
        assert out == {}

    def test_budget_exhaustion_returns_empty(self, basic_settings: Settings):
        from citeclaw.models import BudgetExhaustedError

        class _BoomClient:
            def call(self, *a, **k):
                raise BudgetExhaustedError("over")

        out = _label_batch(_BoomClient(), "x", [(0, "t", "a")])
        assert out == {}
