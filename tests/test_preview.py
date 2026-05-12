"""Tests for citeclaw.preview — the pipeline ASCII diagram + prompt."""
from __future__ import annotations

import io

import pytest

from citeclaw.config import Settings
from citeclaw.preview import extract, render_pipeline
from citeclaw.preview.flow_box import render
from citeclaw.preview.model import StepNode
from citeclaw.preview.prompt import _read_key, confirm, select


def _settings(**pipeline) -> Settings:
    """Build a stub-mode Settings with given pipeline + minimum blocks."""
    return Settings(
        screening_model="stub",
        s2_api_key="dummy",
        topic_description="t",
        seed_papers=[],
        blocks={
            "yf": {"type": "YearFilter", "min": 2020},
            "screener": {"type": "Sequential", "layers": ["yf"]},
        },
        **pipeline,
    )


class TestExtract:
    def test_linear_pipeline(self):
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 200, "screener": "screener"},
            {"step": "Finalize"},
        ])
        nodes = extract(cfg.pipeline_built)
        assert [n.name for n in nodes] == ["LoadSeeds", "ExpandForward", "Finalize"]
        # ExpandForward params surfaced.
        ef = nodes[1]
        assert ("max_citations", "200") in ef.params

    def test_parallel_with_nested_branches(self):
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "Parallel", "branches": [
                [{"step": "ExpandForward", "max_citations": 100, "screener": "screener"}],
                [{"step": "ExpandBackward", "screener": "screener"}],
            ]},
        ])
        nodes = extract(cfg.pipeline_built)
        par = nodes[1]
        assert par.name == "Parallel"
        assert len(par.branches) == 2
        assert par.branches[0][0].name == "ExpandForward"
        assert par.branches[1][0].name == "ExpandBackward"


class TestRenderFlowBox:
    def test_simple_linear_renders_three_boxes(self):
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 100, "screener": "screener"},
            {"step": "Finalize"},
        ])
        out = render_pipeline(cfg.pipeline_built, width=80)
        assert "LoadSeeds" in out
        assert "ExpandForward" in out
        assert "Finalize" in out
        # The arrow connector should appear between each pair.
        assert "▼" in out
        # ExpandForward's side annotation uses the compact "n=" form.
        assert "n=100" in out

    def test_parallel_shows_divergence_and_merge(self):
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "Parallel", "branches": [
                [{"step": "ExpandForward", "max_citations": 100, "screener": "screener"}],
                [{"step": "ExpandBackward", "screener": "screener"}],
            ]},
            {"step": "Finalize"},
        ])
        out = render_pipeline(cfg.pipeline_built, width=100)
        assert "Parallel" in out
        # Circuit-style divergence: ┌──┴──┐ + merge connector + union annotation
        assert "┴" in out  # ┴ at the centre of the split bar
        assert "┌" in out and "┐" in out  # corners of split bar AND of step boxes
        assert "∪" in out
        assert "union by paper_id" in out
        assert "Branch 1" in out
        assert "Branch 2" in out

    def test_asymmetric_branch_bottom_aligned(self):
        """Rerank + ExpandForward (2 steps) vs just ExpandBackward (1 step) —
        the shorter branch must pad with a vertical pipe so the merge
        aligns at the bottom of both columns."""
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "Cluster", "store_as": "t", "algorithm": {"type": "louvain"}},
            {"step": "Parallel", "branches": [
                [
                    {"step": "Rerank", "metric": "pagerank", "k": 20,
                     "diversity": {"cluster": "t"}},
                    {"step": "ExpandForward", "max_citations": 100,
                     "screener": "screener"},
                ],
                [{"step": "ExpandBackward", "screener": "screener"}],
            ]},
            {"step": "Finalize"},
        ])
        out = render_pipeline(cfg.pipeline_built, width=100)
        lines = out.split("\n")

        # Sanity: both branch boxes exist.
        assert any("Rerank" in ln for ln in lines)
        assert any("ExpandForward" in ln for ln in lines)
        assert any("ExpandBackward" in ln for ln in lines)

        # ExpandForward and ExpandBackward must appear on the SAME line
        # (the bottom-row of the diverging columns), since the shorter
        # branch is bottom-aligned.
        forward_line = next(
            i for i, ln in enumerate(lines) if "ExpandForward" in ln
        )
        assert "ExpandBackward" in lines[forward_line]

    def test_trailing_whitespace_stripped(self):
        cfg = _settings(pipeline=[
            {"step": "LoadSeeds"},
            {"step": "Finalize"},
        ])
        out = render_pipeline(cfg.pipeline_built, width=80)
        for line in out.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace: {line!r}"


class TestRenderRawNodes:
    def test_empty_pipeline(self):
        assert render([], width=80) == ""

    def test_handcrafted_nodes(self):
        nodes = [
            StepNode(idx=1, name="LoadSeeds", params=[]),
            StepNode(
                idx=2, name="ExpandForward",
                params=[("max_citations", "50")],
            ),
        ]
        out = render(nodes, width=80)
        assert "LoadSeeds" in out
        assert "ExpandForward" in out
        assert "n=50" in out


# ---------------------------------------------------------------------------
# Arrow-key prompt
# ---------------------------------------------------------------------------


class TestReadKey:
    @pytest.mark.parametrize("raw, expected", [
        ("\x1b[A", "UP"),
        ("\x1b[B", "DOWN"),
        ("\x1b[C", "RIGHT"),
        ("\x1b[D", "LEFT"),
        ("\r", "ENTER"),
        ("\n", "ENTER"),
        ("\x03", "INT"),
        ("y", "Y"),
        ("Y", "Y"),
        ("n", "N"),
        ("k", "K"),
        ("j", "J"),
    ])
    def test_key_mapping(self, raw, expected):
        assert _read_key(io.StringIO(raw)) == expected

    def test_unknown_char_passes_through(self):
        assert _read_key(io.StringIO("q")) == "q"

    def test_lone_escape_returns_esc(self):
        # ESC without a following [X gets caught when read(2) returns empty.
        assert _read_key(io.StringIO("\x1b")) == "ESC"


class TestSelectFallback:
    def test_non_tty_returns_default(self, monkeypatch):
        # When stdin isn't a tty, select() must return the default index
        # immediately rather than blocking on input.
        class _FakeStdin:
            def isatty(self):
                return False
        monkeypatch.setattr("citeclaw.preview.prompt.sys.stdin", _FakeStdin())
        assert select(["A", "B", "C"], default_idx=2) == 2

    def test_empty_options_returns_none(self):
        assert select([], default_idx=0) is None


class TestConfirmFallback:
    def test_non_tty_returns_true(self, monkeypatch):
        # Non-interactive runs should auto-proceed (matches the old behaviour
        # so piping into the CLI doesn't hang).
        class _FakeStdin:
            def isatty(self):
                return False
        monkeypatch.setattr("citeclaw.preview.prompt.sys.stdin", _FakeStdin())
        assert confirm("Go?") is True
