"""Tests for the ``python -m citeclaw rebuild-graph`` subcommand.

The rebuild command reconstructs ``citation_network.graphml`` and
``collaboration_network.graphml`` from an on-disk data directory
(``literature_collection*.json`` + ``cache.db`` + ``run_state*.json``)
without re-running the pipeline. These tests exercise it fully offline
via the fake S2 client.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import pytest
import yaml

from citeclaw.__main__ import _run_rebuild_graph
from citeclaw.cache import Cache
from citeclaw.budget import BudgetTracker
from citeclaw.config import load_settings
from citeclaw.context import Context
from citeclaw.pipeline import run_pipeline
from tests.fakes import build_chain_corpus


def _write_pipeline_config(tmp_path: Path) -> Path:
    """Minimal offline pipeline config: LoadSeeds → ExpandForward → Finalize."""
    data_dir = tmp_path / "data"
    cfg = {
        "screening_model": "stub",
        "data_dir": str(data_dir),
        "max_papers_total": 100,
        "llm_batch_size": 8,
        "llm_concurrency": 1,
        "seed_papers": [{"paper_id": "SEED"}],
        "blocks": {
            "yr": {"type": "YearFilter", "min": 2000, "max": 2030},
            "forward": {"type": "Sequential", "layers": ["yr"]},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "forward"},
            {"step": "ExpandBackward", "screener": "forward"},
            {"step": "Finalize"},
        ],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def _run_pipeline_with_fake_s2(cfg_path: Path) -> Path:
    """Run the pipeline once using the fake S2 client; return the data dir."""
    cfg = load_settings(cfg_path)
    cache = Cache(cfg.data_dir / "cache.db")
    budget = BudgetTracker()
    s2 = build_chain_corpus()
    ctx = Context(config=cfg, s2=s2, cache=cache, budget=budget)
    try:
        run_pipeline(ctx)
    finally:
        cache.close()
    return cfg.data_dir


@pytest.fixture
def populated_data_dir(tmp_path: Path, monkeypatch) -> Path:
    """Run the pipeline once and return the data_dir containing artifacts.

    We monkey-patch ``build_context`` so the rebuild command also uses the
    fake S2 client — the rebuild is otherwise identical to a real run
    except for which client it dials.
    """
    cfg_path = _write_pipeline_config(tmp_path)
    return _run_pipeline_with_fake_s2(cfg_path)


def _patch_build_context_to_use_fake_s2(monkeypatch):
    """Make ``citeclaw.__main__.build_context`` return a Context wired to FakeS2Client."""
    import citeclaw.__main__ as main_mod

    def _fake_build_context(config):
        cache = Cache(config.data_dir / "cache.db")
        budget = BudgetTracker()
        s2 = build_chain_corpus()
        ctx = Context(config=config, s2=s2, cache=cache, budget=budget)
        return ctx, s2, cache

    monkeypatch.setattr(main_mod, "build_context", _fake_build_context)


class TestRebuildGraph:
    def test_writes_regen_files_by_default(self, populated_data_dir: Path, monkeypatch):
        """Default rebuild writes ``.regen.graphml`` siblings without
        touching the originals."""
        data_dir = populated_data_dir
        orig_cit = data_dir / "citation_network.graphml"
        orig_collab = data_dir / "collaboration_network.graphml"
        assert orig_cit.exists()
        assert orig_collab.exists()

        # Corrupt the originals so we can prove they are untouched.
        orig_cit.write_text("NOT A REAL GRAPH")
        orig_collab.write_text("NOT A REAL GRAPH")

        _patch_build_context_to_use_fake_s2(monkeypatch)
        _run_rebuild_graph([str(data_dir)])

        regen_cit = data_dir / "citation_network.regen.graphml"
        regen_collab = data_dir / "collaboration_network.regen.graphml"
        assert regen_cit.exists()
        assert regen_collab.exists()
        # Originals are untouched: still contain our corruption marker.
        assert "NOT A REAL GRAPH" in orig_cit.read_text()
        assert "NOT A REAL GRAPH" in orig_collab.read_text()
        # New files parse cleanly.
        g = ig.Graph.Read_GraphML(str(regen_cit))
        assert g.vcount() >= 2
        cg = ig.Graph.Read_GraphML(str(regen_collab))
        assert cg.vcount() >= 1

    def test_force_overwrites_originals(self, populated_data_dir: Path, monkeypatch):
        """``--force`` overwrites the original filenames instead of writing
        a ``.regen`` variant."""
        data_dir = populated_data_dir
        orig_cit = data_dir / "citation_network.graphml"
        # Wreck the original.
        orig_cit.write_text("BROKEN")

        _patch_build_context_to_use_fake_s2(monkeypatch)
        _run_rebuild_graph([str(data_dir), "--force"])

        # Original is overwritten with a valid graph.
        assert "BROKEN" not in orig_cit.read_text()
        g = ig.Graph.Read_GraphML(str(orig_cit))
        assert g.vcount() >= 2
        # No .regen files should have been created.
        assert not (data_dir / "citation_network.regen.graphml").exists()

    def test_missing_data_dir_exits_cleanly(self, tmp_path: Path):
        """Pointing rebuild at a non-existent dir should exit with code 1."""
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(SystemExit) as excinfo:
            _run_rebuild_graph([str(bogus)])
        assert excinfo.value.code == 1

    def test_empty_data_dir_exits_cleanly(self, tmp_path: Path, monkeypatch):
        """A directory that exists but has no literature_collection*.json
        should error out via the load_checkpoint FileNotFoundError branch."""
        empty = tmp_path / "empty"
        empty.mkdir()
        _patch_build_context_to_use_fake_s2(monkeypatch)
        with pytest.raises(SystemExit) as excinfo:
            _run_rebuild_graph([str(empty)])
        assert excinfo.value.code == 1

    def test_regen_graph_has_new_edge_attributes(
        self, populated_data_dir: Path, monkeypatch,
    ):
        """The rebuilt graph should carry the semantic_similarity and
        weight attributes from Revision 1."""
        _patch_build_context_to_use_fake_s2(monkeypatch)
        _run_rebuild_graph([str(populated_data_dir)])
        regen_cit = populated_data_dir / "citation_network.regen.graphml"
        g = ig.Graph.Read_GraphML(str(regen_cit))
        edge_attrs = set(g.es.attributes())
        assert {"semantic_similarity", "weight", "ref_similarity", "cit_similarity"}.issubset(edge_attrs)
