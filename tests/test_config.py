"""Tests for :mod:`citeclaw.config` — Settings, YAML loading, BudgetTracker."""

from __future__ import annotations


import pytest
import yaml

from citeclaw.config import BudgetTracker, SeedPaper, Settings, load_settings


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self, tmp_path):
        s = Settings(data_dir=tmp_path)
        assert s.screening_model == "stub"
        assert s.search_model == ""
        assert s.seed_papers == []
        assert s.pipeline == []
        assert s.blocks == {}
        assert s.blocks_built == {}
        assert s.pipeline_built == []

    def test_search_model_explicit_override(self, tmp_path):
        """PC-06: search_model can be set independently of screening_model
        so the iterative search agent uses a more capable model than the
        per-paper screener."""
        s = Settings(
            data_dir=tmp_path,
            screening_model="gemini-3-flash-lite",
            search_model="gemini-3-pro",
        )
        assert s.screening_model == "gemini-3-flash-lite"
        assert s.search_model == "gemini-3-pro"

    def test_search_model_empty_default_signals_fallback(self, tmp_path):
        """An empty search_model is the documented signal for callers to
        fall back to screening_model. ExpandBySearch consumes this via
        ``self.agent.model or ctx.config.search_model or ctx.config.screening_model``."""
        s = Settings(data_dir=tmp_path, screening_model="gpt-4o-mini")
        assert s.search_model == ""
        # The cascade callers use to resolve the effective model.
        effective = s.search_model or s.screening_model
        assert effective == "gpt-4o-mini"

    def test_search_model_loaded_from_yaml(self, tmp_path):
        """search_model round-trips through load_settings just like any
        other top-level YAML field."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "screening_model: gpt-4o-mini\nsearch_model: gemini-3-pro\n"
        )
        s = load_settings(cfg_path)
        assert s.screening_model == "gpt-4o-mini"
        assert s.search_model == "gemini-3-pro"

    def test_seed_schema_accepts_title_only_entry(self, tmp_path):
        """PC-04 + PC-06: SeedPaper.paper_id is optional so YAML callers
        can write ``{title: ...}``-only entries that ResolveSeeds will
        resolve via search_match. Verify the parsing path."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "seed_papers:\n"
            "  - title: 'Highly Accurate Protein Structure Prediction'\n"
            "  - paper_id: 'DOI:10.1/abc'\n"
            "  - paper_id: 'DOI:10.2/def'\n"
            "    title: 'Both fields set'\n"
        )
        s = load_settings(cfg_path)
        assert len(s.seed_papers) == 3
        assert s.seed_papers[0].paper_id == ""
        assert s.seed_papers[0].title == "Highly Accurate Protein Structure Prediction"
        assert s.seed_papers[1].paper_id == "DOI:10.1/abc"
        assert s.seed_papers[1].title == ""
        assert s.seed_papers[2].paper_id == "DOI:10.2/def"
        assert s.seed_papers[2].title == "Both fields set"

    def test_blocks_are_built_lazily(self, tmp_path):
        s = Settings(
            data_dir=tmp_path,
            blocks={"y": {"type": "YearFilter", "min": 2020, "max": 2024}},
        )
        assert "y" in s.blocks_built
        filt = s.blocks_built["y"]
        assert filt.name == "y"

    def test_pipeline_is_built_with_block_references(self, tmp_path):
        s = Settings(
            data_dir=tmp_path,
            blocks={"y": {"type": "YearFilter", "min": 2020, "max": 2024}},
            pipeline=[
                {"step": "LoadSeeds"},
                {"step": "Rerank", "metric": "citation", "k": 5},
            ],
        )
        assert len(s.pipeline_built) == 2
        assert s.pipeline_built[0].name == "LoadSeeds"
        assert s.pipeline_built[1].name == "Rerank"


# ---------------------------------------------------------------------------
# load_settings — YAML loading
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_no_file(self, tmp_path):
        s = load_settings(None)
        assert isinstance(s, Settings)
        assert s.screening_model == "stub"

    def test_missing_file_is_tolerated(self, tmp_path):
        s = load_settings(tmp_path / "nope.yaml")
        assert isinstance(s, Settings)

    def test_yaml_loading(self, tmp_path):
        cfg = {
            "screening_model": "stub",
            "data_dir": str(tmp_path / "out"),
            "topic_description": "Plant science",
            "seed_papers": [{"paper_id": "DOI:10.1/abc", "title": "Seed 1"}],
            "blocks": {"y": {"type": "YearFilter", "min": 2018}},
            "pipeline": [{"step": "LoadSeeds"}],
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        s = load_settings(cfg_path)
        assert s.topic_description == "Plant science"
        assert len(s.seed_papers) == 1
        assert s.seed_papers[0].paper_id == "DOI:10.1/abc"
        assert len(s.pipeline_built) == 1
        assert "y" in s.blocks_built

    def test_overrides_dict_wins(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump({"screening_model": "from_yaml", "data_dir": str(tmp_path)})
        )
        s = load_settings(cfg_path, overrides={"screening_model": "stub"})
        assert s.screening_model == "stub"

    def test_env_overrides_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        s = load_settings(None, overrides={"data_dir": str(tmp_path)})
        assert s.openai_api_key == "env-key"

    def test_yaml_must_be_mapping(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("- one\n- two\n")
        with pytest.raises(ValueError):
            load_settings(cfg_path)

    def test_yaml_rejects_api_keys(self, tmp_path):
        """API keys in YAML must be rejected — they belong in env vars only."""
        for key_name in (
            "openai_api_key",
            "gemini_api_key",
            "s2_api_key",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY",
        ):
            cfg_path = tmp_path / "cfg.yaml"
            cfg_path.write_text(
                yaml.safe_dump({key_name: "leaked", "data_dir": str(tmp_path)})
            )
            with pytest.raises(ValueError, match="API keys must not be set"):
                load_settings(cfg_path)


class TestSeedPaper:
    def test_defaults(self):
        sp = SeedPaper(paper_id="x")
        assert sp.paper_id == "x"
        assert sp.title == ""
        assert sp.abstract is None


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_empty(self):
        b = BudgetTracker()
        assert b.llm_total_tokens == 0
        assert b.llm_calls == 0
        assert b.s2_requests == 0
        assert b.exhausted() is False

    def test_record_llm_accumulates(self):
        b = BudgetTracker()
        b.record_llm(100, 50, category="title")
        b.record_llm(200, 100, category="title")
        b.record_llm(10, 5, category="abstract", reasoning_tokens=3)
        assert b.llm_total_tokens == 100 + 50 + 200 + 100 + 10 + 5
        assert b.llm_calls == 3
        assert b.llm_reasoning_tokens == 3

    def test_record_s2(self):
        b = BudgetTracker()
        b.record_s2("metadata")
        b.record_s2("references")
        b.record_s2("references", cached=True)
        assert b.s2_requests == 2  # cached doesn't count toward api bucket

    def test_is_exhausted_on_tokens(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, max_llm_tokens=100, max_s2_requests=1000)
        b = BudgetTracker()
        assert b.is_exhausted(cfg) is False
        b.record_llm(60, 60)  # 120 > 100
        assert b.is_exhausted(cfg) is True

    def test_is_exhausted_on_s2(self, tmp_path):
        cfg = Settings(data_dir=tmp_path, max_llm_tokens=10_000_000, max_s2_requests=2)
        b = BudgetTracker()
        b.record_s2("metadata")
        b.record_s2("metadata")
        assert b.is_exhausted(cfg) is True

    def test_summary_strings(self):
        b = BudgetTracker()
        b.record_llm(1000, 500, "title")
        b.record_s2("metadata")
        assert "LLM" in b.summary()
        assert "S2" in b.summary()
        detailed = b.detailed_summary()
        assert "Budget breakdown" in detailed
        assert "title" in detailed
        assert "metadata" in detailed

    def test_to_dict_shape(self):
        b = BudgetTracker()
        b.record_llm(10, 5, "title", reasoning_tokens=2)
        b.record_s2("metadata")
        d = b.to_dict()
        assert d["llm"]["total_tokens"] == 15
        assert d["llm"]["total_calls"] == 1
        assert d["llm"]["by_category"]["title"]["tokens"] == 15
        assert d["llm"]["by_category"]["title"]["reasoning_tokens"] == 2
        assert d["s2"]["total_api_requests"] == 1
