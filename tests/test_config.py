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
        assert s.seed_papers == []
        assert s.pipeline == []
        assert s.blocks == {}
        assert s.blocks_built == {}
        assert s.pipeline_built == []

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
