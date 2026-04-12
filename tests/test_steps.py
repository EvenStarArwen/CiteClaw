"""Tests for :mod:`citeclaw.steps` — every pipeline step.

All S2 calls are routed through :class:`tests.fakes.FakeS2Client` so nothing
touches the network. LLM calls go through the built-in stub.
"""

from __future__ import annotations

import pytest

from citeclaw.config import SeedPaper
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.builder import build_blocks
from citeclaw.models import PaperRecord
from citeclaw.steps import build_step
from citeclaw.steps.base import StepResult
from citeclaw.steps.expand_backward import ExpandBackward
from citeclaw.steps.expand_forward import ExpandForward
from citeclaw.steps.finalize import Finalize
from citeclaw.steps.load_seeds import LoadSeeds
from citeclaw.steps.parallel import Parallel
from citeclaw.steps.rerank import Rerank
from citeclaw.steps.rescreen import ReScreen
from citeclaw.steps.shape_log import ShapeLog


# ---------------------------------------------------------------------------
# Build step registry
# ---------------------------------------------------------------------------


class TestBuildStep:
    def test_load_seeds(self):
        s = build_step({"step": "LoadSeeds"})
        assert isinstance(s, LoadSeeds)

    def test_merge_duplicates(self):
        from citeclaw.steps.merge_duplicates import MergeDuplicates
        s = build_step({
            "step": "MergeDuplicates",
            "title_threshold": 0.9,
            "semantic_threshold": 0.97,
            "year_window": 2,
            "use_embeddings": False,
        })
        assert isinstance(s, MergeDuplicates)
        assert s.title_threshold == 0.9
        assert s.semantic_threshold == 0.97
        assert s.year_window == 2
        assert s.use_embeddings is False

    def test_expand_forward_inline_screener(self):
        s = build_step(
            {"step": "ExpandForward", "max_citations": 10,
             "screener": {"type": "YearFilter", "min": 2020}},
        )
        assert isinstance(s, ExpandForward)
        assert s.max_citations == 10
        assert s.screener is not None

    def test_expand_forward_by_name(self):
        blocks = build_blocks({"y": {"type": "YearFilter", "min": 2020}})
        s = build_step({"step": "ExpandForward", "screener": "y"}, blocks)
        assert isinstance(s, ExpandForward)

    def test_expand_backward(self):
        s = build_step({"step": "ExpandBackward"})
        assert isinstance(s, ExpandBackward)
        assert s.screener is None

    def test_rerank(self):
        s = build_step({"step": "Rerank", "metric": "citation", "k": 5})
        assert isinstance(s, Rerank)
        assert s.metric == "citation"
        assert s.k == 5

    def test_rescreen(self):
        s = build_step({"step": "ReScreen", "screener": {"type": "YearFilter", "min": 2020}})
        assert isinstance(s, ReScreen)

    def test_finalize(self):
        s = build_step({"step": "Finalize"})
        assert isinstance(s, Finalize)

    def test_parallel(self):
        spec = {
            "step": "Parallel",
            "branches": [
                [{"step": "Rerank", "metric": "citation", "k": 3}],
                [{"step": "Rerank", "metric": "citation", "k": 5}],
            ],
        }
        s = build_step(spec)
        assert isinstance(s, Parallel)
        assert len(s.branches) == 2

    def test_unknown_step(self):
        with pytest.raises(ValueError, match="Unknown step"):
            build_step({"step": "MysteryStep"})

    def test_missing_screener_ref(self):
        with pytest.raises(KeyError):
            build_step({"step": "ExpandForward", "screener": "nope"}, {})


# ---------------------------------------------------------------------------
# LoadSeeds
# ---------------------------------------------------------------------------


class TestLoadSeeds:
    def test_loads_seeds_from_config(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED"), SeedPaper(paper_id="CITER1")]
        step = LoadSeeds()
        result = step.run([], ctx)
        assert isinstance(result, StepResult)
        assert len(result.signal) == 2
        assert "SEED" in ctx.collection
        assert "CITER1" in ctx.collection
        # Seed identity / verdict
        for rec in result.signal:
            assert rec.depth == 0
            assert rec.source == "seed"
            assert rec.llm_verdict == "accept_seed"
            assert rec.paper_id in ctx.seed_ids
            assert rec.paper_id in ctx.seen
        assert set(ctx.new_seed_ids) == {"SEED", "CITER1"}

    def test_missing_paper_is_logged_and_skipped(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED"), SeedPaper(paper_id="NOT_REAL")]
        step = LoadSeeds()
        result = step.run([], ctx)
        assert len(result.signal) == 1
        assert "SEED" in ctx.collection

    def test_empty_paper_id_skipped(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="   "), SeedPaper(paper_id="SEED")]
        step = LoadSeeds()
        result = step.run([], ctx)
        assert len(result.signal) == 1

    def test_duplicate_seed_skipped(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED")]
        step = LoadSeeds()
        step.run([], ctx)
        # Run again — the second load should be a no-op.
        result = step.run([], ctx)
        assert len(result.signal) == 0

    def test_title_and_abstract_override_when_missing(self, ctx):
        """If the seed spec provides title/abstract, they should fill in
        when the fetched record has empty strings."""
        # Wipe out title/abstract in the fake corpus for SEED
        ctx.s2._papers["SEED"]["title"] = ""
        ctx.s2._papers["SEED"]["abstract"] = None
        ctx.config.seed_papers = [
            SeedPaper(paper_id="SEED", title="User Title", abstract="User Abs"),
        ]
        step = LoadSeeds()
        step.run([], ctx)
        rec = ctx.collection["SEED"]
        assert rec.title == "User Title"
        assert rec.abstract == "User Abs"


# ---------------------------------------------------------------------------
# ExpandForward
# ---------------------------------------------------------------------------


class TestExpandForward:
    def _ctx_with_seed(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED")]
        LoadSeeds().run([], ctx)
        return list(ctx.collection.values())

    def test_no_screener_accepts_all(self, ctx):
        seeds = self._ctx_with_seed(ctx)
        step = ExpandForward(max_citations=10, screener=None)
        result = step.run(seeds, ctx)
        # Without a screener every candidate passes through unfiltered.
        assert len(result.signal) >= 1

    def test_expand_forward_with_stub_llm(self, ctx):
        seeds = self._ctx_with_seed(ctx)
        # Very permissive screener: just year >=2019 → CITER1 and CITER2 pass
        screener = YearFilter(min=2019)
        step = ExpandForward(max_citations=10, screener=screener)
        result = step.run(seeds, ctx)
        assert len(result.signal) >= 1
        # SEED's citers in the fake corpus are CITER1 and CITER2
        accepted_ids = {p.paper_id for p in result.signal}
        assert "CITER1" in accepted_ids
        assert "CITER2" in accepted_ids
        # They should be recorded in the context
        for pid in accepted_ids:
            assert pid in ctx.collection
        assert "SEED" in ctx.expanded_forward

    def test_expansion_deduped_across_calls(self, ctx):
        seeds = self._ctx_with_seed(ctx)
        step = ExpandForward(max_citations=10, screener=YearFilter(min=2019))
        step.run(seeds, ctx)
        # Running the same step again should NOT re-expand SEED.
        before_calls = ctx.s2.calls.get("fetch_citation_ids_and_counts", 0)
        step.run(seeds, ctx)
        after_calls = ctx.s2.calls.get("fetch_citation_ids_and_counts", 0)
        assert after_calls == before_calls

    def test_s2_failure_is_logged_not_raised(self, ctx, monkeypatch):
        seeds = self._ctx_with_seed(ctx)

        def boom(*a, **kw):
            raise RuntimeError("down")

        monkeypatch.setattr(ctx.s2, "fetch_citation_ids_and_counts", boom)
        step = ExpandForward(max_citations=10, screener=YearFilter(min=2019))
        result = step.run(seeds, ctx)
        assert result.signal == []


# ---------------------------------------------------------------------------
# ExpandBackward
# ---------------------------------------------------------------------------


class TestExpandBackward:
    def test_expand_backward(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED")]
        seeds = LoadSeeds().run([], ctx).signal
        screener = YearFilter(min=2010)
        step = ExpandBackward(screener=screener)
        result = step.run(seeds, ctx)
        # SEED references REF1 and REF2 — both should be accepted.
        accepted = {p.paper_id for p in result.signal}
        assert {"REF1", "REF2"}.issubset(accepted)
        assert "SEED" in ctx.expanded_backward

    def test_no_screener_accepts_all(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED")]
        seeds = LoadSeeds().run([], ctx).signal
        step = ExpandBackward(screener=None)
        result = step.run(seeds, ctx)
        # Without a screener every reference passes through unfiltered.
        assert len(result.signal) >= 1

    def test_screener_can_reject(self, ctx):
        ctx.config.seed_papers = [SeedPaper(paper_id="SEED")]
        seeds = LoadSeeds().run([], ctx).signal
        # REF1 is year 2015, REF2 is year 2016 — filter min=2020 drops both.
        step = ExpandBackward(screener=YearFilter(min=2020))
        result = step.run(seeds, ctx)
        assert result.signal == []


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------


class TestRerankStep:
    def test_empty_signal(self, ctx):
        step = Rerank(metric="citation", k=5)
        r = step.run([], ctx)
        assert r.signal == []
        assert r.stats["kept"] == 0

    def test_citation_top_k(self, ctx):
        signal = [
            PaperRecord(paper_id="A", citation_count=10),
            PaperRecord(paper_id="B", citation_count=50),
            PaperRecord(paper_id="C", citation_count=30),
        ]
        step = Rerank(metric="citation", k=2)
        r = step.run(signal, ctx)
        out_ids = [p.paper_id for p in r.signal]
        assert out_ids == ["B", "C"]
        assert r.stats["kept"] == 2

    def test_non_destructive_to_collection(self, ctx):
        """Rerank must never remove items from ``ctx.collection``."""
        signal = [PaperRecord(paper_id="A", citation_count=10)]
        ctx.collection = {p.paper_id: p for p in signal}
        ctx.collection["B"] = PaperRecord(paper_id="B", citation_count=5)
        before = dict(ctx.collection)
        Rerank(metric="citation", k=1).run(signal, ctx)
        assert ctx.collection == before

    def test_div_label_default_off(self, ctx):
        step = Rerank(metric="citation", k=5, diversity=None)
        assert step._div_label() == "off"

    def test_div_label_string(self, ctx):
        step = Rerank(metric="citation", k=5, diversity="walktrap")
        assert step._div_label() == "walktrap"

    def test_div_label_dict(self, ctx):
        step = Rerank(metric="citation", k=5, diversity={"type": "louvain"})
        assert step._div_label() == "louvain"

    def test_div_label_cluster_reference(self, ctx):
        step = Rerank(metric="citation", k=5, diversity={"cluster": "forward_topics"})
        assert step._div_label() == "ref:forward_topics"


# ---------------------------------------------------------------------------
# ReScreen (destructive)
# ---------------------------------------------------------------------------


class TestReScreen:
    def test_no_screener_is_noop(self, ctx):
        ctx.collection = {"A": PaperRecord(paper_id="A", source="forward")}
        step = ReScreen(screener=None)
        r = step.run([], ctx)
        assert r.stats["removed"] == 0

    def test_rescreen_drops_rejected_from_collection(self, ctx):
        ctx.collection = {
            "A": PaperRecord(paper_id="A", year=2025, source="forward"),
            "B": PaperRecord(paper_id="B", year=2000, source="forward"),
            "SEED": PaperRecord(paper_id="SEED", year=2000, source="seed"),
        }
        step = ReScreen(screener=YearFilter(min=2020))
        r = step.run([], ctx)
        # B dropped, A kept, SEED is protected (source=seed)
        assert "A" in ctx.collection
        assert "B" not in ctx.collection
        assert "SEED" in ctx.collection
        assert r.stats["removed"] == 1
        # Rejection counts get a 'rescreen_' prefix
        assert any(k.startswith("rescreen_") for k in ctx.rejection_counts)

    def test_signal_is_filtered_to_survivors(self, ctx):
        """ReScreen must also drop rejected papers from the ``signal`` so
        downstream steps don't see them."""
        ctx.collection = {
            "A": PaperRecord(paper_id="A", year=2025, source="forward"),
            "B": PaperRecord(paper_id="B", year=2000, source="forward"),
        }
        signal = [ctx.collection["A"], ctx.collection["B"]]
        step = ReScreen(screener=YearFilter(min=2020))
        r = step.run(signal, ctx)
        assert [p.paper_id for p in r.signal] == ["A"]


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


class _EchoStep:
    def __init__(self, name="echo", *, add_id: str | None = None, keep: int | None = None):
        self.name = name
        self.add_id = add_id
        self.keep = keep

    def run(self, signal, ctx):
        out = list(signal) if self.keep is None else signal[: self.keep]
        if self.add_id:
            out.append(PaperRecord(paper_id=self.add_id))
        return StepResult(signal=out, in_count=len(signal), stats={})


class TestParallel:
    def test_union_of_branches(self, ctx):
        signal = [PaperRecord(paper_id="A"), PaperRecord(paper_id="B")]
        step = Parallel(branches=[
            [_EchoStep("b1", add_id="X")],
            [_EchoStep("b2", add_id="Y")],
        ])
        r = step.run(signal, ctx)
        out_ids = {p.paper_id for p in r.signal}
        assert out_ids == {"A", "B", "X", "Y"}
        assert r.stats["merged"] == 4
        assert r.stats["branches"][0]["branch"] == 0
        assert r.stats["branches"][1]["branch"] == 1

    def test_dedup_on_merge(self, ctx):
        """If two branches emit the same paper, it should appear once."""
        signal = [PaperRecord(paper_id="A")]
        step = Parallel(branches=[
            [_EchoStep("b1")],
            [_EchoStep("b2")],
        ])
        r = step.run(signal, ctx)
        assert len(r.signal) == 1

    def test_branches_get_independent_snapshot(self, ctx):
        """Each branch should see the original signal, not a mutated version."""
        signal = [PaperRecord(paper_id="A"), PaperRecord(paper_id="B")]
        step = Parallel(branches=[
            [_EchoStep("drop", keep=1)],
            [_EchoStep("keep_all")],
        ])
        r = step.run(signal, ctx)
        # Branch 1 keeps all, so B should appear.
        assert "B" in {p.paper_id for p in r.signal}


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_writes_all_artifacts(self, ctx):
        # Seed the collection
        ctx.collection = {
            "A": PaperRecord(
                paper_id="A", title="Paper A", year=2020, citation_count=10,
                authors=[{"authorId": "au1", "name": "Alice"}],
            ),
            "B": PaperRecord(
                paper_id="B", title="Paper B", year=2021, citation_count=20,
                references=["A"],
                authors=[{"authorId": "au2", "name": "Bob"}],
            ),
        }
        ctx.seen = {"A", "B"}
        ctx.iteration = 1
        step = Finalize()
        result = step.run([], ctx)
        assert result.stats["wrote"] == 2
        data_dir = ctx.config.data_dir
        assert (data_dir / "literature_collection.json").exists()
        assert (data_dir / "literature_collection.bib").exists()
        assert (data_dir / "run_state.json").exists()
        assert (data_dir / "citation_network.graphml").exists()
        assert (data_dir / "collaboration_network.graphml").exists()
        # Finalize must have prefetched embeddings so semantic_similarity can
        # be populated on the output graph.
        assert ctx.s2.calls.get("fetch_embeddings_batch", 0) >= 1

    def test_iteration_suffix_added_when_gt1(self, ctx):
        ctx.collection = {
            "A": PaperRecord(paper_id="A", title="A", year=2020, citation_count=1),
            "B": PaperRecord(paper_id="B", title="B", year=2021, citation_count=2),
        }
        ctx.iteration = 3
        step = Finalize()
        step.run([], ctx)
        data_dir = ctx.config.data_dir
        assert (data_dir / "literature_collection.exp3.json").exists()
        assert (data_dir / "run_state.exp3.json").exists()

    def test_too_few_papers_skips_graph(self, ctx):
        ctx.collection = {"A": PaperRecord(paper_id="A", title="A", year=2020)}
        ctx.iteration = 1
        Finalize().run([], ctx)
        data_dir = ctx.config.data_dir
        # Graph and collab graph should NOT exist because len(collection) < 2.
        assert not (data_dir / "citation_network.graphml").exists()
        assert not (data_dir / "collaboration_network.graphml").exists()


# ---------------------------------------------------------------------------
# ShapeLog
# ---------------------------------------------------------------------------


class TestShapeLog:
    def test_render_header(self):
        log = ShapeLog()
        log.record("LoadSeeds", 0, 3, 3, {"loaded": 3})
        out = log.render()
        assert "Step" in out
        assert "In" in out
        assert "Out" in out
        assert "LoadSeeds" in out
        assert "loaded=3" in out

    def test_render_with_branches(self):
        log = ShapeLog()
        branches = [
            {"branch": 0, "steps": [{"name": "Rerank", "in": 5, "out": 3}]},
            {"branch": 1, "steps": [{"name": "ExpandForward", "in": 5, "out": 2}]},
        ]
        log.record("Parallel", 5, 5, 0, {"branches": branches})
        out = log.render()
        assert "Parallel" in out
        assert "branches=2" in out
        assert "branch[0]" in out
        assert "Rerank" in out


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_missing_dir_raises(self, ctx, tmp_path):
        from citeclaw.steps.checkpoint import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint(ctx, tmp_path / "nope")

    def test_missing_files_raise(self, ctx, tmp_path):
        from citeclaw.steps.checkpoint import load_checkpoint
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(ctx, empty)

    def test_roundtrip_via_finalize(self, ctx):
        """Finalize → load_checkpoint should produce an identical collection."""
        from citeclaw.steps.checkpoint import load_checkpoint

        ctx.collection = {
            "A": PaperRecord(paper_id="A", title="A", year=2020, citation_count=10,
                             source="seed"),
            "B": PaperRecord(paper_id="B", title="B", year=2021, citation_count=20,
                             references=["A"], source="backward"),
        }
        ctx.seed_ids = {"A"}
        ctx.seen = {"A", "B"}
        ctx.iteration = 1
        Finalize().run([], ctx)
        data_dir = ctx.config.data_dir
        # Build a fresh context and load the checkpoint.
        fresh = type(ctx)(
            config=ctx.config, s2=ctx.s2, cache=ctx.cache, budget=ctx.budget,
        )
        load_checkpoint(fresh, data_dir)
        assert set(fresh.collection) == {"A", "B"}
        assert "A" in fresh.seed_ids
        assert fresh.iteration == 2
        # expanded_* should be populated for loaded papers
        assert "A" in fresh.expanded_forward
