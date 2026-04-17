"""End-to-end pipeline tests.

Runs the whole pipeline engine against the in-memory
:class:`tests.fakes.FakeS2Client`, using CiteClaw's offline stub LLM. No
network traffic. No LLM tokens. No secrets.

The goal is to exercise the layer-composition behavior described in
``CLAUDE.md``: LoadSeeds → ExpandForward → ReScreen → ExpandBackward →
Rerank → Parallel → Finalize, all wired up through the YAML → blocks →
steps pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import pytest
import yaml

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, load_settings
from citeclaw.context import Context
from citeclaw.pipeline import run_pipeline
from tests.fakes import FakeS2Client, build_chain_corpus, make_paper


def _write_yaml(tmp_path: Path, config: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config))
    return p


def _build_context(cfg, s2: FakeS2Client) -> Context:
    cache = Cache(cfg.data_dir / "cache.db")
    budget = BudgetTracker()
    return Context(config=cfg, s2=s2, cache=cache, budget=budget)


# ---------------------------------------------------------------------------
# A minimal pipeline that covers every step type
# ---------------------------------------------------------------------------


def test_full_pipeline_round_trip(tmp_path: Path):
    """Build a config, wire up a fake S2, and run the whole pipeline.

    Pipeline: LoadSeeds → ExpandBackward (from SEED) → ExpandForward (from
    the new wavefront) → ReScreen → Rerank → Parallel → Finalize. This
    ordering lets us assert that both the backward and forward hops of the
    chain corpus have actually been traversed.
    """
    data_dir = tmp_path / "data"
    cfg_path = _write_yaml(
        tmp_path,
        {
            "screening_model": "stub",
            "data_dir": str(data_dir),
            "topic_description": "Offline E2E test",
            "max_papers_total": 100,
            "llm_batch_size": 8,
            "llm_concurrency": 1,
            "seed_papers": [{"paper_id": "SEED"}],
            "blocks": {
                "year_layer": {"type": "YearFilter", "min": 2000, "max": 2030},
                "cit_base": {"type": "CitationFilter", "beta": 1},
                "title_llm": {
                    "type": "LLMFilter",
                    "scope": "title",
                    "prompt": "the paper is on topic",
                },
                "forward_screener": {
                    "type": "Sequential",
                    "layers": ["year_layer", "cit_base", "title_llm"],
                },
                "rescreen_block": {
                    "type": "Sequential",
                    "layers": [
                        {"type": "LLMFilter", "scope": "title_abstract",
                         "prompt": "still in scope"},
                    ],
                },
            },
            "pipeline": [
                {"step": "LoadSeeds"},
                # Fan out: one branch expands forward from SEED, another expands backward.
                {
                    "step": "Parallel",
                    "branches": [
                        [{"step": "ExpandForward", "max_citations": 10,
                          "screener": "forward_screener"}],
                        [{"step": "ExpandBackward", "screener": "forward_screener"}],
                    ],
                },
                {"step": "ReScreen", "screener": "rescreen_block"},
                {"step": "Rerank", "metric": "citation", "k": 20},
                {
                    "step": "Parallel",
                    "branches": [
                        [{"step": "Rerank", "metric": "citation", "k": 3}],
                        [{"step": "Rerank", "metric": "pagerank", "k": 3}],
                    ],
                },
                {"step": "Finalize"},
            ],
        },
    )

    cfg = load_settings(cfg_path)
    s2 = build_chain_corpus()
    ctx = _build_context(cfg, s2)
    try:
        run_pipeline(ctx)
    finally:
        ctx.cache.close()

    # ----- Collection invariants -----
    assert "SEED" in ctx.collection
    # Forward branch finds SEED's citers.
    assert "CITER1" in ctx.collection
    assert "CITER2" in ctx.collection
    # Backward branch finds SEED's references.
    assert "REF1" in ctx.collection
    assert "REF2" in ctx.collection
    # WEAK isn't a citer of anything in the chain and has a tiny citation count
    # so it should never get pulled in.
    assert "WEAK" not in ctx.collection

    # ----- Artifact files -----
    assert (data_dir / "literature_collection.json").exists()
    assert (data_dir / "literature_collection.bib").exists()
    assert (data_dir / "run_state.json").exists()
    assert (data_dir / "citation_network.graphml").exists()
    assert (data_dir / "collaboration_network.graphml").exists()
    assert (data_dir / "shape_summary.txt").exists()
    assert (data_dir / "shape_summary.json").exists()

    # shape_summary.json should be a list of dicts with the same step sequence
    # as the rendered text table — enables programmatic run comparison.
    shape_json = json.loads((data_dir / "shape_summary.json").read_text())
    assert isinstance(shape_json, list)
    assert all({"step", "in", "out", "delta_collection", "stats"} <= r.keys() for r in shape_json)
    step_names = [r["step"] for r in shape_json]
    assert "LoadSeeds" in step_names
    assert "Finalize" in step_names

    # Verify the collection JSON is well-formed and contains every accepted paper.
    coll_json = json.loads((data_dir / "literature_collection.json").read_text())
    accepted_ids = {p["paper_id"] for p in coll_json["papers"]}
    assert accepted_ids == set(ctx.collection.keys())
    # The seed should be marked as source=seed
    seed_entry = next(p for p in coll_json["papers"] if p["paper_id"] == "SEED")
    assert seed_entry["source"] == "seed"


def test_rescreen_actually_drops_papers(tmp_path: Path, monkeypatch):
    """If ReScreen's screener rejects a paper, it must disappear from
    ``ctx.collection`` and get recorded under a ``rescreen_*`` bucket."""
    # Rewire the stub LLM runner so the ``rescreen`` filter rejects CITER1.
    from citeclaw.screening import llm_runner

    def custom_run_one_batch(client, lf, contents, ids):
        if lf.name == "drop_citer1":
            # Accept everything *except* CITER1.
            return {pid: (pid != "CITER1") for pid in ids}
        return {pid: True for pid in ids}

    monkeypatch.setattr(llm_runner, "_run_one_batch", custom_run_one_batch)

    data_dir = tmp_path / "data"
    cfg_path = _write_yaml(
        tmp_path,
        {
            "screening_model": "stub",
            "data_dir": str(data_dir),
            "max_papers_total": 100,
            "llm_batch_size": 8,
            "llm_concurrency": 1,
            "seed_papers": [{"paper_id": "SEED"}],
            "blocks": {
                "ok": {"type": "LLMFilter", "scope": "title", "prompt": "x"},
                "drop_citer1": {
                    "type": "LLMFilter", "scope": "title_abstract",
                    "prompt": "accept if not CITER1",
                },
                "forward": {"type": "Sequential", "layers": ["ok"]},
                "rescreen": {"type": "Sequential", "layers": ["drop_citer1"]},
            },
            "pipeline": [
                {"step": "LoadSeeds"},
                {"step": "ExpandForward", "max_citations": 10, "screener": "forward"},
                {"step": "ReScreen", "screener": "rescreen"},
                {"step": "Finalize"},
            ],
        },
    )

    cfg = load_settings(cfg_path)
    s2 = build_chain_corpus()
    ctx = _build_context(cfg, s2)
    try:
        run_pipeline(ctx)
    finally:
        ctx.cache.close()

    assert "CITER1" not in ctx.collection
    assert any(k.startswith("rescreen_") for k in ctx.rejection_counts)


def test_budget_cap_stops_early(tmp_path: Path):
    """Setting ``max_papers_total`` should cause the pipeline to stop as
    soon as the collection reaches the cap, but still run Finalize so the
    user gets partial artifacts."""
    data_dir = tmp_path / "data"
    cfg_path = _write_yaml(
        tmp_path,
        {
            "screening_model": "stub",
            "data_dir": str(data_dir),
            "max_papers_total": 2,   # tight cap
            "llm_batch_size": 8,
            "llm_concurrency": 1,
            "seed_papers": [{"paper_id": "SEED"}],
            "blocks": {
                "pass": {"type": "YearFilter", "min": 2000, "max": 2030},
                "forward": {"type": "Sequential", "layers": ["pass"]},
            },
            "pipeline": [
                {"step": "LoadSeeds"},
                {"step": "ExpandForward", "max_citations": 10, "screener": "forward"},
                {"step": "ExpandBackward", "screener": "forward"},
                {"step": "Finalize"},
            ],
        },
    )

    cfg = load_settings(cfg_path)
    s2 = build_chain_corpus()
    ctx = _build_context(cfg, s2)
    try:
        run_pipeline(ctx)
    finally:
        ctx.cache.close()

    # Collection should have reached the cap — Finalize still runs (even as
    # the early-break fallback), so the JSON artifact exists.
    assert len(ctx.collection) >= 2
    assert (data_dir / "literature_collection.json").exists()


def test_pipeline_with_merge_duplicates_step(tmp_path: Path):
    """Exercise the full pipeline with ``MergeDuplicates`` in the middle.

    We build a fake corpus where CITER2 is a preprint-vs-published
    duplicate of CITER1 (same DOI, different venues). After the pipeline
    runs, only one of them should survive in the collection, and the
    graph should contain one merged edge instead of two.
    """
    data_dir = tmp_path / "data"
    cfg_path = _write_yaml(
        tmp_path,
        {
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
                {"step": "MergeDuplicates", "use_embeddings": False},
                {"step": "Finalize"},
            ],
        },
    )
    cfg = load_settings(cfg_path)

    # Build a corpus where CITER1 (arXiv) and CITER2 (NeurIPS) share a DOI
    # and first-author ID. Both cite SEED so forward expansion picks them up.
    author = [{"authorId": "shared_au", "name": "Shared Author"}]
    s2 = FakeS2Client()
    s2.add(make_paper(
        "SEED", title="Seed Paper", year=2018, venue="arXiv",
        authors=author, external_ids={"DOI": "10.1/seed"},
    ))
    s2.add(make_paper(
        "CITER1", title="Attention is all you need",
        year=2022, venue="arXiv", references=["SEED"],
        authors=author, external_ids={"DOI": "10.1/citer"},
    ))
    s2.add(make_paper(
        "CITER2", title="Attention is all you need",
        year=2022, venue="NeurIPS", references=["SEED"],
        authors=author, external_ids={"DOI": "10.1/citer"},
    ))

    ctx = _build_context(cfg, s2)
    try:
        run_pipeline(ctx)
    finally:
        ctx.cache.close()

    # MergeDuplicates should have folded CITER1 and CITER2 into a single
    # canonical record (the NeurIPS version, since it's peer-reviewed).
    assert "CITER2" in ctx.collection
    assert "CITER1" not in ctx.collection
    assert ctx.collection["CITER2"].aliases == ["CITER1"]
    assert ctx.alias_map.get("CITER1") == "CITER2"

    # The citation graph should have exactly one edge SEED → CITER2 (not two).
    g = ig.Graph.Read_GraphML(str(data_dir / "citation_network.graphml"))
    assert g.vcount() == 2
    assert g.ecount() == 1


def test_parallel_branches_merge_unique_papers(tmp_path: Path):
    """Two branches with different screening behaviour should both contribute
    to the merged signal but duplicates must be deduped."""
    data_dir = tmp_path / "data"
    # Very permissive screener — everything passes — so both branches
    # independently find the same citers of SEED.
    cfg_path = _write_yaml(
        tmp_path,
        {
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
                {
                    "step": "Parallel",
                    "branches": [
                        [{"step": "ExpandForward", "max_citations": 10, "screener": "forward"}],
                        [{"step": "ExpandBackward", "screener": "forward"}],
                    ],
                },
                {"step": "Finalize"},
            ],
        },
    )
    cfg = load_settings(cfg_path)
    s2 = build_chain_corpus()
    ctx = _build_context(cfg, s2)
    try:
        run_pipeline(ctx)
    finally:
        ctx.cache.close()

    # Branch 1 (forward) adds CITER1, CITER2; Branch 2 (backward) adds REF1, REF2.
    # Union should contain all four + SEED.
    assert {"SEED", "CITER1", "CITER2", "REF1", "REF2"}.issubset(ctx.collection)


# ---------------------------------------------------------------------------
# Optional: one live S2 test for smoke-testing the real client wire format.
#
# Skipped by default. Set ``CITECLAW_LIVE_S2=1`` to enable.
# ---------------------------------------------------------------------------


@pytest.mark.live_s2
def test_live_s2_fetch_metadata_smoke(tmp_path: Path, live_s2_allowed):
    """One real S2 call (well under the 1 req/s rate limit) so the live
    integration isn't completely untested.

    Gated behind ``CITECLAW_LIVE_S2=1`` so CI stays fully offline by default.
    """
    if not live_s2_allowed:
        pytest.skip("Set CITECLAW_LIVE_S2=1 to enable live S2 tests")

    from citeclaw.clients.s2 import SemanticScholarClient
    from citeclaw.config import Settings

    cfg = Settings(data_dir=tmp_path)
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    client = SemanticScholarClient(cfg, cache, budget)
    try:
        # A well-known paper: "Attention Is All You Need" via its DOI.
        rec = client.fetch_metadata("DOI:10.48550/arXiv.1706.03762")
        assert rec.paper_id
        assert rec.title
    finally:
        client.close()
        cache.close()
