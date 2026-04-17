"""End-to-end stub-mode pipeline test for the ExpandBy* family (PC-08).

Wires every Phase B/C step into a single ``Settings.pipeline`` and
runs it through ``run_pipeline`` against the deterministic offline
pieces:

  * :class:`StubClient` for LLM calls (no real provider)
  * :class:`_BudgetAwareFakeS2` (a thin in-test subclass of the
    Phase A ``FakeS2Client``) for every S2 surface — bumps
    ``budget._s2_api`` per call so the budget assertions hold without
    touching the shared fake.

The pipeline:

    ResolveSeeds → LoadSeeds → ExpandForward → Rerank →
    ExpandBySearch → ExpandBySemantics → ExpandByAuthor → Finalize

The asserted contract (from the PC-08 spec):

  1. Every step appears in the rendered shape table
     (``shape_summary.txt`` written by ``run_pipeline``).
  2. ``ctx.collection`` grows across each expand step.
  3. ``budget._s2_api`` has entries for ``search``, ``recommendations``,
     and ``author_papers``.
  4. ``budget._llm_tokens["meta_search_agent"] > 0``.
  5. Re-running the pipeline is a no-op — final ``ctx.collection``
     size is unchanged.

Phase C is screening-blocked on PC-08 + PC-09; this file is the
canonical contract test for the whole expansion family.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, SeedPaper, Settings
from citeclaw.context import Context
from citeclaw.pipeline import run_pipeline
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Budget-aware fake — bumps record_s2 per surface so the spec's
# ``budget._s2_api`` assertions can hold against the offline fake.
# ---------------------------------------------------------------------------


class _BudgetAwareFakeS2(FakeS2Client):
    """Bumps ``budget.record_s2(...)`` for every PC-08-relevant surface
    and falls back to a default recommendation set when the agent's
    rerank-anchored ``positive_ids`` don't have an exact canned entry.
    """

    def __init__(self, budget: BudgetTracker) -> None:
        super().__init__()
        self._tracked_budget = budget
        self._default_recs: list[dict[str, Any]] = []

    def search_bulk(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._tracked_budget.record_s2("search")
        return super().search_bulk(*args, **kwargs)

    def fetch_recommendations(
        self,
        positive_ids: list[str],
        *,
        negative_ids: list[str] | None = None,
        limit: int = 100,
        fields: str = "paperId,title",
    ) -> list[dict[str, Any]]:
        self._tracked_budget.record_s2("recommendations")
        key = tuple(sorted(positive_ids))
        if key in self._recommendation_results:
            papers = self._recommendation_results[key]
        else:
            # Default fallback so the test doesn't have to predict the
            # exact rerank anchor set.
            papers = self._default_recs
        return [copy.deepcopy(p) for p in papers[:limit]]

    def fetch_author_papers(
        self,
        author_id: str,
        *,
        limit: int = 100,
        fields: str = "paperId,title,year,venue,citationCount",
    ) -> list[dict[str, Any]]:
        self._tracked_budget.record_s2("author_papers")
        return super().fetch_author_papers(author_id, limit=limit, fields=fields)


# ---------------------------------------------------------------------------
# Corpus builder — designs the fake S2 universe so every step in the
# pipeline finds something to expand.
# ---------------------------------------------------------------------------


def _build_corpus(fs: _BudgetAwareFakeS2) -> None:
    """Populate the fake S2 client with a corpus that drives every
    step in the pipeline through to a non-zero delta_collection.

    Authors are wired so that the recommendations (REC-1, REC-2)
    list ``A2_HIGH`` / ``A3_MID`` as authors — that's the signal
    ExpandByAuthor will see, and those two authors have h-index
    metadata + author-paper lists registered, so the author-graph
    traversal lands on real new candidates.
    """
    # ----- Authors -----
    fs.add_author("A1_LOW",  {"name": "Alice", "hIndex": 5,  "citationCount": 100})
    fs.add_author("A2_HIGH", {"name": "Bob",   "hIndex": 20, "citationCount": 1000})
    fs.add_author("A3_MID",  {"name": "Carol", "hIndex": 12, "citationCount": 300})

    # ----- Seeds -----
    seed_1 = make_paper(
        "SEED-1",
        title="Foundational Paper On Protein Folding",
        year=2020,
        citation_count=1000,
        venue="Nature",
        authors=[
            {"authorId": "A1_LOW",  "name": "Alice"},
            {"authorId": "A2_HIGH", "name": "Bob"},
        ],
    )
    seed_2 = make_paper(
        "SEED-2",
        title="A Sequel Foundation",
        year=2021,
        citation_count=800,
        venue="Science",
        authors=[
            {"authorId": "A2_HIGH", "name": "Bob"},
            {"authorId": "A3_MID",  "name": "Carol"},
        ],
    )
    fs.add(seed_1)
    fs.add(seed_2)

    # ResolveSeeds resolves the title-only seed via search_match.
    fs.register_search_match(
        "Foundational Paper On Protein Folding", seed_1,
    )

    # ----- Citers (for ExpandForward) -----
    citer_1 = make_paper(
        "CITER-1",
        title="A Follow-up to the Foundation",
        year=2022,
        citation_count=300,
        venue="ICML",
        references=["SEED-1"],
        authors=[{"authorId": "A1_LOW", "name": "Alice"}],
    )
    citer_2 = make_paper(
        "CITER-2",
        title="Another Follow-up",
        year=2023,
        citation_count=200,
        venue="NeurIPS",
        references=["SEED-2"],
        authors=[{"authorId": "A3_MID", "name": "Carol"}],
    )
    fs.add(citer_1)
    fs.add(citer_2)

    # ----- Search-bulk results (for ExpandBySearch) -----
    # The stub LLM emits two distinct queries across iterations:
    #   "test topic"           → initial state
    #   "test topic narrowed"  → refine + satisfied states (same text)
    search_a = make_paper(
        "SEARCH-1",
        title="A Searched Paper",
        year=2022,
        citation_count=150,
        venue="JMLR",
        authors=[{"authorId": "A1_LOW", "name": "Alice"}],
    )
    search_b = make_paper(
        "SEARCH-2",
        title="Another Searched Paper",
        year=2023,
        citation_count=120,
        venue="ICLR",
        authors=[{"authorId": "A2_HIGH", "name": "Bob"}],
    )
    search_c = make_paper(
        "SEARCH-3",
        title="A Narrowed Result",
        year=2023,
        citation_count=80,
        venue="ACL",
        authors=[{"authorId": "A3_MID", "name": "Carol"}],
    )
    fs.add(search_a)
    fs.add(search_b)
    fs.add(search_c)
    fs.register_search_bulk("test topic", [search_a, search_b])
    fs.register_search_bulk("test topic narrowed", [search_c])

    # ----- Recommendations (for ExpandBySemantics) -----
    # REC-1 / REC-2 list A2_HIGH / A3_MID as authors so ExpandByAuthor
    # (which runs after ExpandBySemantics) finds those high-hIndex
    # authors in its input signal.
    rec_1 = make_paper(
        "REC-1",
        title="A Recommended Paper",
        year=2022,
        citation_count=250,
        venue="Nature Methods",
        authors=[{"authorId": "A2_HIGH", "name": "Bob"}],
    )
    rec_2 = make_paper(
        "REC-2",
        title="Another Recommended Paper",
        year=2023,
        citation_count=180,
        venue="Cell",
        authors=[{"authorId": "A3_MID", "name": "Carol"}],
    )
    fs.add(rec_1)
    fs.add(rec_2)
    # Default fallback: any rerank-anchored positive_ids returns these.
    fs._default_recs = [rec_1, rec_2]

    # ----- Author papers (for ExpandByAuthor) -----
    auth_paper_1 = make_paper(
        "AUTH-1",
        title="Bob's Other Paper 1",
        year=2022,
        citation_count=90,
        venue="Bioinformatics",
        authors=[{"authorId": "A2_HIGH", "name": "Bob"}],
    )
    auth_paper_2 = make_paper(
        "AUTH-2",
        title="Bob's Other Paper 2",
        year=2023,
        citation_count=70,
        venue="bioRxiv",
        authors=[{"authorId": "A2_HIGH", "name": "Bob"}],
    )
    auth_paper_3 = make_paper(
        "AUTH-3",
        title="Carol's Other Paper",
        year=2024,
        citation_count=60,
        venue="Nature",
        authors=[{"authorId": "A3_MID", "name": "Carol"}],
    )
    fs.add(auth_paper_1)
    fs.add(auth_paper_2)
    fs.add(auth_paper_3)
    fs.register_author_papers("A2_HIGH", [auth_paper_1, auth_paper_2])
    fs.register_author_papers("A3_MID",  [auth_paper_3])


# ---------------------------------------------------------------------------
# Settings + Context fixtures
# ---------------------------------------------------------------------------


def _build_pipeline_dict() -> list[dict]:
    """The minimal but complete chain from the PC-08 spec."""
    return [
        {"step": "ResolveSeeds", "include_siblings": False},
        {"step": "LoadSeeds"},
        {"step": "ExpandForward",
         "max_citations": 10, "screener": "permissive"},
        {"step": "Rerank", "metric": "citation", "k": 10},
        {"step": "ExpandBySearch",
         "agent": {"worker_max_turns": 15, "supervisor_max_turns": 6,
                   "max_queries_per_worker": 2},
         "screener": "permissive"},
        {"step": "ExpandBySemantics",
         "max_anchor_papers": 2, "limit": 10,
         "screener": "permissive"},
        {"step": "ExpandByAuthor",
         "top_k_authors": 2, "author_metric": "h_index",
         "papers_per_author": 10, "screener": "permissive"},
        {"step": "Finalize"},
    ]


@pytest.fixture
def e2e_settings(tmp_path: Path) -> Settings:
    """A Settings instance with the full ExpandBy* pipeline + a single
    permissive YearFilter screener that all the expand steps share."""
    return Settings(
        screening_model="stub",
        data_dir=tmp_path / "e2e_data",
        topic_description="A test topic for the expansion family",
        seed_papers=[
            SeedPaper(title="Foundational Paper On Protein Folding"),
            SeedPaper(paper_id="SEED-2"),
        ],
        max_papers_total=10_000,
        blocks={
            "permissive": {
                "type": "YearFilter",
                "min": 2018,
                "max": 2030,
            },
        },
        pipeline=_build_pipeline_dict(),
    )


@pytest.fixture
def e2e_ctx(
    e2e_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Context:
    """A Context wired with the budget-aware fake S2 client + a real
    Cache + the test data dir. ``CITECLAW_NO_DASHBOARD=1`` keeps the
    pipeline runner from constructing a real Dashboard even on a TTY."""
    monkeypatch.setenv("CITECLAW_NO_DASHBOARD", "1")
    e2e_settings.data_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(e2e_settings.data_dir / "cache.db")
    budget = BudgetTracker()
    fake = _BudgetAwareFakeS2(budget)
    _build_corpus(fake)
    return Context(config=e2e_settings, s2=fake, cache=cache, budget=budget)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_shape_summary(ctx: Context) -> str:
    path = ctx.config.data_dir / "shape_summary.txt"
    assert path.exists(), f"shape_summary.txt not written: {path}"
    return path.read_text()


def _delta_for_step(shape_text: str, step_name: str) -> int:
    """Parse the Δcoll column from the shape table for ``step_name``.

    Format per row: ``Step                  | In | Out | Δcoll | Notes``
    """
    for line in shape_text.splitlines():
        if not line.startswith(step_name):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        try:
            return int(parts[3])
        except ValueError:
            continue
    raise AssertionError(
        f"Δcoll for step {step_name!r} not found in shape table:\n{shape_text}"
    )


# ---------------------------------------------------------------------------
# The contract test
# ---------------------------------------------------------------------------


class TestExpandFamilyEndToEnd:
    def test_full_pipeline_runs_to_finalize(self, e2e_ctx: Context):
        """The whole 8-step chain (plus the auto-injected
        MergeDuplicates) should run without raising and write a
        shape_summary.txt to data_dir."""
        run_pipeline(e2e_ctx)
        shape = _read_shape_summary(e2e_ctx)
        # Every user-defined step appears in the shape table.
        for name in (
            "ResolveSeeds", "LoadSeeds", "ExpandForward", "Rerank",
            "ExpandBySearch", "ExpandBySemantics", "ExpandByAuthor",
            "MergeDuplicates",  # auto-injected before Finalize
            "Finalize",
        ):
            assert name in shape, f"step {name!r} missing from shape table:\n{shape}"

    def test_collection_grows_across_each_expand_step(self, e2e_ctx: Context):
        """Each expand step (LoadSeeds, ExpandForward, ExpandBySearch,
        ExpandBySemantics, ExpandByAuthor) must show a positive
        ``Δcoll`` in the shape table — proving the corpus drives every
        step through to a non-zero delta."""
        run_pipeline(e2e_ctx)
        shape = _read_shape_summary(e2e_ctx)
        for name in (
            "LoadSeeds",
            "ExpandForward",
            "ExpandBySearch",
            "ExpandBySemantics",
            "ExpandByAuthor",
        ):
            delta = _delta_for_step(shape, name)
            assert delta > 0, (
                f"step {name!r} added {delta} papers — expected > 0\n{shape}"
            )

    def test_budget_s2_api_records_three_required_surfaces(self, e2e_ctx: Context):
        """``budget._s2_api`` must carry positive counts for the three
        Phase B/C surfaces the spec calls out."""
        run_pipeline(e2e_ctx)
        api = e2e_ctx.budget._s2_api
        assert api.get("search", 0) > 0, (
            f"missing 'search' s2_api entry; have {dict(api)}"
        )
        assert api.get("recommendations", 0) > 0, (
            f"missing 'recommendations' s2_api entry; have {dict(api)}"
        )
        assert api.get("author_papers", 0) > 0, (
            f"missing 'author_papers' s2_api entry; have {dict(api)}"
        )

    def test_budget_llm_tokens_expand_by_search_agents_recorded(self, e2e_ctx: Context):
        """The v2 supervisor + worker must spend their LLM tokens under
        their own categories so cost tracking attributes spend
        correctly. Categories:
          - expand_by_search_supervisor
          - expand_by_search_worker:<spec_id>
        """
        run_pipeline(e2e_ctx)
        by_cat = e2e_ctx.budget._llm_tokens
        assert by_cat.get("expand_by_search_supervisor", 0) > 0, (
            f"supervisor tokens not recorded; have {dict(by_cat)}"
        )
        worker_cats = [k for k in by_cat if k.startswith("expand_by_search_worker:")]
        assert worker_cats, f"no worker categories recorded; have {dict(by_cat)}"
        assert sum(by_cat[c] for c in worker_cats) > 0

    def test_rerunning_pipeline_is_a_no_op(self, e2e_ctx: Context):
        """Re-running the same pipeline through ``run_pipeline`` on the
        same Context must NOT grow the collection — every paper is
        already either in ``ctx.collection`` or in ``ctx.seen``, and
        the searched_signals fingerprints / per-paper dedup short-
        circuit each expand step."""
        run_pipeline(e2e_ctx)
        size_after_first = len(e2e_ctx.collection)
        assert size_after_first > 0
        run_pipeline(e2e_ctx)
        size_after_second = len(e2e_ctx.collection)
        assert size_after_second == size_after_first, (
            f"second run grew collection from {size_after_first} to "
            f"{size_after_second}"
        )
