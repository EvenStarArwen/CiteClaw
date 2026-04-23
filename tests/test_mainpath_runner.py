"""End-to-end test for :func:`citeclaw.mainpath.run_mpa`.

Writes a small GraphML to a tmp dir, runs the pipeline, and verifies
both the output GraphML and the JSON summary end up on disk with
sensible contents.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import pytest

from citeclaw.mainpath import MainPathResult, run_mpa


_LABELS = ["3", "5", "12", "15", "20", "21", "22"]
_EDGES = [
    ("3", "5"),
    ("3", "21"),
    ("5", "12"),
    ("12", "15"),
    ("12", "20"),
    ("15", "22"),
    ("20", "21"),
    ("20", "22"),
    ("21", "22"),
]


def _write_hd_graphml(path: Path) -> None:
    g = ig.Graph(directed=True)
    g.add_vertices(_LABELS)
    g.vs["paper_id"] = list(_LABELS)
    g.vs["title"] = [f"Paper {lbl}" for lbl in _LABELS]
    # Ascending years so topological order = chronological.
    g.vs["year"] = [2010, 2011, 2012, 2013, 2013, 2014, 2015]
    g.add_edges(_EDGES)
    g.write_graphml(str(path))


def test_run_mpa_end_to_end(tmp_path: Path):
    graph_path = tmp_path / "input.graphml"
    _write_hd_graphml(graph_path)
    output_path = tmp_path / "mp.graphml"

    result = run_mpa(
        graph_path=graph_path,
        output_path=output_path,
    )

    assert isinstance(result, MainPathResult)
    assert result.weight_variant == "spc"
    assert result.search_variant == "key-route"
    assert result.cycle_policy == "shrink"

    # Files exist.
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()

    # Subgraph contains the expected 6 nodes / 6 edges (the key-route
    # forward + top-arc combination, same as test_mainpath_search).
    assert result.subgraph.vcount() == 6
    assert result.subgraph.ecount() == 6

    # Roles: 3 → 5 → 12 → 20 → {21, 22}; 21 → 22. So 3 is source,
    # 22 is sink, everything else is intermediate.
    roles = {v["paper_id"]: v["mp_role"] for v in result.subgraph.vs}
    assert roles["3"] == "source"
    assert roles["22"] == "sink"
    for pid in ("5", "12", "20", "21"):
        assert roles[pid] == "intermediate"

    # All edges have mp_weight set and positive.
    assert all(e["mp_weight"] > 0 for e in result.subgraph.es)

    # paper_order is chronological (topological).
    assert result.paper_order[0] == "3"
    assert result.paper_order[-1] == "22"

    # JSON summary has provenance + papers list.
    summary = json.loads(output_path.with_suffix(".json").read_text())
    assert summary["weight_variant"] == "spc"
    assert summary["search_variant"] == "key-route"
    assert summary["cycle_policy"] == "shrink"
    assert len(summary["papers"]) == 6
    assert summary["papers"][0]["paper_id"] == "3"
    assert summary["papers"][0]["role"] == "source"


def test_run_mpa_rejects_unknown_variant(tmp_path: Path):
    graph_path = tmp_path / "input.graphml"
    _write_hd_graphml(graph_path)
    output_path = tmp_path / "mp.graphml"

    with pytest.raises(ValueError, match="weight"):
        run_mpa(
            graph_path=graph_path, output_path=output_path,
            weight="not-a-weight",
        )
    with pytest.raises(ValueError, match="search"):
        run_mpa(
            graph_path=graph_path, output_path=output_path,
            search="not-a-search",
        )
    with pytest.raises(ValueError, match="cycle"):
        run_mpa(
            graph_path=graph_path, output_path=output_path,
            cycle="not-a-cycle",
        )


def test_run_mpa_handles_cyclic_input(tmp_path: Path):
    """A 2-cycle should be collapsed by the default shrink policy."""
    g = ig.Graph(directed=True)
    g.add_vertices(["a", "b", "c"])
    g.vs["paper_id"] = ["a", "b", "c"]
    g.vs["title"] = ["A", "B", "C"]
    g.vs["year"] = [2010, 2012, 2020]
    g.add_edges([("a", "b"), ("b", "a"), ("b", "c")])  # 2-cycle between a & b
    graph_path = tmp_path / "cyclic.graphml"
    g.write_graphml(str(graph_path))

    output_path = tmp_path / "mp.graphml"
    result = run_mpa(graph_path=graph_path, output_path=output_path)

    # After shrink: {a,b} collapse to 'a' (older year), so output is a→c.
    assert result.subgraph.vcount() == 2
    assert result.subgraph.ecount() == 1
    assert result.cycle_trace.scc_sizes == [2]
    assert "a" in result.cycle_trace.supernode_members


def test_run_mpa_all_variants_run_cleanly(tmp_path: Path):
    """Smoke test: every combination of weight × search × cycle should
    produce a non-empty result on this small graph without error."""
    graph_path = tmp_path / "input.graphml"
    _write_hd_graphml(graph_path)

    for weight in ("spc", "splc", "spnp"):
        for search in ("local-forward", "local-backward", "global",
                       "key-route", "multi-local"):
            for cycle in ("shrink", "preprint"):
                output_path = tmp_path / f"mp_{weight}_{search}_{cycle}.graphml"
                result = run_mpa(
                    graph_path=graph_path, output_path=output_path,
                    weight=weight, search=search, cycle=cycle,
                    tolerance=0.2,
                )
                assert result.subgraph.vcount() > 0, (
                    f"Empty output for weight={weight} search={search} "
                    f"cycle={cycle}"
                )
