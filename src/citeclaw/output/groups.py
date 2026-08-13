"""Group artifacts — per-paper coordinates + topic / community tables.

Writes the run-directory files the Explore screen's topic map and
community cards read. Everything here is derived from the
:class:`~citeclaw.cluster.base.ClusterResult` objects the pipeline
already parked in ``ctx.clusters``; nothing re-runs an algorithm and
nothing hits the network, so this is safe to call at Finalize time.

Files written (all optional — a run with no ``Cluster`` step writes none):

``paper_groups.csv``
    One row per paper: display coordinates plus every group id it holds.
``topics.csv`` / ``communities_<algorithm>.csv``
    One row per group, largest first, matching the design fixture's
    column names.
``community_methods.csv``
    One row per graph partition with modularity + agreement against the
    topic model — the numbers behind the ⓘ "How this was computed" panel.
``groups.json``
    Machine-readable index of the above: what ran, with which knobs, and
    how many groups came out.

The full schema, including how each column is computed and where it
diverges from ``web/design-reference/uploads/run37/``, is documented in
``web/wiring/artifacts-groups.md``. Keep the two in sync.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import median

from citeclaw.cluster.agreement import adjusted_rand, normalized_mutual_info
from citeclaw.cluster.base import ClusterResult

log = logging.getLogger("citeclaw.output.groups")

#: Multi-value cell separators, copied from the design fixture so the UI's
#: existing split() calls keep working unchanged.
KEYWORD_SEP = "; "
ID_SEP = "; "
TITLE_SEP = " | "

#: Algorithms that partition the citation graph (as opposed to the
#: embedding space). These get a ``communities_*.csv`` and a
#: ``community_methods.csv`` row; the topic model gets ``topics.csv``.
GRAPH_ALGORITHMS = ("leiden", "louvain", "walktrap", "infomap", "label_propagation")
TOPIC_ALGORITHM = "topic_model"

GROUP_COLUMNS = [
    "label", "description", "size", "keywords",
    "centroid_x", "centroid_y",
    "representative_paper_ids", "representative_paper_titles",
]
METHOD_COLUMNS = [
    "method", "n_communities", "modularity", "largest_community",
    "largest_fraction", "singletons", "median_size",
    "nmi_vs_topic_model", "adjusted_rand_vs_topic_model",
]


def _fmt(value: float | None, ndigits: int = 2) -> str:
    """Render a float for a CSV cell; ``None``/NaN become an empty cell.

    An empty cell is the honest encoding of "not computed" — writing 0.0
    would put a paper at the top-left corner of the topic map and a
    method at modularity zero, both of which read as real measurements.

    Formatting is ``str(round(v, ndigits))``, which keeps the trailing
    ``.0`` on whole numbers (``707.0``, ``25.0``) and drops redundant
    trailing zeros elsewhere (``444.9``, not ``444.90``). That is
    precisely what the design fixture's CSVs contain, verified column by
    column against ``run37/accepted.csv``, ``topics.csv`` and
    ``community_methods.csv`` — a ``%g``-style format instead writes bare
    ``707`` and would make the two files diff noisily for no reason.
    """
    if value is None:
        return ""
    if value != value:  # NaN
        return ""
    return str(round(float(value), ndigits))


def _pick_results(clusters: dict[str, ClusterResult]) -> tuple[
    tuple[str, ClusterResult] | None, list[tuple[str, ClusterResult]],
]:
    """Split ``ctx.clusters`` into ``(topic_entry, graph_entries)``.

    Entries keep their ``store_as`` key so a pipeline that clustered
    twice (e.g. a scoped Cluster step inside one Parallel branch plus a
    corpus-wide one) still produces distinctly-named files. The *last*
    topic-model result wins when several exist, on the assumption that a
    later step saw more of the corpus.
    """
    topic: tuple[str, ClusterResult] | None = None
    graphs: list[tuple[str, ClusterResult]] = []
    for name, result in clusters.items():
        if not isinstance(result, ClusterResult) or not result.membership:
            continue
        if result.algorithm == TOPIC_ALGORITHM:
            topic = (name, result)
        elif result.algorithm in GRAPH_ALGORITHMS:
            graphs.append((name, result))
        else:
            # An out-of-tree clusterer. We have no idea whether its ids
            # live in citation space or embedding space, so we can't file
            # it — but say so rather than dropping it silently.
            log.warning(
                "write_group_artifacts: cluster %r uses unrecognised algorithm %r; "
                "it will not appear in the group artifacts. Add it to "
                "GRAPH_ALGORITHMS in citeclaw.output.groups if it partitions "
                "the citation graph.",
                name, result.algorithm,
            )
    return topic, graphs


def _file_stem(kind: str, name: str, result: ClusterResult, seen: Counter) -> str:
    """``communities_leiden`` for the first leiden result, ``…_<store_as>`` after.

    Single-partition runs — the overwhelmingly common case, and the one
    the UI fixture encodes — get the plain fixture filename. Only a
    genuine collision pulls in the ``store_as`` disambiguator, so the
    happy path never surprises a consumer with an unexpected name.
    """
    algo = result.algorithm or "unknown"
    seen[algo] += 1
    if seen[algo] == 1:
        return f"{kind}_{algo}"
    return f"{kind}_{algo}_{name}"


def _centroid(
    paper_ids: list[str], coords: dict[str, tuple[float, float]],
) -> tuple[float | None, float | None]:
    """Arithmetic mean of the members' 2D coordinates.

    The mean (not the median) is what the design fixture's
    ``centroid_x`` / ``centroid_y`` columns hold — verified to two
    decimals across all 19 topics and 12 communities of run37. Returns
    ``(None, None)`` when no member has coordinates.
    """
    pts = [coords[p] for p in paper_ids if p in coords]
    if not pts:
        return None, None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _group_rows(
    result: ClusterResult,
    coords: dict[str, tuple[float, float]],
    collection: dict,
    *,
    id_column: str,
) -> list[dict]:
    """Build the per-group rows for a ``topics.csv`` / ``communities_*.csv``.

    Rows are sorted largest-group-first (the order the UI renders cards
    in) and the noise bucket ``-1`` is excluded — it is not a group, it
    is the absence of one, and the UI renders unassigned papers as grey
    dots straight from ``paper_groups.csv``.

    ``size`` is recomputed from ``membership`` rather than trusted from
    ``ClusterMetadata.size``: the metadata is filled in by the naming
    pipeline, which may not have run.
    """
    members: dict[int, list[str]] = {}
    for pid, cid in result.membership.items():
        if cid == -1:
            continue
        members.setdefault(cid, []).append(pid)

    rows: list[dict] = []
    for cid, paper_ids in members.items():
        paper_ids.sort()
        meta = result.metadata.get(cid)
        reps = list(meta.representative_papers) if meta else []
        rep_titles = []
        for pid in reps:
            paper = collection.get(pid)
            title = getattr(paper, "title", "") or ""
            rep_titles.append(title.replace(TITLE_SEP.strip(), "/"))
        cx, cy = _centroid(paper_ids, coords)
        rows.append({
            id_column: cid,
            "label": (meta.label if meta else "") or "",
            "description": (meta.summary if meta else "") or "",
            "size": len(paper_ids),
            "keywords": KEYWORD_SEP.join(meta.keywords) if meta else "",
            "centroid_x": _fmt(cx),
            "centroid_y": _fmt(cy),
            "representative_paper_ids": ID_SEP.join(reps),
            "representative_paper_titles": TITLE_SEP.join(rep_titles),
        })
    rows.sort(key=lambda r: (-r["size"], r[id_column]))
    return rows


def _method_row(
    result: ClusterResult,
    n_papers: int,
    topic_membership: dict[str, int] | None,
) -> dict:
    """One ``community_methods.csv`` row: shape + quality of one partition.

    ``modularity`` comes from ``ClusterResult.quality`` (the clusterer
    computed it over the graph it actually partitioned); a clusterer that
    didn't report one leaves the cell empty rather than getting a
    recomputed-here number that might describe a different graph.

    NMI / ARI compare against the topic partition over the papers *both*
    partitions cover, with the topic model's ``-1`` noise bucket treated
    as an ordinary label — the convention the fixture used (see
    :mod:`citeclaw.cluster.agreement`). Empty when no topic model ran.
    """
    sizes = sorted(Counter(
        c for c in result.membership.values() if c != -1
    ).values(), reverse=True)
    largest = sizes[0] if sizes else 0
    nmi_cell = ari_cell = ""
    if topic_membership:
        shared = sorted(set(result.membership) & set(topic_membership))
        if len(shared) >= 2:
            a = [result.membership[p] for p in shared]
            b = [topic_membership[p] for p in shared]
            nmi_cell = _fmt(normalized_mutual_info(a, b), 4)
            ari_cell = _fmt(adjusted_rand(a, b), 4)
    return {
        "method": result.algorithm,
        "n_communities": len(sizes),
        "modularity": _fmt(result.quality.get("modularity"), 4),
        "largest_community": largest,
        "largest_fraction": _fmt(largest / n_papers if n_papers else None, 4),
        "singletons": sum(1 for s in sizes if s == 1),
        "median_size": int(median(sizes)) if sizes else 0,
        "nmi_vs_topic_model": nmi_cell,
        "adjusted_rand_vs_topic_model": ari_cell,
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write ``rows`` as UTF-8 CSV with a trailing newline and ``\\n`` line ends."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_group_artifacts(ctx, *, suffix: str | None = None) -> list[Path]:
    """Write every group artifact for ``ctx`` into the run directory.

    Returns the list of paths written (empty when the run has no usable
    cluster results — a pipeline with no ``Cluster`` step is a normal
    configuration, not an error). ``suffix`` is inserted before the file
    extension for non-destructive regeneration, matching
    :func:`citeclaw.steps.finalize.write_graphs`; it defaults to the
    run's ``.expN`` iteration tag so a ``--continue-from`` run does not
    clobber the previous iteration's group tables.

    Coordinates come from whichever result carries ``coords_2d`` — in
    practice the topic model, since it is the only clusterer that
    embeds. Community tables reuse those same coordinates for their
    centroids, exactly as the fixture does: a community's position on
    the topic map is where its papers sit in *embedding* space, which is
    why community centroids look scattered while topic centroids look
    tight.
    """
    clusters = getattr(ctx, "clusters", None) or {}
    topic_entry, graph_entries = _pick_results(clusters)
    if topic_entry is None and not graph_entries:
        log.debug("write_group_artifacts: no cluster results on ctx; nothing to write")
        return []

    data_dir = Path(ctx.config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    collection = getattr(ctx, "collection", {}) or {}
    if suffix is None:
        iteration = int(getattr(ctx, "iteration", 1) or 1)
        suffix = f".exp{iteration}" if iteration > 1 else ""

    def _path(stem: str, ext: str) -> Path:
        return data_dir / f"{stem}{suffix}.{ext}"

    coords: dict[str, tuple[float, float]] = {}
    for _, result in clusters.items():
        if isinstance(result, ClusterResult) and result.coords_2d:
            coords.update(result.coords_2d)

    written: list[Path] = []
    index: dict = {
        "coords_2d_papers": len(coords),
        "coord_space": {"box": 1000.0, "pad": 25.0, "origin": "top_left"},
        "topics": None,
        "communities": [],
    }

    # ----- per-group tables -----
    topic_membership: dict[str, int] | None = None
    if topic_entry is not None:
        name, result = topic_entry
        topic_membership = result.membership
        rows = _group_rows(result, coords, collection, id_column="topic_id")
        path = _path("topics", "csv")
        _write_csv(path, ["topic_id", *GROUP_COLUMNS], rows)
        written.append(path)
        index["topics"] = {
            "store_as": name,
            "file": path.name,
            "algorithm": result.algorithm,
            "n_topics": len(rows),
            "n_noise": sum(1 for c in result.membership.values() if c == -1),
            "n_papers": len(result.membership),
        }

    seen: Counter = Counter()
    for name, result in graph_entries:
        stem = _file_stem("communities", name, result, seen)
        rows = _group_rows(result, coords, collection, id_column="community_id")
        path = _path(stem, "csv")
        _write_csv(path, ["community_id", *GROUP_COLUMNS], rows)
        written.append(path)
        index["communities"].append({
            "store_as": name,
            "file": path.name,
            "algorithm": result.algorithm,
            "n_communities": len(rows),
            "n_papers": len(result.membership),
            "quality": dict(result.quality),
        })

    # ----- per-paper table -----
    paper_ids = sorted(
        set(coords)
        | {p for _, r in graph_entries for p in r.membership}
        | (set(topic_membership) if topic_membership else set())
    )
    topic_labels: dict[int, str] = {}
    if topic_entry is not None:
        topic_labels = {
            cid: (meta.label or "")
            for cid, meta in topic_entry[1].metadata.items()
        }
    community_cols = []
    seen_cols: Counter = Counter()
    for name, result in graph_entries:
        seen_cols[result.algorithm] += 1
        col = (
            f"community_{result.algorithm}"
            if seen_cols[result.algorithm] == 1
            else f"community_{result.algorithm}_{name}"
        )
        community_cols.append((col, result.membership))

    columns = ["id", "x", "y"]
    if topic_entry is not None:
        columns += ["topic", "topic_label"]
    columns += [c for c, _ in community_cols]

    paper_rows: list[dict] = []
    for pid in paper_ids:
        xy = coords.get(pid)
        row: dict = {
            "id": pid,
            "x": _fmt(xy[0]) if xy else "",
            "y": _fmt(xy[1]) if xy else "",
        }
        if topic_entry is not None:
            tid = (topic_membership or {}).get(pid)
            row["topic"] = "" if tid is None else tid
            row["topic_label"] = topic_labels.get(tid, "") if tid is not None else ""
        for col, membership in community_cols:
            cid = membership.get(pid)
            row[col] = "" if cid is None else cid
        paper_rows.append(row)
    papers_path = _path("paper_groups", "csv")
    _write_csv(papers_path, columns, paper_rows)
    written.append(papers_path)
    index["papers"] = {"file": papers_path.name, "n_papers": len(paper_rows)}

    # ----- method comparison table -----
    if graph_entries:
        n_papers = len(paper_ids)
        method_rows = [
            _method_row(result, n_papers, topic_membership)
            for _, result in graph_entries
        ]
        method_rows.sort(key=lambda r: r["method"])
        methods_path = _path("community_methods", "csv")
        _write_csv(methods_path, METHOD_COLUMNS, method_rows)
        written.append(methods_path)
        index["methods"] = {"file": methods_path.name, "rows": method_rows}

    index_path = _path("groups", "json")
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    written.append(index_path)

    log.info(
        "group artifacts: wrote %d files to %s (%d papers, %d with coords)",
        len(written), data_dir, len(paper_rows), len(coords),
    )
    return written
