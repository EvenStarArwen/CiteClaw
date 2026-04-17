"""Finalize step — write output artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from citeclaw.author_graph import export_author_graphml
from citeclaw.models import PaperRecord
from citeclaw.output import (
    build_output,
    export_graphml,
    with_iteration_suffix,
    write_bibtex,
    write_json,
    write_run_state,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.finalize")


def _write_rejections_json(ctx, path: Path) -> None:
    """Write ``rejections.json`` — per-paper rejection ledger.

    Schema (compact by design; no abstracts so the file stays scannable
    even on 50k-candidate runs)::

        {
          "<paper_id>": {
            "categories": ["year_filter", "llm_topic_llm"],   # dedup'd, ordered
            "title": "<first 200 chars>",                     # best-effort
            "year": 2023                                       # best-effort
          }
        }

    Title + year are pulled from the S2 metadata cache when available,
    so rejected papers that were never loaded into ``ctx.collection``
    still get an identifying row. Missing cache entries produce the
    ``categories``-only shape.
    """
    rejections: dict[str, dict] = {}
    for pid, categories in ctx.rejection_ledger.items():
        seen: set[str] = set()
        unique_cats: list[str] = []
        for c in categories:
            if c not in seen:
                seen.add(c)
                unique_cats.append(c)
        entry: dict = {"categories": unique_cats}
        try:
            cached = ctx.cache.get_metadata(pid)
        except Exception:
            cached = None
        if isinstance(cached, dict):
            title = cached.get("title")
            if isinstance(title, str) and title:
                entry["title"] = title[:200]
            year = cached.get("year")
            if isinstance(year, int):
                entry["year"] = year
        rejections[pid] = entry
    path.write_text(json.dumps(rejections, indent=2) + "\n")


def _inject_suffix(path: Path, suffix: str) -> Path:
    """Insert ``suffix`` before the file extension, e.g. ``.regen``.

    ``foo/citation_network.graphml`` + ``".regen"`` →
    ``foo/citation_network.regen.graphml``. An empty suffix is a no-op.
    """
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def write_graphs(ctx, *, suffix: str = "") -> None:
    """Write citation + collaboration graphs for the current ``ctx.collection``.

    Extracted from :meth:`Finalize.run` so the ``rebuild-graph`` CLI
    subcommand can invoke it without re-running the whole pipeline.

    ``suffix`` is inserted between the filename stem and extension (after
    any ``.expN`` iteration suffix), so callers can write non-destructively
    to e.g. ``citation_network.regen.graphml``.
    """
    cfg = ctx.config
    data_dir = cfg.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if len(ctx.collection) < 2:
        log.info("Skipping graph export: collection has %d papers (< 2)", len(ctx.collection))
        return

    graph_metadata = {
        "citeclaw_iteration": ctx.iteration,
        "citeclaw_generated_at": datetime.now().isoformat(timespec="seconds"),
        "citeclaw_parent_dir": str(ctx.prior_dir) if ctx.prior_dir else "",
        "citeclaw_total_papers": len(ctx.collection),
        "citeclaw_new_seed_count": len(ctx.new_seed_ids),
        "citeclaw_new_seed_ids": ",".join(ctx.new_seed_ids),
        "citeclaw_screening_model": cfg.screening_model,
    }

    # Populate per-edge metadata from the cached references for every
    # paper in the collection. This catches both backward edges (this
    # paper's references) and, transitively, forward edges (because the
    # citing paper is also in the collection and walks its own
    # references). One pass, no API calls — everything is in the cache.
    #
    # If MergeDuplicates has run, ``ctx.alias_map`` maps absorbed IDs to
    # their canonical replacements. Rewrite each edge's target through
    # the alias map *before* checking collection membership so we don't
    # silently drop edges that pointed at the preprint version of a
    # record we've since merged into the peer-reviewed canonical.
    alias_map = getattr(ctx, "alias_map", {}) or {}
    for p in ctx.collection.values():
        try:
            edges = ctx.s2.fetch_reference_edges(p.paper_id)
        except Exception:
            edges = []
        for edge in edges:
            tid = edge.get("target_id")
            if not tid:
                continue
            tid = alias_map.get(tid, tid)
            if tid not in ctx.collection or tid == p.paper_id:
                continue
            # If an earlier alias has already deposited metadata for this
            # edge, don't overwrite — keep the first wins.
            ctx.edge_meta.setdefault((tid, p.paper_id), {
                "contexts": edge.get("contexts") or [],
                "intents": edge.get("intents") or [],
                "is_influential": bool(edge.get("is_influential", False)),
            })

    graph_path = _inject_suffix(
        with_iteration_suffix(data_dir / "citation_network.graphml", ctx.iteration),
        suffix,
    )
    export_graphml(
        ctx.collection, graph_path,
        metadata=graph_metadata, edge_meta=ctx.edge_meta,
        s2=ctx.s2,
        clusters=getattr(ctx, "clusters", None) or None,
    )

    # ----- Author collaboration graph -----
    author_ids: list[str] = []
    seen_aids: set[str] = set()
    for p in ctx.collection.values():
        for a in (p.authors or []):
            if not isinstance(a, dict):
                continue
            aid = a.get("authorId")
            if aid and aid not in seen_aids:
                seen_aids.add(aid)
                author_ids.append(aid)
    author_details: dict = {}
    if author_ids:
        try:
            author_details = ctx.s2.fetch_authors_batch(author_ids)
        except Exception as exc:
            log.warning("fetch_authors_batch failed: %s", exc)
            author_details = {}
    collab_path = _inject_suffix(
        with_iteration_suffix(
            data_dir / "collaboration_network.graphml", ctx.iteration,
        ),
        suffix,
    )
    try:
        export_author_graphml(
            ctx.collection, author_details, collab_path,
            metadata=graph_metadata,
        )
    except Exception as exc:
        log.warning("collaboration graph export failed: %s", exc)


class Finalize:
    name = "Finalize"

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        cfg = ctx.config
        data_dir = cfg.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        dash = ctx.dashboard
        dash.note_candidates_seen(len(ctx.collection))

        # Enrich missing refs/abstracts
        no_refs = [p for p in ctx.collection.values() if not p.references]
        if no_refs:
            dash.begin_phase("enrich missing references", total=max(1, len(no_refs)))
            for p in no_refs:
                try:
                    p.references = ctx.s2.fetch_reference_ids(p.paper_id)
                except Exception:
                    pass
                dash.tick_inner(1)
        no_abs = [p for p in ctx.collection.values() if not p.abstract]
        if no_abs:
            dash.begin_phase("enrich missing abstracts", total=1)
            try:
                ctx.s2.enrich_with_abstracts(no_abs)
            except Exception as exc:
                log.warning("abstract enrich failed: %s", exc)
            dash.tick_inner(1)

        dash.begin_phase("build output", total=1)
        output = build_output(ctx.collection, ctx.rejected, ctx.seen, ctx.budget)
        output["summary"]["iteration"] = ctx.iteration
        output["summary"]["parent_dir"] = str(ctx.prior_dir) if ctx.prior_dir else ""
        output["summary"]["new_seed_ids"] = list(ctx.new_seed_ids)
        dash.tick_inner(1)

        dash.begin_phase("write JSON", total=1)
        coll_path = with_iteration_suffix(data_dir / "literature_collection.json", ctx.iteration)
        write_json(output, coll_path)
        dash.tick_inner(1)

        dash.begin_phase("write BibTeX", total=1)
        sorted_papers = sorted(
            ctx.collection.values(), key=lambda p: p.citation_count or 0, reverse=True,
        )
        bib_path = with_iteration_suffix(data_dir / "literature_collection.bib", ctx.iteration)
        write_bibtex(sorted_papers, bib_path)
        dash.tick_inner(1)

        dash.begin_phase("write run_state", total=1)
        state_path = with_iteration_suffix(data_dir / "run_state.json", ctx.iteration)
        write_run_state(
            ctx.collection, ctx.rejected, ctx.seen, [],
            ctx.budget, state_path,
            iteration=ctx.iteration,
            parent_dir=str(ctx.prior_dir) if ctx.prior_dir else "",
            new_seed_ids=list(ctx.new_seed_ids),
        )
        dash.tick_inner(1)

        dash.begin_phase("write graphs · citation + collab", total=1)
        write_graphs(ctx, suffix="")
        dash.tick_inner(1)

        dash.begin_phase("write rejections", total=1)
        rej_path = with_iteration_suffix(data_dir / "rejections.json", ctx.iteration)
        try:
            _write_rejections_json(ctx, rej_path)
        except Exception as exc:
            log.warning("rejections.json write failed: %s", exc)
        dash.tick_inner(1)

        ctx.result = output  # type: ignore[attr-defined]
        log.info(
            "PIPELINE COMPLETE: %d accepted, %d rejected, %d seen",
            len(ctx.collection), len(ctx.rejected), len(ctx.seen),
        )
        return StepResult(signal=[], in_count=len(signal), stats={"wrote": len(ctx.collection)})
