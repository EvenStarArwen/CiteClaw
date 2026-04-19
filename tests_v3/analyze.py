"""Quality report on a finished V3 test run.

Usage:
    python tests_v3/analyze.py <scenario_id> <model_key>

Emits one-shot stats + top-cited + cluster noise estimate to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("S2_API_KEY", os.environ.get("S2_API_KEY", ""))

from citeclaw.cache import Cache
from citeclaw.config import Settings, BudgetTracker
from citeclaw.clients.s2.api import SemanticScholarClient
from citeclaw.agents.v3.analysis import topic_model
from citeclaw.agents.v3.query_translate import to_natural


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: analyze.py <scenario_id> <model_key>", file=sys.stderr)
        sys.exit(1)
    scenario_id, model_key = sys.argv[1], sys.argv[2]
    data_dir = REPO_ROOT / "tests_v3" / "data" / f"{scenario_id}_{model_key}"
    runs = sorted(data_dir.glob("search_agent_transcripts/v3_run_*"))
    if not runs:
        print(f"no runs in {data_dir}")
        return
    run = runs[-1]
    events = [json.loads(l) for l in (run / "events.jsonl").open()]

    # -- Supervisor strategy
    print("=" * 72)
    print(f"scenario={scenario_id}  model={model_key}")
    print("=" * 72)
    strategy = []
    done_summary = ""
    for e in events:
        if e.get("type") != "tool_call":
            continue
        if e.get("scope") == "v3_supervisor":
            tn = e.get("tool_name", "")
            res = e.get("result") or {}
            args = e.get("args") or {}
            if tn == "set_strategy":
                for st in args.get("sub_topics", []):
                    strategy.append((st.get("id", ""), (st.get("description") or "")[:120]))
            elif tn == "add_sub_topics":
                for st in args.get("sub_topics", []):
                    strategy.append((st.get("id", "") + " [added]", (st.get("description") or "")[:120]))
            elif tn == "done":
                done_summary = (args.get("summary") or "")

    print("\n# sub-topics")
    for sid, desc in strategy:
        print(f"  · {sid}: {desc}")

    # -- Per-worker queries
    by_worker: dict[str, list[dict]] = {}
    for e in events:
        if e.get("type") != "tool_call":
            continue
        sc = e.get("scope", "")
        if not sc.startswith("v3_worker::"):
            continue
        wid = sc.split("::", 1)[1]
        by_worker.setdefault(wid, []).append(e)

    print("\n# per-worker iterations")
    for w, es in by_worker.items():
        props = [e for e in es if e.get("tool_name") == "propose_query"]
        summaries = [e for e in es if e.get("tool_name") == "iter_summary"]
        print(f"\n[{w}] {len(props)} iter(s):")
        for i, p in enumerate(props):
            q_nl = (p.get("result") or {}).get("query_nl") or (p.get("result") or {}).get("query", "")
            s = summaries[i] if i < len(summaries) else None
            if s:
                r = s.get("result") or {}
                total = r.get("total_count")
                new = r.get("new")
                seen = r.get("seen")
                diag = (r.get("diagnosis") or "")[:120]
                print(f"  iter {i}: total={total} new={new} seen={seen}")
                print(f"    query: {q_nl[:200]}")
                if diag:
                    print(f"    diag:  {diag}")

    # -- Re-fetch union from cache
    config = Settings(
        data_dir=str(data_dir),
        topic_description="dummy",
        seed_papers=[{"title": "dummy"}],
        screening_model="stub",
        pipeline=[],
    )
    budget = BudgetTracker()
    cache = Cache(Path(data_dir) / "cache.db")
    s2 = SemanticScholarClient(config, cache, budget)

    import logging
    logging.basicConfig(level=logging.WARNING)

    worker_ids: dict[str, list[str]] = {}
    for wid, es in by_worker.items():
        all_ids: list[str] = []
        for e in es:
            if e.get("tool_name") != "propose_query":
                continue
            q_lucene = (e.get("result") or {}).get("query_lucene") or ""
            if not q_lucene:
                # fallback: if only query_nl was logged (older worker), re-translate
                q_nl = (e.get("result") or {}).get("query", "")
                from citeclaw.agents.v3.query_translate import to_lucene
                q_lucene = to_lucene(q_nl)
            token = None
            while len(all_ids) < 10000:
                try:
                    resp = s2.search_bulk(query=q_lucene, limit=500, token=token)
                except Exception:
                    break
                data = resp.get("data") or []
                for row in data:
                    pid = row.get("paperId")
                    if pid and pid not in all_ids:
                        all_ids.append(pid)
                token = resp.get("token")
                if not token or not data:
                    break
        worker_ids[wid] = all_ids[:10000]

    print("\n# paper counts")
    for w, ids in worker_ids.items():
        print(f"  {w}: {len(ids)} papers")
    union: list[str] = []
    seen: set[str] = set()
    for w, ids in worker_ids.items():
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                union.append(pid)
    print(f"UNION (deduped): {len(union)}")

    enrich_n = min(2000, len(union))
    enriched = s2.enrich_batch([{"paper_id": pid} for pid in union[:enrich_n]])
    print(f"Enriched: {len(enriched)}")

    enriched_sorted = sorted(enriched, key=lambda r: -(r.citation_count or 0))
    print("\n# top-20 cited")
    for r in enriched_sorted[:20]:
        cc = r.citation_count or 0
        yr = r.year or "?"
        print(f"  [{cc:5d}c {yr}] {(r.title or '')[:95]}")

    paper_dicts = [
        {"paperId": r.paper_id, "title": r.title or "", "abstract": r.abstract or "",
         "citationCount": r.citation_count or 0}
        for r in enriched
    ]
    clusters = topic_model(paper_dicts, s2_client=s2)
    print("\n# topic clusters (UMAP+HDBSCAN on SPECTER2, size_factor=0.5)")
    for c in clusters:
        kw = ", ".join(c.keywords[:6])
        print(f"  cluster {c.cluster_id} ({c.count}p): {kw}")
        for t in c.representative_titles[:1]:
            print(f"      · {t[:90]}")

    if done_summary:
        print(f"\n# done summary\n  {done_summary[:500]}")


if __name__ == "__main__":
    main()
