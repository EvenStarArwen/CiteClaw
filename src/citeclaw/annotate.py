"""Annotate a CiteClaw GraphML file with LLM-generated node labels.

Integrated as the ``annotate`` subcommand:
    python -m citeclaw annotate <graphml> [-c config.yaml] [-i instruction] [-o out]

When ``screening_model`` is ``stub``, no real LLM is called — labels are
deterministic stub strings. All real provider routing (OpenAI / custom
endpoint / Gemini) goes through the standard
:func:`citeclaw.clients.llm.build_llm_client` factory so this module no
longer carries its own provider sniffing logic.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from citeclaw.clients.llm import build_llm_client
from citeclaw.config import BudgetTracker
from citeclaw.models import BudgetExhaustedError
from citeclaw.progress import console
from citeclaw.prompts.annotation import SYSTEM as _SYSTEM
from citeclaw.prompts.annotation import USER_TEMPLATE as _USER

log = logging.getLogger("citeclaw.annotate")


def _label_one(client, instruction: str, title: str, abstract: str) -> str:
    """Single-paper labelling round-trip through the unified LLMClient."""
    user = _USER.format(
        instruction=instruction,
        title=title,
        abstract=abstract or "(no abstract)",
    )
    try:
        resp = client.call(_SYSTEM, user, category="annotate")
    except BudgetExhaustedError:
        return ""
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("annotate LLM call failed: %s", exc)
        return ""
    return (resp.text or "").strip().strip('"\'').strip()


def annotate_graph(
    *,
    graph_path: Path,
    output_path: Path,
    instruction: str | None,
    config_path: Path | None,
    api_key: str | None = None,
    model_override: str | None = None,
    limit: int | None = None,
) -> None:
    import igraph as ig

    # Resolve config: use CiteClaw's own loader so the same env vars apply
    from citeclaw.config import load_settings

    cfg = load_settings(config_path, overrides={}) if config_path else None
    if cfg is None:
        # No config: build a minimal Settings stub so build_llm_client has
        # something to dispatch on. Default to the stub backend.
        cfg = load_settings(None, overrides={"screening_model": "stub"})
    if model_override:
        # Override the model on the resolved settings.
        object.__setattr__(cfg, "screening_model", model_override)
    if api_key:
        object.__setattr__(cfg, "openai_api_key", api_key)

    if not instruction:
        instruction = cfg.graph_label_instruction

    console.print(f"[bold]Loading graph:[/] {graph_path}")
    g = ig.Graph.Read_GraphML(str(graph_path))
    console.print(f"  {g.vcount()} nodes, {g.ecount()} edges")

    if not instruction:
        console.print("  No instruction provided — using paper titles as labels")
        g.vs["label"] = [
            (v["title"][:40] if "title" in v.attributes() else v.get("label", "?"))
            for v in g.vs
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.write_graphml(str(output_path))
        console.print(f"[bold green]✓[/] Graph saved to {output_path}")
        return

    # Build the unified LLMClient. Annotation is its own short-lived run, so
    # the budget tracker is fresh — it doesn't share with any pipeline run.
    budget = BudgetTracker()
    client = build_llm_client(cfg, budget)
    is_stub = cfg.screening_model.strip().lower() == "stub"

    n_total = g.vcount()
    n_to_label = min(limit, n_total) if limit else n_total
    console.print(
        f"[bold]Labelling {n_to_label}/{n_total} papers[/] "
        f"(model={cfg.screening_model}, instruction: {instruction[:60]})"
    )

    # IMPORTANT: never expose the node id (or anything derived from it, e.g. the
    # graphml `label` attribute, which some pipelines populate with the paper id)
    # to the LLM. If the model can't find a real name in title+abstract it will
    # happily regurgitate the id as the label.
    nodes: list[tuple[str, str]] = []
    for v in g.vs:
        title = v["title"] if "title" in v.attributes() else ""
        abstract = v["abstract"] if "abstract" in v.attributes() else ""
        nodes.append((title or "", abstract or ""))

    labels: list[str] = [(t or "?")[:40] for t, _ in nodes]

    def _do_one(idx: int) -> tuple[int, str]:
        title, abstract = nodes[idx]
        return idx, _label_one(client, instruction, title, abstract)

    max_workers = 1 if is_stub else 16
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_do_one, i) for i in range(n_to_label)]
        done = 0
        for fut in as_completed(futs):
            idx, lbl = fut.result()
            if lbl:
                labels[idx] = lbl
            done += 1
            console.print(
                f"  [dim][{done}/{n_to_label}][/] {nodes[idx][0][:60]}  →  [bold]{lbl}[/]"
            )

    if "title" in g.vs.attributes():
        g.vs["original_title"] = g.vs["title"]
    g.vs["label"] = labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(output_path))
    console.print(f"[bold green]✓[/] Annotated graph saved to {output_path}")
