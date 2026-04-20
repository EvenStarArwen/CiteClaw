"""Annotate a CiteClaw GraphML file with LLM-generated node labels.

Integrated as the ``annotate`` subcommand:
    python -m citeclaw annotate <graphml> [-c config.yaml] [-i instruction] [-o out]

When ``screening_model`` is ``stub``, no real LLM is called — labels are
deterministic stub strings. All real provider routing (OpenAI / custom
endpoint / Gemini) goes through the standard
:func:`citeclaw.clients.llm.build_llm_client` factory so this module no
longer carries its own provider sniffing logic.

Batched LLM dispatch: ``annotate_graph`` packs up to
``ctx.config.llm_batch_size`` papers per LLM call with a JSON schema
constraining the response to one label per input index. Falls back to
per-paper ``_label_one`` on parse failure so a single malformed batch
doesn't wipe out the rest.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from citeclaw.clients.llm import build_llm_client
from citeclaw.budget import BudgetTracker
from citeclaw.models import BudgetExhaustedError
from citeclaw.progress import console
from citeclaw.prompts.annotation import BATCH_SYSTEM as _BATCH_SYSTEM
from citeclaw.prompts.annotation import BATCH_USER_TEMPLATE as _BATCH_USER
from citeclaw.prompts.annotation import PAPER_BLOCK_TEMPLATE as _PAPER_BLOCK
from citeclaw.prompts.annotation import SYSTEM as _SYSTEM
from citeclaw.prompts.annotation import USER_TEMPLATE as _USER

log = logging.getLogger("citeclaw.annotate")


def _annotation_batch_schema() -> dict[str, Any]:
    """JSON Schema for a batched annotation response.

    Shape: ``{"results": [{"index": int, "label": str}, ...]}``
    """
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["index", "label"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def _clean_label(text: str) -> str:
    """Normalise a model's label output: strip quotes, whitespace, newlines."""
    return (text or "").strip().strip('"\'').strip().replace("\n", " ")


def _parse_annotation_batch(raw: str) -> dict[int, str] | None:
    """Parse a batched annotation response into ``{index: label}``.

    Accepts the structured-output shape ``{"results": [{"index": int,
    "label": str}, ...]}`` and the bare-array shape for providers that
    drop the wrapper. Returns ``None`` on parse failure so the caller
    can fall back to per-paper single-call dispatch.
    """
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except Exception:
        return None
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return None
    out: dict[int, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if not isinstance(idx, (int, float)) or isinstance(idx, bool):
            continue
        label = _clean_label(str(entry.get("label") or ""))
        if not label:
            continue
        out[int(idx)] = label
    return out if out else None


def _label_one(client, instruction: str, title: str, abstract: str) -> str:
    """Single-paper labelling round-trip through the unified LLMClient.

    Kept for legacy callers and as the per-paper fallback when a batch
    response is unparseable or missing entries.
    """
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
    return _clean_label(resp.text)


def _label_batch(
    client,
    instruction: str,
    items: list[tuple[int, str, str]],
) -> dict[int, str]:
    """Label a batch of papers in a single LLM call.

    ``items`` is a list of ``(index, title, abstract)`` tuples. Returns
    ``{index: label}`` covering whichever entries the LLM named. Missing
    indices (parse failure, omitted entries, budget exhaustion) come
    back empty; the caller falls back to ``_label_one`` for them.
    """
    if not items:
        return {}
    block = "\n".join(
        _PAPER_BLOCK.format(
            idx=idx, title=title, abstract=abstract or "(no abstract)",
        )
        for idx, title, abstract in items
    )
    user = _BATCH_USER.format(instruction=instruction, n=len(items), papers=block)
    schema = _annotation_batch_schema()
    try:
        resp = client.call(
            _BATCH_SYSTEM, user,
            category="annotate",
            response_schema=schema,
        )
    except BudgetExhaustedError:
        return {}
    except TypeError:
        # Legacy fake clients without response_schema.
        try:
            resp = client.call(_BATCH_SYSTEM, user, category="annotate")
        except BudgetExhaustedError:
            return {}
        except Exception as exc:
            log.warning("annotate batched LLM call failed: %s", exc)
            return {}
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("annotate batched LLM call failed: %s", exc)
        return {}
    parsed = _parse_annotation_batch(resp.text or "")
    return parsed or {}


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

    # Batched dispatch: pack batch_size papers per LLM call. Entries the
    # batch misses (parse failure, omitted by the model) fall back to
    # per-paper ``_label_one`` so a single bad batch doesn't blank out
    # many rows.
    batch_size = max(1, cfg.llm_batch_size)
    indices = list(range(n_to_label))
    batches: list[list[int]] = [
        indices[i:i + batch_size] for i in range(0, n_to_label, batch_size)
    ]

    def _do_batch(batch: list[int]) -> list[tuple[int, str]]:
        items = [(i, nodes[i][0], nodes[i][1]) for i in batch]
        parsed = _label_batch(client, instruction, items)
        out: list[tuple[int, str]] = []
        for i in batch:
            lbl = parsed.get(i)
            if lbl:
                out.append((i, lbl))
                continue
            # Per-paper fallback — only for the indices the batch missed.
            out.append((i, _label_one(client, instruction, nodes[i][0], nodes[i][1])))
        return out

    max_workers = 1 if is_stub else 16
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_do_batch, b) for b in batches]
        done = 0
        for fut in as_completed(futs):
            for idx, lbl in fut.result():
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
