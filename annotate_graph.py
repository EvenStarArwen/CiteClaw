#!/usr/bin/env python3
"""
annotate_graph.py
-----------------
Read a CiteClaw GraphML file, use an LLM to generate a concise label for each
node from paper metadata (title + abstract), and write the annotated graph back.

Usage:
    python annotate_graph.py data_bio/citation_network.graphml \
        -c config_bio.yaml \
        -i "use the 1-word model name for the label (e.g., AlphaFold3, RNAErine, ProGen2, DNABERT)"

    # instruction is optional — if omitted, labels default to paper titles
    python annotate_graph.py data_bio/citation_network.graphml -c config_bio.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Regex to strip any leftover <think>...</think> blocks (safety net in case
# the OSS server isn't running with --reasoning-parser).
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _custom_reasoning_kwargs(reasoning_effort: str) -> dict[str, Any]:
    """Map unified ``reasoning_effort`` to OSS thinking controls.

    - ``""``: no overrides (server default)
    - ``"off"`` / ``"none"``: disable thinking
    - ``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"``: enable thinking +
      forward effort level to servers that honor it natively.
    """
    e = (reasoning_effort or "").strip().lower()
    if not e:
        return {}
    if e in ("off", "none", "false", "disable", "disabled"):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {
        "reasoning_effort": e,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def _normalize_base_url(url: str) -> str:
    """Ensure the base_url ends with '/v1' so the OpenAI SDK targets vLLM correctly."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url

import igraph as ig
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are labelling a paper in a citation network.\n"
    "Generate a concise label based on the instruction.\n"
    "Reply with ONLY the label text, nothing else — no quotes, no explanation."
)

_USER = (
    "Instruction: {instruction}\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}\n\n"
    "Label:"
)


import threading

_total_input_tokens = 0
_total_output_tokens = 0
_total_reasoning_tokens = 0
_total_calls = 0
# Lock protects the four global token counters above and also serializes
# the per-paper progress print in the concurrent loop below, so output
# doesn't get interleaved when multiple workers finish at once.
_stats_lock = threading.Lock()

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_openai_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


_gemini_client = None  # reused across calls


def _get_gemini_client(api_key: str):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _call_llm_gemini(
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    reasoning_effort: str = "",
    max_retries: int = 3,
) -> str:
    from google.genai import types

    global _total_input_tokens, _total_output_tokens, _total_reasoning_tokens, _total_calls
    client = _get_gemini_client(api_key)
    for attempt in range(max_retries):
        try:
            gen_config: dict[str, Any] = {
                "temperature": 0.0,
                "system_instruction": system,
            }
            if reasoning_effort:
                gen_config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort,
                )
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(**gen_config),
            )
            parts = (resp.candidates[0].content.parts if resp.candidates else []) or []
            text_parts = [
                p.text for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            ]
            text = "\n".join(text_parts) if text_parts else (getattr(resp, "text", "") or "")
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                with _stats_lock:
                    _total_input_tokens += getattr(um, "prompt_token_count", 0) or 0
                    _total_output_tokens += getattr(um, "candidates_token_count", 0) or 0
                    _total_reasoning_tokens += getattr(um, "thinking_token_count", 0) or 0
                    _total_calls += 1
            return text.strip()
        except Exception as e:
            console.print(f"[red]gemini call failed (attempt {attempt+1}): {type(e).__name__}: {e}[/]")
            time.sleep(2 ** attempt)
    return ""


def _call_llm_openai(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    reasoning_effort: str = "",
    is_custom: bool = False,
    max_retries: int = 3,
) -> str:
    global _total_input_tokens, _total_output_tokens, _total_reasoning_tokens, _total_calls
    is_reasoning = _is_openai_reasoning(model)
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # OpenAI o-series reasoning models don't support temperature.
            # Everything else (including OSS via vLLM) is fine with temp=0.
            if not is_reasoning:
                kwargs["temperature"] = 0.0
            if is_custom:
                # Custom endpoint: map reasoning_effort to OSS thinking controls
                kwargs.update(_custom_reasoning_kwargs(reasoning_effort))
            elif reasoning_effort and is_reasoning:
                kwargs["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**kwargs)
            usage = resp.usage
            if usage:
                with _stats_lock:
                    _total_input_tokens += usage.prompt_tokens
                    _total_output_tokens += usage.completion_tokens
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        _total_reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
                    _total_calls += 1
            text = (resp.choices[0].message.content or "").strip()
            if is_custom:
                text = _strip_think_tags(text)
            return text
        except openai.RateLimitError:
            time.sleep(2**attempt)
    return ""


def label_paper(
    *,
    model: str,
    instruction: str,
    title: str,
    abstract: str,
    api_key: str,
    reasoning_effort: str = "",
    openai_client: openai.OpenAI | None = None,
    is_custom: bool = False,
) -> str:
    """Generate a concise label for one paper."""
    user_msg = _USER.format(
        instruction=instruction,
        title=title,
        abstract=abstract or "(no abstract)",
    )
    # Custom endpoints (vLLM, Modal, etc.) go through the OpenAI SDK path
    # even if the model name happens to start with "gemini-".
    if model.startswith("gemini-") and not is_custom:
        label = _call_llm_gemini(api_key, model, _SYSTEM, user_msg, reasoning_effort=reasoning_effort)
    else:
        assert openai_client is not None
        label = _call_llm_openai(
            openai_client,
            model,
            _SYSTEM,
            user_msg,
            reasoning_effort=reasoning_effort,
            is_custom=is_custom,
        )
    # Strip quotes if the model wraps the label
    label = label.strip('"\'').strip()
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def annotate(
    graph_path: Path,
    output_path: Path,
    instruction: str | None,
    api_key: str,
    model: str = "gpt-5.4-nano",
    reasoning_effort: str = "",
    base_url: str = "",
    request_timeout: float = 60.0,
    limit: int | None = None,
) -> None:
    console.print(f"[bold]Loading graph:[/] {graph_path}")
    g = ig.Graph.Read_GraphML(str(graph_path))
    console.print(f"  {g.vcount()} nodes, {g.ecount()} edges")

    if not instruction:
        # No instruction — use title as label (truncated)
        console.print("  No instruction provided — using paper titles as labels")
        g.vs["label"] = [
            (v["title"][:40] if "title" in v.attributes() else v.get("label", "?"))
            for v in g.vs
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.write_graphml(str(output_path))
        console.print(f"[bold green]✓[/] Graph saved to {output_path}")
        return

    is_custom = bool(base_url)

    # Create client:
    #   - custom endpoint (vLLM/Modal/etc.) → OpenAI SDK with base_url
    #   - Gemini                            → native google-genai SDK (no OpenAI client needed)
    #   - everything else                   → standard OpenAI
    openai_client: openai.OpenAI | None = None
    if is_custom:
        normalized = _normalize_base_url(base_url)
        openai_client = openai.OpenAI(
            api_key=api_key or "none",
            base_url=normalized,
            timeout=request_timeout,
        )
        console.print(f"[dim]  Using custom endpoint: {normalized}[/]")
    elif not model.startswith("gemini-"):
        openai_client = openai.OpenAI(api_key=api_key)

    total = g.vcount()
    n_to_label = min(limit, total) if limit else total
    if limit and limit < total:
        console.print(
            f"[bold]Labelling first {n_to_label}/{total} papers[/] (--limit active; remaining keep titles)"
        )
    else:
        console.print(f"[bold]Labelling {total} papers[/] (instruction: {instruction[:60]})")
    if reasoning_effort:
        console.print(f"[dim]  reasoning_effort = {reasoning_effort}[/]")

    # Pre-extract (title, abstract) for every node so the worker threads
    # don't touch igraph vertex objects concurrently (igraph isn't thread-safe
    # for reads during this kind of access pattern).
    nodes = []
    for v in g.vs:
        title = v["title"] if "title" in v.attributes() else v.get("label", "")
        abstract = v["abstract"] if "abstract" in v.attributes() else ""
        nodes.append((title, abstract))

    # Results indexed by vertex id so we can write them back in order at
    # the end. Pre-populate with fallback labels for nodes past the limit.
    labels: list[str] = [(t or "?")[:40] for t, _ in nodes]

    # Concurrency: the Modal vLLM server is configured for up to 256
    # concurrent inputs (`@modal.concurrent(max_inputs=256)`), and measured
    # KV cache on 2× B200 with Qwen3.5-122B-A10B-FP8 fits ~370 parallel
    # 16k-context sequences. 64 workers gives good batch utilization
    # without risking HTTP client timeouts from queueing. If you see
    # 429/timeouts, drop this to 32 or 16.
    max_workers = 64

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _label_one(idx: int) -> tuple[int, str]:
        title, abstract = nodes[idx]
        lbl = label_paper(
            model=model,
            instruction=instruction,
            title=title,
            abstract=abstract,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            openai_client=openai_client,
            is_custom=is_custom,
        )
        return idx, lbl

    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_label_one, i) for i in range(n_to_label)]
        for fut in as_completed(futures):
            idx, lbl = fut.result()
            labels[idx] = lbl
            title = nodes[idx][0]
            with _stats_lock:
                done_count += 1
                console.print(
                    f"  [dim][{done_count}/{n_to_label}][/] {title[:60]}  →  [bold]{lbl}[/]"
                )

    # Preserve original title, set label
    if "title" in g.vs.attributes():
        g.vs["original_title"] = g.vs["title"]
    g.vs["label"] = labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(output_path))
    console.print(f"[bold green]✓[/] Annotated graph saved to {output_path}")

    # Show sample
    console.print("\n[bold]Sample labels:[/]")
    ranked = sorted(
        range(g.vcount()),
        key=lambda i: g.vs[i]["citation_count"] if "citation_count" in g.vs.attributes() else 0,
        reverse=True,
    )
    for i in ranked[:10]:
        v = g.vs[i]
        orig = v["original_title"][:45] if "original_title" in v.attributes() else "?"
        console.print(f"  [bold]{v['label']:<25}[/]  [dim]← {orig}[/]")

    # Token usage summary
    total = _total_input_tokens + _total_output_tokens
    reasoning_part = f", {_total_reasoning_tokens:,} reasoning" if _total_reasoning_tokens else ""
    console.print(
        f"\n[dim]Token usage: {total / 1_000_000:.3f}M total "
        f"({_total_input_tokens:,} in + {_total_output_tokens:,} out{reasoning_part}, "
        f"{_total_calls} calls)[/]"
    )


def main():
    parser = argparse.ArgumentParser(description="Annotate citation graph nodes with LLM-generated labels.")
    parser.add_argument("graph", type=Path, help="Input GraphML file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output GraphML (default: <input>_annotated.graphml)")
    parser.add_argument("-i", "--instruction", type=str, default=None,
                        help='Labelling instruction (optional). If omitted, uses paper titles.')
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help="Config YAML (for API key and model)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides config)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only label the first N nodes (for testing). Remaining nodes keep their titles as labels.",
    )
    args = parser.parse_args()

    # Resolve API key, model, and reasoning effort
    api_key = args.api_key
    model = args.model or "gpt-5.4-nano"
    reasoning_effort = ""
    base_url = ""
    request_timeout = 60.0

    instruction = args.instruction

    if args.config and args.config.exists():
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        # API keys MUST NOT appear in YAML — reject loudly if they do.
        _forbidden = {
            "openai_api_key", "gemini_api_key", "s2_api_key", "llm_api_key",
            "OPENAI_API_KEY", "GEMINI_API_KEY", "S2_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY", "LLM_API_KEY",
        }
        _leaked = [k for k in cfg if k in _forbidden]
        if _leaked:
            console.print(
                "[red]Error:[/] API keys must not be set in config YAML "
                f"(found: {', '.join(sorted(_leaked))}). Remove these fields "
                "and set environment variables instead (OPENAI_API_KEY, "
                "GEMINI_API_KEY, S2_API_KEY)."
            )
            sys.exit(1)
        if not args.model:
            model = cfg.get("screening_model", model)
        base_url = cfg.get("llm_base_url", "") or ""
        request_timeout = float(cfg.get("llm_request_timeout", 60.0))
        if not instruction:
            instruction = cfg.get("graph_label_instruction", "")
        reasoning_effort = cfg.get("reasoning_effort", "")

    # API keys come from env vars only.
    if not api_key:
        if base_url:
            api_key = "none"  # most self-hosted servers accept any token
        elif model.startswith("gemini-"):
            api_key = os.environ.get("GEMINI_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key and instruction:
        console.print(
            "[red]Error:[/] API key needed for LLM labelling. Pass --api-key, -c config.yaml, or set the appropriate env var."
        )
        sys.exit(1)

    output = args.output or args.graph.with_name(args.graph.stem + "_annotated.graphml")

    annotate(
        graph_path=args.graph,
        output_path=output,
        instruction=instruction or None,
        api_key=api_key or "",
        model=model,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        request_timeout=request_timeout,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
