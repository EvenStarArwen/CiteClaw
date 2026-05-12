"""Boxed-flow-chart renderer for citeclaw pipeline previews.

Layout grammar:

  - Linear steps stack vertically, each in a ``┌─name─┐`` box, connected
    by ``│ … ▼`` arrows. Box position is centered on the page axis.
  - The right side of each box's middle line carries an optional
    description (one-line summary of key params).
  - ``Parallel`` opens with its own box + ``╱ ╲`` divergence to two
    sub-columns labelled ``Branch i``. Each branch is a stack of boxes.
    Asymmetric branches (e.g. ``Rerank → ExpandForward`` vs. just
    ``ExpandBackward``) are bottom-aligned so the merge line works,
    with the shorter branch padded by a long vertical pipe.
  - Merge: ``└── ∪ ──┘`` with the union symbol and a
    ``union by paper_id`` side annotation; the merged signal flows
    back into the page axis and continues.

Pure Unicode box-drawing characters; no Rich, no terminal hacks.
"""
from __future__ import annotations

from citeclaw.preview.model import StepNode


def _box(name: str) -> list[str]:
    """Three-line ``┌─name─┐ / │ name │ / └─────┘`` box."""
    inner = " " + name + " "
    bar = "─" * len(inner)
    return [
        "┌" + bar + "┐",
        "│" + inner + "│",
        "└" + bar + "┘",
    ]


def _center_block(block: list[str], center: int) -> list[str]:
    width = len(block[0])
    left_pad = max(0, center - width // 2)
    return [" " * left_pad + line for line in block]


def _pipe(center: int) -> str:
    return " " * center + "│"


def _arrow(center: int) -> str:
    return " " * center + "▼"


def _annotated(line: str, annotation: str) -> str:
    return line + "   " + annotation if annotation else line


def _lookup(params: list[tuple[str, str]], key: str) -> str:
    for k, v in params:
        if k == key:
            return v
    return ""


def _step_description(node: StepNode) -> str:
    """One-line side annotation for a step. Per-step formatting keeps
    things terse so annotations fit alongside the box."""
    n = node.name
    p = node.params

    if n in ("LoadSeeds", "ResolveSeeds", "Finalize", "MergeDuplicates"):
        return ""

    if n == "ExpandForward":
        v = _lookup(p, "max_citations")
        return f"n={v}" if v else ""

    if n == "ExpandBackward":
        v = _lookup(p, "pdf_refs")
        return f"pdf={v}" if v else ""

    if n == "ExpandBySearch":
        bits = []
        if (v := _lookup(p, "reasoning")):
            bits.append(f"reasoning={v}")
        if (v := _lookup(p, "max_papers")):
            bits.append(f"max={v}")
        return ", ".join(bits)

    if n == "ExpandBySemantics":
        mode = _lookup(p, "mode") or "?"
        recs = _lookup(p, "recs")
        limit = _lookup(p, "limit")
        if recs:
            return f"{mode}, recs={recs}"
        if limit:
            return f"{mode}, limit={limit}"
        return mode

    if n == "Rerank":
        bits = []
        if (v := _lookup(p, "metric")):
            bits.append(v)
        if (v := _lookup(p, "k")):
            bits.append(f"k={v}")
        if _lookup(p, "diversity"):
            bits.append("+div")
        return ", ".join(bits)

    if n == "Cluster":
        algo = _lookup(p, "algorithm")
        store = _lookup(p, "store_as")
        if algo and store:
            return f"{algo} → {store}"
        return algo or store

    if n == "ExpandByAuthor":
        bits = []
        if (v := _lookup(p, "top_k_authors")):
            bits.append(f"top_k={v}")
        if (v := _lookup(p, "author_metric")):
            bits.append(v)
        return ", ".join(bits)

    if n == "Parallel":
        v = _lookup(p, "branches")
        return f"broadcast snapshot to {v} branches" if v else ""

    return ", ".join(f"{k}={v}" for k, v in p[:2])


def _render_linear_step(
    node: StepNode, center: int, max_annotation_col: int,
) -> list[str]:
    """A non-Parallel step + side annotation, truncated to fit
    within ``max_annotation_col`` (so it doesn't overflow into a
    sibling branch's column)."""
    block = _center_block(_box(node.name), center)
    desc = _step_description(node)
    if desc:
        free = max_annotation_col - len(block[1]) - 3
        if free <= 0:
            desc = ""
        elif len(desc) > free:
            desc = desc[: free - 1] + "…"
    if desc:
        block[1] = _annotated(block[1], desc)
    return block


def _render_branch(
    branch: list[StepNode], center: int, max_annotation_col: int,
) -> list[str]:
    lines: list[str] = []
    for i, step in enumerate(branch):
        if i > 0:
            lines.append(_pipe(center))
            lines.append(_arrow(center))
        for raw in _render_linear_step(step, center, max_annotation_col):
            lines.append(raw)
    return lines


def _two_col_line(
    left_c: int, right_c: int, left_ch: str, right_ch: str, width: int,
) -> str:
    line = [" "] * (width + 1)
    if 0 <= left_c < len(line):
        line[left_c] = left_ch
    if 0 <= right_c < len(line):
        line[right_c] = right_ch
    return "".join(line)


def _overlay(base: str, top: str) -> str:
    """Character-wise combine; non-space chars in ``top`` win."""
    n = max(len(base), len(top))
    out = []
    for i in range(n):
        b = base[i] if i < len(base) else " "
        t = top[i] if i < len(top) else " "
        out.append(t if t != " " else b)
    return "".join(out)


def _merge_connector(left_c: int, right_c: int, width: int) -> str:
    line = [" "] * (width + 1)
    line[left_c] = "└"
    line[right_c] = "┘"
    for c in range(left_c + 1, right_c):
        line[c] = "─"
    mid = (left_c + right_c) // 2
    line[mid - 1] = " "
    line[mid] = "∪"
    line[mid + 1] = " "
    return "".join(line)


def _render_parallel(
    node: StepNode, page_width: int, page_center: int,
) -> list[str]:
    out: list[str] = []
    parallel_box = _center_block(_box(node.name), page_center)
    out.extend(parallel_box)

    n = len(node.branches)
    if n == 0:
        return out

    # Single-branch Parallel collapses to a labelled column.
    if n == 1:
        out.append(_pipe(page_center))
        out.append(_arrow(page_center))
        out.extend(_render_branch(node.branches[0], page_center, page_width))
        return out

    half = max(15, page_width // 5)
    left_center = max(8, page_center - half)
    right_center = min(page_width - 8, page_center + half)

    # ╱ … ╲ divergence + branch labels + initial arrows.
    div = [" "] * (page_width + 1)
    div[left_center + 1] = "╱"
    div[right_center - 1] = "╲"
    out.append("".join(div))

    labels = [" "] * (page_width + 1)
    for i, ch in enumerate("Branch 1"):
        col = left_center - 4 + i
        if 0 <= col < len(labels):
            labels[col] = ch
    for i, ch in enumerate("Branch 2"):
        col = right_center - 4 + i
        if 0 <= col < len(labels):
            labels[col] = ch
    out.append("".join(labels))

    out.append(_two_col_line(left_center, right_center, "│", "│", page_width))
    out.append(_two_col_line(left_center, right_center, "▼", "▼", page_width))

    left_lines = _render_branch(
        node.branches[0], left_center, max_annotation_col=right_center - 10,
    )
    right_lines = _render_branch(
        node.branches[1], right_center, max_annotation_col=page_width,
    )

    # Bottom-align asymmetric branches by top-padding the shorter one
    # with vertical pipes — the long arrow communicates "this branch's
    # first action happens later than the sibling's".
    h = max(len(left_lines), len(right_lines))
    if len(left_lines) < h:
        left_lines = [_pipe(left_center)] * (h - len(left_lines)) + left_lines
    if len(right_lines) < h:
        right_lines = [_pipe(right_center)] * (h - len(right_lines)) + right_lines

    for i in range(h):
        l = left_lines[i].ljust(left_center + 24)
        r_line = right_lines[i]
        if r_line.strip():
            out.append(_overlay(l, r_line))
        else:
            out.append(l)

    # rstrip first so the annotation sits right after the ┘ rather than
    # being pushed out by the connector's trailing padding spaces.
    merge_line = _merge_connector(left_center, right_center, page_width).rstrip()
    merge_line = merge_line + "   union by paper_id"
    out.append(merge_line)

    return out


def render(
    nodes: list[StepNode],
    width: int = 100,
) -> str:
    """Render the full pipeline node tree as a multi-line ASCII diagram."""
    center = width // 2
    lines: list[str] = []
    for i, node in enumerate(nodes):
        if i > 0:
            lines.append(_pipe(center))
            lines.append(_arrow(center))
        if node.branches:
            lines.extend(_render_parallel(node, width, center))
        else:
            lines.extend(_render_linear_step(node, center, max_annotation_col=width))
    return "\n".join(line.rstrip() for line in lines)
