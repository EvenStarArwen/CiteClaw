"""Structural parser for Semantic Scholar bulk-search query strings.

S2's bulk-search query text is a small Lucene-flavoured DSL: quoted
phrases, parenthesised groups, and three boolean operators — ``+``
for AND, ``|`` for OR, and ``-`` for NOT. Our workers *write* these
queries directly in their tool-args; this module lets downstream
tools (notably :func:`query_diagnostics`) *read* them back as a tree
so we can reason about per-branch contribution.

Two callers exist today:

- :func:`enumerate_or_leaves` — for each OR group in the query, yields
  one substituted query per leaf where that OR group is replaced by
  just that leaf. The diagnostics tool issues a count-only S2 call on
  each substituted query so the agent sees how much each OR branch
  contributes *in the context of the rest of the query*.
- :func:`tree_signature` — a stable string summary ("4 OR leaves
  across 2 groups, 1 AND, 0 NOT") the prompt can surface without
  dumping the full AST.

Intentionally *small*: no field-qualified terms (``title:"foo"``), no
phrase-slop (``"foo"~5``), no boost (``^2``). S2's bulk endpoint
accepts those silently but the lint rejects them upstream and the
worker prompt never produces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class QueryNode:
    """One node in the parsed query tree.

    ``kind`` is one of:

    - ``PHRASE`` — a quoted phrase; ``value`` holds the text without quotes.
    - ``WORD``   — a bare (unquoted) token; ``value`` holds the text.
    - ``OR``     — boolean OR over ``children`` (``|``).
    - ``AND``    — boolean AND over ``children`` (``+`` or implicit).
    - ``NOT``    — unary NOT over ``children[0]`` (``-``).
    """

    kind: str
    value: str = ""
    children: list["QueryNode"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


_SPECIAL_CHARS = '()+|-"'


def _tokenize(query: str) -> list[tuple[str, str]]:
    """Turn ``query`` into a list of ``(type, text)`` tuples.

    Token types: ``PHRASE``, ``WORD``, ``LPAREN``, ``RPAREN``, ``PLUS``,
    ``OR``, ``MINUS``. Whitespace is stripped. Unterminated quotes raise
    :class:`ValueError`.
    """
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(query)
    while i < n:
        c = query[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            end = query.find('"', i + 1)
            if end < 0:
                raise ValueError(f"unterminated quote at position {i}")
            tokens.append(("PHRASE", query[i + 1:end]))
            i = end + 1
        elif c == "(":
            tokens.append(("LPAREN", "("))
            i += 1
        elif c == ")":
            tokens.append(("RPAREN", ")"))
            i += 1
        elif c == "+":
            tokens.append(("PLUS", "+"))
            i += 1
        elif c == "|":
            tokens.append(("OR", "|"))
            i += 1
        elif c == "-":
            tokens.append(("MINUS", "-"))
            i += 1
        else:
            # Bare word — consume until the next special char or whitespace.
            j = i
            while j < n and not query[j].isspace() and query[j] not in _SPECIAL_CHARS:
                j += 1
            tokens.append(("WORD", query[i:j]))
            i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------


def parse_query(query: str) -> QueryNode:
    """Parse ``query`` into a :class:`QueryNode` tree.

    Precedence: NOT > AND (``+``) > OR (``|``). Explicit parens always
    override. Bare adjacent terms are parsed as **implicit AND**
    (matching Lucene's default operator), so ``"A" "B"`` parses as
    ``"A" + "B"``. Parse errors raise :class:`ValueError` with a
    location hint.
    """
    tokens = _tokenize(query)
    if not tokens:
        raise ValueError("empty query")
    pos = [0]

    def peek() -> tuple[str, str] | None:
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def consume() -> tuple[str, str]:
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    # Atom-initial tokens that start a new term (implicit-AND continuation).
    _ATOM_STARTS = {"PHRASE", "WORD", "LPAREN", "MINUS"}

    def parse_or() -> QueryNode:
        first = parse_and()
        items = [first]
        while peek() and peek()[0] == "OR":
            consume()
            items.append(parse_and())
        if len(items) == 1:
            return items[0]
        # Flatten nested ORs for a cleaner tree.
        flat: list[QueryNode] = []
        for it in items:
            if it.kind == "OR":
                flat.extend(it.children)
            else:
                flat.append(it)
        return QueryNode("OR", children=flat)

    def parse_and() -> QueryNode:
        first = parse_term()
        items = [first]
        while peek() and peek()[0] in ("PLUS", *_ATOM_STARTS):
            if peek()[0] == "PLUS":
                consume()
            # Implicit AND: adjacent term without explicit `+` is still AND.
            # (S2 accepts both shapes; our serializer emits explicit `+`.)
            items.append(parse_term())
        if len(items) == 1:
            return items[0]
        flat2: list[QueryNode] = []
        for it in items:
            if it.kind == "AND":
                flat2.extend(it.children)
            else:
                flat2.append(it)
        return QueryNode("AND", children=flat2)

    def parse_term() -> QueryNode:
        t = peek()
        if t is None:
            raise ValueError("unexpected end of query")
        kind = t[0]
        if kind == "MINUS":
            consume()
            inner = parse_term()
            return QueryNode("NOT", children=[inner])
        if kind == "PHRASE":
            consume()
            return QueryNode("PHRASE", value=t[1])
        if kind == "WORD":
            consume()
            return QueryNode("WORD", value=t[1])
        if kind == "LPAREN":
            consume()
            inner = parse_or()
            nxt = peek()
            if nxt is None or nxt[0] != "RPAREN":
                raise ValueError("unbalanced parens: missing ')'")
            consume()
            return inner
        raise ValueError(f"unexpected token {kind}: {t[1]!r}")

    tree = parse_or()
    if peek() is not None:
        raise ValueError(f"unexpected trailing token {peek()[1]!r}")
    return tree


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def serialize_query(node: QueryNode, *, top: bool = True) -> str:
    """Render a :class:`QueryNode` tree back to an S2-compatible string.

    The output always passes :func:`citeclaw.agents.s2_query_lint.lint_s2_query`
    when the input tree originated from a valid parse. ``top=False`` adds
    parens around composite nodes so they're safe to nest inside another
    operator.
    """
    if node.kind == "PHRASE":
        return f'"{node.value}"'
    if node.kind == "WORD":
        return node.value
    if node.kind == "NOT":
        inner = serialize_query(node.children[0], top=False)
        return f"-{inner}"
    if node.kind == "OR":
        parts = [serialize_query(c, top=False) for c in node.children]
        inner = " | ".join(parts)
        return inner if top else f"({inner})"
    if node.kind == "AND":
        parts = []
        for i, c in enumerate(node.children):
            s = serialize_query(c, top=False)
            parts.append(s if i == 0 else f"+{s}")
        inner = " ".join(parts)
        return inner if top else f"({inner})"
    raise ValueError(f"unknown node kind {node.kind!r}")


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def _copy(node: QueryNode) -> QueryNode:
    return QueryNode(
        kind=node.kind,
        value=node.value,
        children=[_copy(c) for c in node.children],
    )


def _walk_ors(
    node: QueryNode, path: tuple[int, ...] = (),
) -> Iterator[tuple[tuple[int, ...], QueryNode]]:
    """Yield ``(path, or_node)`` for every OR node in the tree, outermost first."""
    if node.kind == "OR":
        yield (path, node)
    for i, child in enumerate(node.children):
        yield from _walk_ors(child, path + (i,))


def _replace_at_path(
    root: QueryNode, path: tuple[int, ...], new_subtree: QueryNode,
) -> QueryNode:
    """Return a deep copy of ``root`` with the subtree at ``path`` replaced."""
    if not path:
        return _copy(new_subtree)
    new_root = _copy(root)
    parent = new_root
    for idx in path[:-1]:
        parent = parent.children[idx]
    parent.children[path[-1]] = _copy(new_subtree)
    return new_root


@dataclass
class OrLeafSubstitution:
    """One (OR group, leaf) -> substituted query pair."""

    group_id: int
    leaf_idx: int
    leaf_text: str
    substituted_query: str


def enumerate_or_leaves(query: str) -> tuple[QueryNode, list[OrLeafSubstitution]]:
    """For each OR leaf in ``query``, compute the query you'd get by
    replacing its enclosing OR group with just that leaf.

    Returns ``(tree, substitutions)``. The ``substitutions`` list has
    one entry per OR leaf; running :func:`search_bulk` on each
    substituted query gives that leaf's **in-context** hit count (how
    many papers would match if this were the only branch of its OR
    group — with all other parts of the query unchanged).

    Empty return when the query has no OR operators — the caller
    surfaces a "no breakdown available" message instead of issuing
    zero S2 calls.
    """
    tree = parse_query(query)
    substitutions: list[OrLeafSubstitution] = []
    for group_id, (or_path, or_node) in enumerate(_walk_ors(tree)):
        for leaf_idx, leaf in enumerate(or_node.children):
            substituted_tree = _replace_at_path(tree, or_path, leaf)
            substitutions.append(OrLeafSubstitution(
                group_id=group_id,
                leaf_idx=leaf_idx,
                leaf_text=serialize_query(leaf),
                substituted_query=serialize_query(substituted_tree),
            ))
    return tree, substitutions


def tree_signature(tree: QueryNode) -> dict[str, int]:
    """One-shot summary of the tree's structural shape.

    Keys: ``or_groups``, ``or_leaves`` (sum of branches across all OR
    groups), ``and_nodes``, ``not_nodes``, ``phrases``, ``words``,
    ``depth``. Useful for surfacing "this query has 4 OR leaves
    across 2 groups" in tool responses without dumping the AST.
    """
    sig = {
        "or_groups": 0, "or_leaves": 0, "and_nodes": 0,
        "not_nodes": 0, "phrases": 0, "words": 0, "depth": 0,
    }

    def walk(n: QueryNode, d: int) -> None:
        sig["depth"] = max(sig["depth"], d)
        if n.kind == "OR":
            sig["or_groups"] += 1
            sig["or_leaves"] += len(n.children)
        elif n.kind == "AND":
            sig["and_nodes"] += 1
        elif n.kind == "NOT":
            sig["not_nodes"] += 1
        elif n.kind == "PHRASE":
            sig["phrases"] += 1
        elif n.kind == "WORD":
            sig["words"] += 1
        for c in n.children:
            walk(c, d + 1)

    walk(tree, 1)
    return sig


__all__ = [
    "QueryNode",
    "OrLeafSubstitution",
    "parse_query",
    "serialize_query",
    "enumerate_or_leaves",
    "tree_signature",
]
