"""Boolean formula DSL: ``(q1 | q2) & !q3`` over named YES/NO queries.

Standalone — no LLM, no config dependency. Tokenizer + recursive-descent
parser + AST evaluator + a tiny ``FormulaError`` exception type.
"""

from __future__ import annotations

import re

# AST node = tuple of (op, ...children)
ASTNode = tuple


class FormulaError(Exception):
    """Raised when a Boolean formula cannot be parsed or evaluated."""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<NAME>[A-Za-z_]\w*)"
    r"|(?P<OP>[&|!()])"
    r")\s*"
)


def tokenize(expr: str) -> list[tuple[str, str]]:
    """Split a formula into ``[(kind, value), ...]`` tokens."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise FormulaError(f"Unexpected character at position {pos}: '{expr[pos:]}'")
        if m.group("NAME"):
            tokens.append(("NAME", m.group("NAME")))
        elif m.group("OP"):
            tokens.append(("OP", m.group("OP")))
        pos = m.end()
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> ASTNode:
        ast = self._parse_or()
        if self._pos < len(self._tokens):
            raise FormulaError(f"Unexpected token after complete parse: {self._tokens[self._pos:]}")
        return ast

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: str | None = None) -> tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise FormulaError("Unexpected end of expression")
        if expected is not None and tok[1] != expected:
            raise FormulaError(f"Expected '{expected}', got '{tok[1]}'")
        self._pos += 1
        return tok

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._peek() == ("OP", "|"):
            self._consume("|")
            left = ("or", left, self._parse_and())
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_unary()
        while self._peek() == ("OP", "&"):
            self._consume("&")
            left = ("and", left, self._parse_unary())
        return left

    def _parse_unary(self) -> ASTNode:
        if self._peek() == ("OP", "!"):
            self._consume("!")
            return ("not", self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> ASTNode:
        tok = self._peek()
        if tok is None:
            raise FormulaError("Unexpected end of expression")
        if tok[0] == "NAME":
            self._consume()
            return ("name", tok[1])
        if tok == ("OP", "("):
            self._consume("(")
            inner = self._parse_or()
            self._consume(")")
            return inner
        raise FormulaError(f"Unexpected token: {tok}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BooleanFormula:
    """Parsed Boolean formula. Operators: ``&`` AND, ``|`` OR, ``!`` NOT."""

    def __init__(self, expression: str) -> None:
        self._raw = expression
        self._ast = _Parser(tokenize(expression)).parse()

    def query_names(self) -> set[str]:
        """All leaf names referenced anywhere in the formula."""
        out: set[str] = set()
        self._collect(self._ast, out)
        return out

    def _collect(self, node: ASTNode, out: set[str]) -> None:
        kind = node[0]
        if kind == "name":
            out.add(node[1])
        elif kind == "not":
            self._collect(node[1], out)
        elif kind in ("and", "or"):
            self._collect(node[1], out)
            self._collect(node[2], out)

    def evaluate(self, values: dict[str, bool]) -> bool:
        return self._eval(self._ast, values)

    def _eval(self, node: ASTNode, values: dict[str, bool]) -> bool:
        kind = node[0]
        if kind == "name":
            return values.get(node[1], False)
        if kind == "not":
            return not self._eval(node[1], values)
        if kind == "and":
            return self._eval(node[1], values) and self._eval(node[2], values)
        if kind == "or":
            return self._eval(node[1], values) or self._eval(node[2], values)
        raise FormulaError(f"Unknown node type: {kind}")

    def __repr__(self) -> str:
        return f"BooleanFormula({self._raw!r})"
