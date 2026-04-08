"""Tests for the Boolean formula DSL (``citeclaw.screening.formula``)."""

from __future__ import annotations

import pytest

from citeclaw.screening.formula import BooleanFormula, FormulaError, tokenize


class TestTokenize:
    def test_names(self):
        toks = tokenize("q1 q_2 foo_bar")
        assert toks == [("NAME", "q1"), ("NAME", "q_2"), ("NAME", "foo_bar")]

    def test_operators(self):
        toks = tokenize("&|!()")
        kinds = [k for k, _ in toks]
        assert kinds == ["OP"] * 5

    def test_mixed(self):
        toks = tokenize("(q1 | q2) & !q3")
        assert toks == [
            ("OP", "("),
            ("NAME", "q1"),
            ("OP", "|"),
            ("NAME", "q2"),
            ("OP", ")"),
            ("OP", "&"),
            ("OP", "!"),
            ("NAME", "q3"),
        ]

    def test_unexpected_char_raises(self):
        with pytest.raises(FormulaError):
            tokenize("q1 @ q2")


class TestBooleanFormula:
    def test_single_name(self):
        f = BooleanFormula("q1")
        assert f.query_names() == {"q1"}
        assert f.evaluate({"q1": True}) is True
        assert f.evaluate({"q1": False}) is False
        # Unknown name defaults to False.
        assert f.evaluate({}) is False

    def test_and_or_not(self):
        f = BooleanFormula("(q1 | q2) & !q3")
        assert f.query_names() == {"q1", "q2", "q3"}
        assert f.evaluate({"q1": True, "q2": False, "q3": False}) is True
        assert f.evaluate({"q1": True, "q2": True, "q3": True}) is False
        assert f.evaluate({"q1": False, "q2": False, "q3": False}) is False
        assert f.evaluate({"q1": False, "q2": True, "q3": False}) is True

    def test_nested_parentheses(self):
        f = BooleanFormula("!((q1 & q2) | q3)")
        assert f.evaluate({"q1": True, "q2": True, "q3": False}) is False
        assert f.evaluate({"q1": False, "q2": False, "q3": False}) is True

    def test_double_negation(self):
        f = BooleanFormula("!!q1")
        assert f.evaluate({"q1": True}) is True
        assert f.evaluate({"q1": False}) is False

    def test_repr(self):
        f = BooleanFormula("q1 & q2")
        assert "q1 & q2" in repr(f)

    @pytest.mark.parametrize(
        "expr",
        [
            "",               # empty
            "q1 &",           # trailing operator
            "q1 q2",          # missing op between names
            "(q1",            # unclosed paren
            "q1)",            # unbalanced paren
            "&& q1",          # leading binary op
        ],
    )
    def test_malformed(self, expr):
        with pytest.raises((FormulaError, IndexError)):
            BooleanFormula(expr)
