"""AND / OR / NOT word operators (any case) in the boolean-formula DSL."""

from __future__ import annotations

from citeclaw.screening.formula import BooleanFormula, tokenize


class TestWordOperators:
    def test_and_or_not_words(self):
        f = BooleanFormula("(q1 or q2) and not q3")
        assert f.query_names() == {"q1", "q2", "q3"}
        assert f.evaluate({"q1": True, "q3": False}) is True
        assert f.evaluate({"q2": True, "q3": False}) is True
        assert f.evaluate({"q1": True, "q3": True}) is False
        assert f.evaluate({"q3": False}) is False  # neither q1 nor q2

    def test_case_insensitive(self):
        assert BooleanFormula("q1 AND q2").evaluate({"q1": True, "q2": True})
        assert BooleanFormula("q1 Or q2").evaluate({"q1": False, "q2": True})
        assert BooleanFormula("NoT q1").evaluate({"q1": False}) is True

    def test_glyphs_still_work_and_can_mix(self):
        assert BooleanFormula("q1 & q2").evaluate({"q1": True, "q2": True})
        f = BooleanFormula("q1 and (q2 | q3)")
        assert f.evaluate({"q1": True, "q3": True})
        assert not f.evaluate({"q1": False, "q3": True})

    def test_identifiers_containing_op_words_stay_names(self):
        assert BooleanFormula("android").query_names() == {"android"}
        assert BooleanFormula("q_or_1").query_names() == {"q_or_1"}
        assert BooleanFormula("nothing & orbit").query_names() == {"nothing", "orbit"}

    def test_tokenize_normalises_words_to_glyphs(self):
        assert tokenize("a and b") == [("NAME", "a"), ("OP", "&"), ("NAME", "b")]
        assert tokenize("a or not b") == [
            ("NAME", "a"),
            ("OP", "|"),
            ("OP", "!"),
            ("NAME", "b"),
        ]
