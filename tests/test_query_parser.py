"""Tests for :mod:`citeclaw.agents.query_parser`."""

from __future__ import annotations

import pytest

from citeclaw.agents.query_parser import (
    enumerate_or_leaves,
    parse_query,
    serialize_query,
    tree_signature,
)


def test_parse_single_phrase():
    tree = parse_query('"foundation model"')
    assert tree.kind == "PHRASE"
    assert tree.value == "foundation model"


def test_parse_flat_or():
    tree = parse_query('"A" | "B" | "C"')
    assert tree.kind == "OR"
    assert [c.kind for c in tree.children] == ["PHRASE"] * 3
    assert [c.value for c in tree.children] == ["A", "B", "C"]


def test_parse_flat_and():
    tree = parse_query('"A" +"B"')
    assert tree.kind == "AND"
    assert [c.value for c in tree.children] == ["A", "B"]


def test_parse_nested_or_and():
    tree = parse_query('("foundation model" | "LLM") + ("DNA" | "RNA")')
    assert tree.kind == "AND"
    assert len(tree.children) == 2
    assert all(c.kind == "OR" for c in tree.children)
    assert [c.value for c in tree.children[0].children] == ["foundation model", "LLM"]
    assert [c.value for c in tree.children[1].children] == ["DNA", "RNA"]


def test_parse_not():
    tree = parse_query('"A" -"B"')
    assert tree.kind == "AND"
    assert tree.children[1].kind == "NOT"
    assert tree.children[1].children[0].value == "B"


def test_parse_implicit_and():
    # Adjacent phrases without explicit + are implicit AND.
    tree = parse_query('"A" "B"')
    assert tree.kind == "AND"
    assert [c.value for c in tree.children] == ["A", "B"]


def test_parse_unterminated_quote_raises():
    with pytest.raises(ValueError, match="unterminated"):
        parse_query('"foo')


def test_parse_unbalanced_paren_raises():
    with pytest.raises(ValueError, match="paren"):
        parse_query('("A" | "B"')


def test_serialize_round_trip_flat_or():
    q = '"A" | "B" | "C"'
    assert serialize_query(parse_query(q)) == q


def test_serialize_round_trip_nested():
    q = '("A" | "B") +("C" | "D")'
    # Our serializer emits `+` prefix (no space) for AND operands.
    assert serialize_query(parse_query(q)) == '("A" | "B") +("C" | "D")'


def test_enumerate_or_leaves_flat():
    tree, subs = enumerate_or_leaves('"A" | "B" | "C"')
    assert len(subs) == 3
    assert {s.leaf_text for s in subs} == {'"A"', '"B"', '"C"'}
    assert {s.substituted_query for s in subs} == {'"A"', '"B"', '"C"'}


def test_enumerate_or_leaves_nested():
    tree, subs = enumerate_or_leaves('("A" | "B") +("C" | "D")')
    assert len(subs) == 4
    # Group 0 is the outer AND's left OR; group 1 is the right OR.
    by_group = {s.group_id: [] for s in subs}
    for s in subs:
        by_group[s.group_id].append(s.substituted_query)
    assert set(by_group[0]) == {'"A" +("C" | "D")', '"B" +("C" | "D")'}
    assert set(by_group[1]) == {'("A" | "B") +"C"', '("A" | "B") +"D"'}


def test_enumerate_no_or_returns_empty():
    tree, subs = enumerate_or_leaves('"A" +"B"')
    assert subs == []


def test_tree_signature_nested():
    tree = parse_query('("A" | "B") +("C" | "D")')
    sig = tree_signature(tree)
    assert sig["or_groups"] == 2
    assert sig["or_leaves"] == 4
    assert sig["and_nodes"] == 1
    assert sig["phrases"] == 4


def test_tree_signature_flat_phrase():
    tree = parse_query('"foo"')
    sig = tree_signature(tree)
    assert sig["or_groups"] == 0
    assert sig["phrases"] == 1
    assert sig["and_nodes"] == 0


def test_parser_handles_triple_nested_or():
    tree = parse_query('("A" | "B" | "C") +("D" | "E")')
    sig = tree_signature(tree)
    # 2 OR groups, 5 total leaves (3+2).
    assert sig["or_groups"] == 2
    assert sig["or_leaves"] == 5
