"""Tests for Rust-accelerated tree module."""

import pytest

from fastnltk.tree import Tree


class TestTree:
    def test_create_with_children(self):
        t = Tree("S", ["The", "cat", "runs"])
        assert t.label() == "S"
        assert t.leaves() == ["The", "cat", "runs"]

    def test_create_without_children(self):
        t = Tree("NP")
        assert t.label() == "NP"
        assert t.leaves() == []

    def test_from_string_simple(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        assert t.label() == "S"
        assert t.leaves() == ["The", "runs"]

    def test_from_string_nested(self):
        t = Tree.from_string("(S (NP (Det The) (N cat)) (VP (V runs)))")
        assert t.leaves() == ["The", "cat", "runs"]
        assert t.height() >= 4

    def test_from_string_invalid(self):
        with pytest.raises(Exception):
            Tree.from_string("not a tree")

        with pytest.raises(Exception):
            Tree.from_string("")

    def test_height(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        assert t.height() == 3

    def test_height_leaf(self):
        t = Tree("X", ["word"])
        assert t.height() == 2

    def test_len_counts_leaves(self):
        t = Tree.from_string("(S (NP (Det The) (N cat)) (VP runs))")
        assert len(t) == 3  # The, cat, runs

    def test_leaves(self):
        t = Tree.from_string("(S (NP (Det The)) (VP runs))")
        assert t.leaves() == ["The", "runs"]

    def test_leaf_treepositions(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        positions = t.leaf_treepositions()
        assert len(positions) == 2

    def test_productions(self):
        t = Tree.from_string("(S (NP (Det The) (N cat)) (VP (V runs)))")
        prods = t.productions()
        assert any("S ->" in p for p in prods)
        assert any("NP ->" in p for p in prods)
        assert any("VP ->" in p for p in prods)

    def test_subtrees(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        subs = t.subtrees()
        assert len(subs) >= 3  # S, NP, VP

    def test_subtrees_with_filter(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        np_subs = t.subtrees(lambda tr: tr.label() == "NP")
        assert len(np_subs) == 1
        assert np_subs[0].label() == "NP"

    def test_getitem_leaf(self):
        t = Tree.from_string("(S The runs)")
        assert t[0] == "The"
        assert t[1] == "runs"

    def test_getitem_subtree(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        child = t[0]
        assert isinstance(child, Tree)
        assert child.label() == "NP"

    def test_getitem_negative_index(self):
        t = Tree.from_string("(S The runs)")
        assert t[-1] == "runs"

    def test_getitem_out_of_range(self):
        t = Tree.from_string("(S The runs)")
        with pytest.raises(IndexError):
            t[99]

    def test_iterator(self):
        t = Tree.from_string("(S The runs)")
        children = list(t)
        assert len(children) == 2

    def test_pprint(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        s = t.pprint()
        assert "S" in s
        assert "The" in s

    def test_str_and_repr(self):
        t = Tree.from_string("(S (NP The) (VP runs))")
        assert str(t).startswith("(")
        assert repr(t).startswith("Tree")

    def test_equality(self):
        t1 = Tree.from_string("(S The runs)")
        t2 = Tree.from_string("(S The runs)")
        t3 = Tree.from_string("(S The walks)")
        assert t1 == t2
        assert t1 != t3

    def test_hash(self):
        t = Tree.from_string("(S The runs)")
        assert isinstance(hash(t), int)

    def test_empty_children(self):
        t = Tree("ROOT")
        assert t.leaves() == []
        assert t.height() == 1

    def test_deep_nesting(self):
        t = Tree.from_string("(A (B (C (D (E leaf)))))")
        assert t.leaves() == ["leaf"]
        assert t.height() == 6
