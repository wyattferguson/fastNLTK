"""
fastnltk.tree — Drop-in replacement for nltk.tree.

Rust-accelerated Tree data structure for common operations:
leaves(), height(), productions(), subtrees(), pprint().
Falls back to NLTK for complex tree operations.
"""

import nltk.tree as _nltk_tree
from nltk.tree import *  # noqa: F403

from fastnltk._rust import Tree as _RustTree

__all__ = [
    "Tree",
    "TreePrettyPrinter",
    "ParentedTree",
    "ImmutableTree",
    "ImmutableParentedTree",
    "MultiParentedTree",
    "ProbabilisticTree",
    "ImmutableProbabilisticTree",
]


class Tree:
    """Rust-accelerated Tree data structure matching NLTK's Tree API."""

    def __init__(self, label, children=None):
        if children is None:
            children = []
        # Build tree: str children passed to Rust, Tree children appended
        str_children = [c for c in children if isinstance(c, str)]
        self._impl = _RustTree(label, str_children)
        for c in children:
            if isinstance(c, Tree):
                self._impl.append(c._impl)

    @classmethod
    def from_string(cls, string):
        return cls._from_impl(_RustTree.from_string(string))

    @classmethod
    def fromstring(cls, string):
        """Alias for from_string, matching NLTK's Tree.fromstring."""
        return cls.from_string(string)

    @classmethod
    def bracket_parse(cls, string):
        """Alias for from_string, matching NLTK's bracket_parse."""
        return cls.from_string(string)

    @classmethod
    def _from_impl(cls, impl):
        inst = cls.__new__(cls)
        inst._impl = impl
        return inst

    def label(self):
        return self._impl.label()

    def leaves(self):
        return self._impl.leaves()

    def leaf_treepositions(self):
        return self._impl.leaf_treepositions()

    def height(self):
        return self._impl.height()

    def productions(self):
        return self._impl.productions()

    def subtrees(self, filter_fn=None):
        results = []
        for sub in self._impl.subtrees():
            tree = Tree._from_impl(sub)
            if filter_fn is None or filter_fn(tree):
                results.append(tree)
        return results

    def pprint(self):
        return self._impl.pprint()

    def __str__(self):
        return str(self._impl)

    def __repr__(self):
        return repr(self._impl)

    def __len__(self):
        return self._impl.__len__()

    def __getitem__(self, index):
        val = self._impl.__getitem__(index)
        if val is None:
            raise IndexError(index)
        # If it's a string, return directly. Otherwise it's a Py Tree (subtree).
        if isinstance(val, str):
            return val
        return Tree._from_impl(val)

    def __iter__(self):
        yield from self._impl.__iter__()

    def __eq__(self, other):
        if isinstance(other, Tree):
            return str(self) == str(other)
        return NotImplemented

    def append(self, child):
        """Append a child (str or Tree)."""
        self._impl.append(child)

    def __hash__(self):
        return hash(str(self))


ParentedTree = _nltk_tree.ParentedTree
ImmutableTree = _nltk_tree.ImmutableTree
ImmutableParentedTree = _nltk_tree.ImmutableParentedTree
MultiParentedTree = _nltk_tree.MultiParentedTree
ProbabilisticTree = _nltk_tree.ProbabilisticTree
ImmutableProbabilisticTree = _nltk_tree.ImmutableProbabilisticTree
TreePrettyPrinter = _nltk_tree.TreePrettyPrinter

immutable = _nltk_tree.immutable
parented = _nltk_tree.parented
parsing = _nltk_tree.parsing
prettyprinter = _nltk_tree.prettyprinter
probabilistic = _nltk_tree.probabilistic
transforms = _nltk_tree.transforms
tree = _nltk_tree.tree
