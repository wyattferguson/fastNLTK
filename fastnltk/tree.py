"""
fastnltk.tree — Drop-in replacement for nltk.tree.

Rust-accelerated Tree data structure for common operations:
leaves(), height(), productions(), subtrees(), pprint().
Falls back to NLTK for complex tree operations.
"""

import warnings

import nltk.tree as _nltk_tree
from nltk.tree import *  # noqa: F403

_rust_available = False
try:
    from fastnltk._rust import Tree as _RustTree
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to NLTK tree"
    )

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
        if _rust_available:
            self._impl = _RustTree(label, children or [])
        else:
            self._impl = _nltk_tree.Tree(label, children or [])

    @classmethod
    def from_string(cls, string):
        if _rust_available:
            return cls.__new__(cls)._from_impl(_RustTree.from_string(string))
        return _nltk_tree.Tree.from_string(string)

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
        if _rust_available:
            return [Tree._from_impl(_RustTree.from_string(s)) for s in self._impl.subtrees()]
        return self._impl.subtrees(filter_fn)

    def pprint(self):
        return self._impl.pprint()

    def __str__(self):
        return str(self._impl)

    def __repr__(self):
        return repr(self._impl)

    def __len__(self):
        return self._impl.__len__()

    def __getitem__(self, index):
        return self._impl.__getitem__(index)

    def __iter__(self):
        return iter(self._impl)


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
