"""
fastnltk.chunk — Drop-in replacement for nltk.chunk.
"""

import nltk.chunk as _nltk_chunk
from nltk.chunk import (
    ChunkParserI,
    ChunkScore,
    Maxent_NE_Chunker,
    accuracy,
    conllstr2tree,
    conlltags2tree,
    tree2conllstr,
    tree2conlltags,
)

from fastnltk._rust import RegexpParser as _RustRegexpParser

__all__ = [
    "ChunkParserI",
    "RegexpParser",
    "Maxent_NE_Chunker",
    "ne_chunk",
    "ne_chunk_sents",
    "ChunkScore",
    "accuracy",
    "conllstr2tree",
    "conlltags2tree",
    "tree2conllstr",
    "tree2conlltags",
]


class RegexpParser:
    """Rust-accelerated RegexpParser for chunk grammar matching."""

    def __init__(self, grammar):
        self._impl = _RustRegexpParser(grammar)

    def parse(self, tokens: list[str]) -> any:
        result = self._impl.parse(tokens)
        return _iob_to_tree(result)


def _iob_to_tree(iob_tags):
    """Convert list of (word, pos_tag, iob_tag) triples to a Tree
    with leaves as "word/pos" matching NLTK's format."""
    from fastnltk.tree import Tree

    if not iob_tags:
        return Tree("S", [])

    # Handle both (w, iob) and (w, pos, iob) formats
    if len(iob_tags[0]) == 3:
        words, pos_tags, iob = zip(*iob_tags)
    else:
        words = [w for w, _ in iob_tags]
        pos_tags = ["" for _ in iob_tags]
        iob = [t for _, t in iob_tags]

    # Build tree as nested list of (label, [children]) or leaf strings
    tree_children: list = []
    current_chunk_label: str | None = None
    current_chunk_children: list | None = None

    for word, pos, tag in zip(words, pos_tags, iob):
        leaf = f"{word}/{pos}" if pos else word
        if tag.startswith("B-"):
            if current_chunk_label is not None:
                tree_children.append((current_chunk_label, current_chunk_children))
            current_chunk_label = tag[2:]
            current_chunk_children = [leaf]
        elif tag.startswith("I-"):
            if current_chunk_children is not None:
                current_chunk_children.append(leaf)
            else:
                current_chunk_label = tag[2:]
                current_chunk_children = [leaf]
        else:  # O
            if current_chunk_label is not None:
                tree_children.append((current_chunk_label, current_chunk_children))
                current_chunk_label = None
                current_chunk_children = None
            tree_children.append(leaf)

    if current_chunk_label is not None:
        tree_children.append((current_chunk_label, current_chunk_children))

    # Convert nested structure to tree string for Rust Tree.from_string
    def _to_str(item) -> str:
        if isinstance(item, str):
            return item
        label, children = item
        inner = " ".join(_to_str(c) for c in children)
        return f"({label} {inner})"

    tree_str = f"(S {' '.join(_to_str(c) for c in tree_children)})"
    try:
        return Tree.from_string(tree_str)
    except Exception:
        # Fallback: build a flat tree

        flat = [f"{w}/{p}" if p else w for w, p in zip(words, pos_tags)]
        return Tree("S", flat)


def ne_chunk(tagged_tokens: list[tuple[str, str]], binary: bool = False) -> any:
    return _nltk_chunk.ne_chunk(tagged_tokens, binary)


def ne_chunk_sents(tagged_sentences: list[list[tuple[str, str]]], binary: bool = False) -> any:
    return _nltk_chunk.ne_chunk_sents(tagged_sentences, binary)


# ── NLTK re-exports ─────
ne_chunker = _nltk_chunk.ne_chunker
named_entity = _nltk_chunk.named_entity
regexp = _nltk_chunk.regexp
api = _nltk_chunk.api
RegexpChunkParser = _nltk_chunk.RegexpChunkParser
ieerstr2tree = _nltk_chunk.ieerstr2tree
tagstr2tree = _nltk_chunk.tagstr2tree
util = _nltk_chunk.util
