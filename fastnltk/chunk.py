"""
fastnltk.chunk — Drop-in replacement for nltk.chunk.
"""

import warnings

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

_rust_available = False
try:
    from fastnltk._rust import RegexpParser as _RustRegexpParser
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK chunk"
    )

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
        if _rust_available:
            self._impl = _RustRegexpParser(grammar)
        else:
            self._impl = _nltk_chunk.RegexpParser(grammar)

    def parse(self, tokens):
        # tokens is list of (word, pos_tag) tuples
        if _rust_available:
            result = self._impl.parse(tokens)
            # Convert IOB tuples to Tree structure matching NLTK's output
            return _iob_to_tree(result)
        return self._impl.parse(tokens)


def _iob_to_tree(iob_tags):
    """Convert list of (word, iob) tuples to an nltk Tree."""
    from nltk import Tree

    words = [w for w, _ in iob_tags]
    tags = [t for _, t in iob_tags]

    tree = Tree("S", [])
    current_chunk = None

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_chunk is not None:
                tree.append(current_chunk)
            label = tag[2:]
            current_chunk = Tree(label, [word])
        elif tag.startswith("I-"):
            if current_chunk is not None:
                current_chunk.append(word)
            else:
                # I- without B-: treat as B-
                label = tag[2:]
                current_chunk = Tree(label, [word])
        else:  # O
            if current_chunk is not None:
                tree.append(current_chunk)
                current_chunk = None
            tree.append(word)

    if current_chunk is not None:
        tree.append(current_chunk)

    return tree


def ne_chunk(tagged_tokens, binary=False):
    """Named entity chunking."""
    return _nltk_chunk.ne_chunk(tagged_tokens, binary)


def ne_chunk_sents(tagged_sentences, binary=False):
    """Named entity chunking for multiple sentences."""
    return _nltk_chunk.ne_chunk_sents(tagged_sentences, binary)
