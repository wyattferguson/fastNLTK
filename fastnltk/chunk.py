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

    def parse(self, tokens):
        result = self._impl.parse(tokens)
        return _iob_to_tree(result)


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
    return _nltk_chunk.ne_chunk(tagged_tokens, binary)


def ne_chunk_sents(tagged_sentences, binary=False):
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
