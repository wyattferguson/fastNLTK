"""
fastnltk.chunk — Drop-in replacement for nltk.chunk.
"""

import nltk.chunk as _nltk_chunk
from nltk.chunk import (
    ChunkParserI,
    RegexpChunkParser,
    RegexpParser,
    Maxent_NE_Chunker,
    ChunkScore,
    accuracy,
    conllstr2tree,
    conlltags2tree,
    ieerstr2tree,
    tagstr2tree,
    tree2conllstr,
    tree2conlltags,
)

__all__ = [
    "ChunkParserI",
    "RegexpParser",
    "RegexpChunkParser",
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


def ne_chunk(tagged_tokens, binary=False):
    """Named entity chunking."""
    return _nltk_chunk.ne_chunk(tagged_tokens, binary)


def ne_chunk_sents(tagged_sentences, binary=False):
    """Named entity chunking for multiple sentences."""
    return _nltk_chunk.ne_chunk_sents(tagged_sentences, binary)
