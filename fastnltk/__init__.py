"""
fastNLTK — Drop-in Rust-accelerated replacement for NLTK.

Same API, same behavior, 5-50x faster on hot paths.

Usage:
    >>> from fastnltk import word_tokenize, pos_tag
    >>> tokens = word_tokenize("Mr. Smith can't believe how fast this is.")
    >>> tags = pos_tag(tokens)
"""

from fastnltk import tokenize, tag, stem, classify, collocations
from fastnltk import probability, lm, metrics, chunk, parse, tree
from fastnltk import corpus, sentiment, translate, sem, inference
from fastnltk import cluster, ccg, chat
from fastnltk import data, downloader

# Re-export top-level convenience functions
from fastnltk.tokenize import sent_tokenize, word_tokenize
from fastnltk.tag import pos_tag, pos_tag_sents
from fastnltk.chunk import ne_chunk, ne_chunk_sents
from fastnltk.data import find, load
from fastnltk.downloader import download

# NLTK-compatible submodule references
from fastnltk import help

__version__ = "0.1.0"
