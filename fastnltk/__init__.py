"""
fastNLTK — Drop-in Rust-accelerated replacement for NLTK.

Same API, same behavior, 5-50x faster on hot paths.

Usage:
    >>> from fastnltk import word_tokenize, pos_tag
    >>> tokens = word_tokenize("Mr. Smith can't believe how fast this is.")
    >>> tags = pos_tag(tokens)
"""

import importlib

__version__ = "0.1.0"

# Import submodules — each handles Rust-availability gracefully
_modules = [
    "tokenize", "tag", "stem", "probability", "lm", "metrics",
    "chunk", "parse", "tree", "corpus", "sentiment", "translate",
    "sem", "inference", "cluster", "ccg", "chat", "data", "downloader",
    "classify", "collocations",
]

for _mod_name in _modules:
    try:
        importlib.import_module(f"fastnltk.{_mod_name}")
    except ImportError:
        pass

# Re-export top-level convenience functions
from fastnltk.tokenize import sent_tokenize, word_tokenize
from fastnltk.tag import pos_tag, pos_tag_sents
from fastnltk.chunk import ne_chunk, ne_chunk_sents
from fastnltk.data import find, load
from fastnltk.downloader import download
