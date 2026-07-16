"""
fastNLTK — Drop-in Rust-accelerated replacement for NLTK.

Same API, same behavior, 5-50x faster on hot paths.

Usage:
    >>> from fastnltk import word_tokenize, pos_tag
    >>> tokens = word_tokenize("Mr. Smith can't believe how fast this is.")
    >>> tags = pos_tag(tokens)
"""

import importlib
import warnings

from fastnltk.chunk import ne_chunk, ne_chunk_sents
from fastnltk.data import find, load
from fastnltk.downloader import download
from fastnltk.tag import pos_tag, pos_tag_sents
from fastnltk.tokenize import sent_tokenize, word_tokenize

__all__ = [
    "sent_tokenize",
    "word_tokenize",
    "pos_tag",
    "pos_tag_sents",
    "ne_chunk",
    "ne_chunk_sents",
    "find",
    "load",
    "download",
]

__version__ = "0.4.1"

# Import remaining submodules — each handles Rust-availability gracefully
_modules = [
    "stem",
    "probability",
    "lm",
    "metrics",
    "parse",
    "tree",
    "corpus",
    "sentiment",
    "translate",
    "sem",
    "inference",
    "cluster",
    "ccg",
    "chat",
    "classify",
    "collocations",
]

for _mod_name in _modules:
    try:
        mod = importlib.import_module(f"fastnltk.{_mod_name}")
        globals()[_mod_name] = mod
    except ImportError as e:
        warnings.warn(f"fastnltk.{_mod_name} not available: {e}")
