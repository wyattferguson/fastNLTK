"""
fastnltk.corpus — Drop-in replacement for nltk.corpus.

Rust-accelerated PlaintextCorpusReader for fast file I/O + tokenization.
Other corpus readers fall back to NLTK.
"""

import warnings

from nltk.corpus import *  # noqa: F403

_rust_available = False
try:
    from fastnltk._rust import PlaintextCorpusReader as _RustPlaintextCorpusReader
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to NLTK corpus"
    )


class PlaintextCorpusReader:
    """Rust-accelerated corpus reader for plaintext files."""
    def __init__(self, root, fileids=None, encoding=None):
        if _rust_available:
            self._impl = _RustPlaintextCorpusReader(root, fileids, encoding)
        else:
            from nltk.corpus.reader.plaintext import PlaintextCorpusReader as _NltkReader
            self._impl = _NltkReader(root, fileids, encoding)

    def fileids(self):
        return self._impl.fileids()

    def raw(self, fileids=None):
        return self._impl.raw(fileids)

    def words(self, fileids=None):
        return self._impl.words(fileids)

    def sents(self, fileids=None):
        return self._impl.sents(fileids)
