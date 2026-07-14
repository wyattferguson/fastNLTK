"""
fastnltk.corpus — Drop-in replacement for nltk.corpus.

Rust-accelerated PlaintextCorpusReader for fast file I/O + tokenization.
Other corpus readers fall back to NLTK.
"""

from nltk.corpus import *  # noqa: F403

from fastnltk._rust import PlaintextCorpusReader as _RustPlaintextCorpusReader


class PlaintextCorpusReader:
    """Rust-accelerated corpus reader for plaintext files."""

    def __init__(self, root, fileids=None, encoding=None):
        self._impl = _RustPlaintextCorpusReader(root, fileids, encoding)

    def fileids(self):
        return self._impl.fileids()

    def raw(self, fileids=None):
        return self._impl.raw(fileids)

    def words(self, fileids=None):
        return self._impl.words(fileids)

    def sents(self, fileids=None):
        return self._impl.sents(fileids)
