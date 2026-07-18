"""
fastnltk.corpus — Drop-in replacement for nltk.corpus.

Rust-accelerated corpus readers for fast file I/O + tokenization:
  - PlaintextCorpusReader
  - TaggedCorpusReader
  - CategorizedPlaintextCorpusReader

Other corpus readers fall back to NLTK.
"""

from nltk.corpus import *  # noqa: F403

from fastnltk._rust import (
    CategorizedPlaintextCorpusReader as _RustCategorizedPlaintextCorpusReader,
)
from fastnltk._rust import (
    PlaintextCorpusReader as _RustPlaintextCorpusReader,
)
from fastnltk._rust import (
    TaggedCorpusReader as _RustTaggedCorpusReader,
)


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

    def paras(self, fileids=None):
        # Fall back to NLTK for paragraph segmentation.
        from nltk.corpus import PlaintextCorpusReader as NltkPCR

        return NltkPCR(self._impl.root if hasattr(self._impl, 'root') else '.', self.fileids()).paras(fileids)


class TaggedCorpusReader:
    """Rust-accelerated reader for word/tag formatted files."""

    def __init__(self, root, fileids, sep='/'):
        self._impl = _RustTaggedCorpusReader(root, fileids, sep)

    def fileids(self):
        return self._impl.fileids()

    def raw(self, fileids=None):
        return self._impl.raw(fileids)

    def tagged_words(self, fileids=None):
        return self._impl.tagged_words(fileids)

    def tagged_sents(self, fileids=None):
        return self._impl.tagged_sents(fileids)

    def words(self, fileids=None):
        return self._impl.words(fileids)


class CategorizedPlaintextCorpusReader:
    """Rust-accelerated reader for categorized plaintext corpora."""

    def __init__(self, root, fileids_map):
        self._impl = _RustCategorizedPlaintextCorpusReader(root, fileids_map)

    def fileids(self):
        return self._impl.fileids()

    def categories(self):
        return self._impl.categories()

    def fileids_by_category(self, category):
        return self._impl.fileids_by_category(category)

    def raw(self, fileids=None):
        return self._impl.raw(fileids)

    def words(self, fileids=None):
        return self._impl.words(fileids)

    def sents(self, fileids=None):
        return self._impl.sents(fileids)
