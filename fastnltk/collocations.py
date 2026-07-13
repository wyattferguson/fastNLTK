"""
fastnltk.collocations — Drop-in replacement for nltk.collocations.
"""

_rust_available = False
try:
    from fastnltk._rust import (
        BigramCollocationFinder as _RustBigramCollocationFinder,
        TrigramCollocationFinder as _RustTrigramCollocationFinder,
        QuadgramCollocationFinder as _RustQuadgramCollocationFinder,
    )
    _rust_available = True
except ImportError:
    pass

from nltk.metrics import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    ContingencyMeasures,
    QuadgramAssocMeasures,
)
from nltk.probability import FreqDist
from nltk.util import ngrams

__all__ = [
    "BigramCollocationFinder",
    "TrigramCollocationFinder",
    "QuadgramCollocationFinder",
]


class BigramCollocationFinder:
    """Bigram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustBigramCollocationFinder(word_fd, ngram_fd) if _rust_available else None
        self._word_fd = word_fd
        self._ngram_fd = ngram_fd

    @classmethod
    def from_words(cls, words, window_size=2):
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        return self._impl.score_ngrams(score_fn) if _rust_available else []

    def nbest(self, score_fn, n):
        return self._impl.nbest(score_fn, n) if _rust_available else []

    def apply_freq_filter(self, min_freq):
        if _rust_available:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        if _rust_available:
            self._impl.apply_word_filter(filter_fn)


class TrigramCollocationFinder:
    """Trigram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustTrigramCollocationFinder(word_fd, ngram_fd) if _rust_available else None

    @classmethod
    def from_words(cls, words, window_size=3):
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)


class QuadgramCollocationFinder:
    """Quadgram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustQuadgramCollocationFinder(word_fd, ngram_fd) if _rust_available else None

    @classmethod
    def from_words(cls, words, window_size=4):
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)
