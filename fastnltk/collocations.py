"""
fastnltk.collocations — Drop-in replacement for nltk.collocations.
"""

from fastnltk._rust import (
    BigramCollocationFinder as _RustBigramCollocationFinder,
    TrigramCollocationFinder as _RustTrigramCollocationFinder,
    QuadgramCollocationFinder as _RustQuadgramCollocationFinder,
)

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
    """Rust-accelerated bigram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustBigramCollocationFinder(word_fd, ngram_fd)

    @classmethod
    def from_words(cls, words, window_size=2):
        """Construct from a sequence of words."""
        return cls._from_words(list(words), window_size)

    @classmethod
    def _from_words(cls, words, window_size):
        """Internal: construct from word list."""
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        """Return scored ngrams."""
        return self._impl.score_ngrams(score_fn)

    def nbest(self, score_fn, n):
        """Return the n best scored ngrams."""
        return self._impl.nbest(score_fn, n)

    def apply_freq_filter(self, min_freq):
        """Filter out ngrams with frequency below min_freq."""
        self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        """Filter out ngrams containing words that pass filter_fn."""
        self._impl.apply_word_filter(filter_fn)


class TrigramCollocationFinder:
    """Rust-accelerated trigram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustTrigramCollocationFinder(word_fd, ngram_fd)

    @classmethod
    def from_words(cls, words, window_size=3):
        return cls._from_words(list(words), window_size)

    @classmethod
    def _from_words(cls, words, window_size):
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        return self._impl.score_ngrams(score_fn)

    def nbest(self, score_fn, n):
        return self._impl.nbest(score_fn, n)

    def apply_freq_filter(self, min_freq):
        self._impl.apply_freq_filter(min_freq)


class QuadgramCollocationFinder:
    """Rust-accelerated quadgram collocation finder."""
    def __init__(self, word_fd, ngram_fd):
        self._impl = _RustQuadgramCollocationFinder(word_fd, ngram_fd)

    @classmethod
    def from_words(cls, words, window_size=4):
        return cls._from_words(list(words), window_size)

    @classmethod
    def _from_words(cls, words, window_size):
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        return self._impl.score_ngrams(score_fn)

    def nbest(self, score_fn, n):
        return self._impl.nbest(score_fn, n)

    def apply_freq_filter(self, min_freq):
        self._impl.apply_freq_filter(min_freq)
