"""fastnltk.collocations — Drop-in replacement for nltk.collocations."""

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

from nltk.probability import FreqDist
from nltk.util import ngrams

__all__ = [
    "BigramCollocationFinder",
    "TrigramCollocationFinder",
    "QuadgramCollocationFinder",
]


class BigramCollocationFinder:
    """Bigram collocation finder — Rust-accelerated."""
    def __init__(self, word_fd, ngram_fd):
        self._word_fd = word_fd
        self._ngram_fd = ngram_fd
        self._impl = None

    @classmethod
    def from_words(cls, words, window_size=2):
        if _rust_available:
            inst = cls.__new__(cls)
            inst._impl = _RustBigramCollocationFinder.from_words(words, window_size)
            inst._word_fd = None
            inst._ngram_fd = None
            return inst
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        if _rust_available and self._impl:
            name = getattr(score_fn, '__name__', str(score_fn)).lower()
            m = {'raw_freq': 'raw_freq', 'pmi': 'pmi', 'chi_sq': 'chi_sq', 'likelihood_ratio': 'likelihood_ratio'}
            return self._impl.score_ngrams(m.get(name, 'raw_freq'))
        return []

    def nbest(self, score_fn, n):
        if _rust_available and self._impl:
            name = getattr(score_fn, '__name__', str(score_fn)).lower()
            m = {'raw_freq': 'raw_freq', 'pmi': 'pmi', 'chi_sq': 'chi_sq', 'likelihood_ratio': 'likelihood_ratio'}
            return self._impl.nbest(m.get(name, 'raw_freq'), n)
        return []

    def apply_freq_filter(self, min_freq):
        if _rust_available and self._impl:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        pass


class TrigramCollocationFinder:
    """Trigram collocation finder — Rust-accelerated."""
    def __init__(self, word_fd, ngram_fd):
        self._word_fd = word_fd
        self._ngram_fd = ngram_fd
        self._impl = None

    @classmethod
    def from_words(cls, words, window_size=3):
        if _rust_available:
            inst = cls.__new__(cls)
            inst._impl = _RustTrigramCollocationFinder.from_words(words, window_size)
            return inst
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        if _rust_available and self._impl:
            return self._impl.score_ngrams('raw_freq')
        return []

    def nbest(self, score_fn, n):
        if _rust_available and self._impl:
            return self._impl.nbest('raw_freq', n)
        return []

    def apply_freq_filter(self, min_freq):
        if _rust_available and self._impl:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        pass


class QuadgramCollocationFinder:
    """Quadgram collocation finder — Rust-accelerated."""
    def __init__(self, word_fd, ngram_fd):
        self._word_fd = word_fd
        self._ngram_fd = ngram_fd
        self._impl = None

    @classmethod
    def from_words(cls, words, window_size=4):
        if _rust_available:
            inst = cls.__new__(cls)
            inst._impl = _RustQuadgramCollocationFinder.from_words(words, window_size)
            return inst
        word_fd = FreqDist(words)
        ngram_fd = FreqDist(ngrams(words, window_size))
        return cls(word_fd, ngram_fd)

    def score_ngrams(self, score_fn):
        if _rust_available and self._impl:
            return self._impl.score_ngrams('raw_freq')
        return []

    def nbest(self, score_fn, n):
        if _rust_available and self._impl:
            return self._impl.nbest('raw_freq', n)
        return []

    def apply_freq_filter(self, min_freq):
        if _rust_available and self._impl:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        pass
