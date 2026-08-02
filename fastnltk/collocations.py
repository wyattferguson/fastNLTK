"""fastnltk.collocations — Drop-in replacement for nltk.collocations."""

import nltk.collocations as _nltk_collocations

from fastnltk._rust import (
    BigramCollocationFinder as _RustBigramCollocationFinder,
)
from fastnltk._rust import (
    QuadgramCollocationFinder as _RustQuadgramCollocationFinder,
)
from fastnltk._rust import (
    TrigramCollocationFinder as _RustTrigramCollocationFinder,
)

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
    def from_words(cls, words: list[str], window_size: int = 2) -> any:
        inst = cls.__new__(cls)
        inst._impl = _RustBigramCollocationFinder.from_words(words, window_size)
        # Build Python-side word/ngram freq dicts for NLTK compat
        from collections import Counter

        inst._word_fd = Counter(words)
        inst._ngram_fd = Counter()
        for i in range(len(words) - window_size + 1):
            inst._ngram_fd[tuple(words[i : i + window_size])] += 1
        return inst

    @property
    def word_fd(self) -> any:
        return self._word_fd

    @property
    def ngram_fd(self) -> any:
        return self._ngram_fd

    def score_ngrams(self, score_fn: any) -> list[tuple[any, float]]:
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.score_ngrams(m.get(name, "raw_freq"))
        return []

    def nbest(self, score_fn: any, n: int) -> list[any]:
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.nbest(m.get(name, "raw_freq"), n)
        return []

    def apply_freq_filter(self, min_freq: int) -> any:
        if self._impl:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn: any) -> any:
        pass


class TrigramCollocationFinder:
    """Trigram collocation finder — Rust-accelerated."""

    def __init__(self, word_fd, ngram_fd):
        self._word_fd = word_fd
        self._ngram_fd = ngram_fd
        self._impl = None

    @classmethod
    def from_words(cls, words: list[str], window_size: int = 3) -> any:
        inst = cls.__new__(cls)
        inst._impl = _RustTrigramCollocationFinder.from_words(words, window_size)
        from collections import Counter

        inst._word_fd = Counter(words)
        inst._ngram_fd = Counter()
        for i in range(len(words) - window_size + 1):
            inst._ngram_fd[tuple(words[i : i + window_size])] += 1
        return inst

    @property
    def word_fd(self):
        return self._word_fd

    @property
    def ngram_fd(self):
        return self._ngram_fd

    def score_ngrams(self, score_fn):
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.score_ngrams(m.get(name, "raw_freq"))
        return []

    def nbest(self, score_fn, n):
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.nbest(m.get(name, "raw_freq"), n)
        return []

    def apply_freq_filter(self, min_freq):
        if self._impl:
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
    def from_words(cls, words: list[str], window_size: int = 4) -> any:
        inst = cls.__new__(cls)
        inst._impl = _RustQuadgramCollocationFinder.from_words(words, window_size)
        from collections import Counter

        inst._word_fd = Counter(words)
        inst._ngram_fd = Counter()
        for i in range(len(words) - window_size + 1):
            inst._ngram_fd[tuple(words[i : i + window_size])] += 1
        return inst

    @property
    def word_fd(self):
        return self._word_fd

    @property
    def ngram_fd(self):
        return self._ngram_fd

    def score_ngrams(self, score_fn):
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.score_ngrams(m.get(name, "raw_freq"))
        return []

    def nbest(self, score_fn, n):
        if self._impl:
            name = getattr(score_fn, "__name__", str(score_fn)).lower()
            m = {
                "raw_freq": "raw_freq",
                "pmi": "pmi",
                "chi_sq": "chi_sq",
                "likelihood_ratio": "likelihood_ratio",
            }
            return self._impl.nbest(m.get(name, "raw_freq"), n)
        return []

    def apply_freq_filter(self, min_freq):
        if self._impl:
            self._impl.apply_freq_filter(min_freq)

    def apply_word_filter(self, filter_fn):
        pass


# ── NLTK re-exports ─────
BigramAssocMeasures = _nltk_collocations.BigramAssocMeasures
TrigramAssocMeasures = _nltk_collocations.TrigramAssocMeasures
QuadgramAssocMeasures = _nltk_collocations.QuadgramAssocMeasures
ContingencyMeasures = _nltk_collocations.ContingencyMeasures
AbstractCollocationFinder = _nltk_collocations.AbstractCollocationFinder
demo = _nltk_collocations.demo
