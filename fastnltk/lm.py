"""
fastnltk.lm — Drop-in replacement for nltk.lm.

All LM models are Rust-accelerated via the compiled `_rust` extension.
"""

import nltk.lm as _nltk_lm

from nltk.lm.preprocessing import (
    everygrams,
    pad_both_ends,
    pad_sequence,
    padded_everygrams,
)
from nltk.lm.util import log_base2

from fastnltk._rust import (
    MLE as _RustMLE,
    KneserNeyInterpolated as _RustKneserNeyInterpolated,
    Laplace as _RustLaplace,
    Lidstone as _RustLidstone,
    StupidBackoff as _RustStupidBackoff,
    WittenBellInterpolated as _RustWittenBellInterpolated,
)

__all__ = [
    "MLE", "Lidstone", "Laplace",
    "KneserNeyInterpolated",
    "StupidBackoff", "WittenBellInterpolated",
    "AbsoluteDiscountingInterpolated",
    "padded_everygrams",
    "everygrams",
    "pad_both_ends",
    "pad_sequence",
    "log_base2",
    "NgramCounter",
    "Vocabulary",
]


class MLE:
    def __init__(self, order):
        self._impl = _RustMLE(order)

    def fit(self, sentences, vocabulary=None):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    def logscore(self, word, context=None):
        return self._impl.logscore(word, context)

    def generate(self, num_words, text_seed=None, random_seed=None):
        return self._impl.generate(num_words, text_seed, random_seed)

    @property
    def order(self):
        return self._impl.order()

    @property
    def vocab_size(self):
        return self._impl.vocab_size()

    @property
    def fitted(self):
        return self._impl.fitted()


class Lidstone:
    def __init__(self, order, gamma):
        self._impl = _RustLidstone(order, gamma)

    def fit(self, sentences, vocabulary=None):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    def logscore(self, word, context=None):
        return self._impl.logscore(word, context)

    def generate(self, num_words, text_seed=None, random_seed=None):
        return self._impl.generate(num_words, text_seed, random_seed)

    @property
    def order(self):
        return self._impl.order()

    @property
    def vocab_size(self):
        return self._impl.vocab_size()

    @property
    def fitted(self):
        return self._impl.fitted()


class Laplace:
    def __init__(self, order):
        self._impl = _RustLaplace(order)

    def fit(self, sentences, vocabulary=None):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    def logscore(self, word, context=None):
        return self._impl.logscore(word, context)

    def generate(self, num_words, text_seed=None, random_seed=None):
        return self._impl.generate(num_words, text_seed, random_seed)

    @property
    def order(self):
        return self._impl.order()

    @property
    def vocab_size(self):
        return self._impl.vocab_size()

    @property
    def fitted(self):
        return self._impl.fitted()


class KneserNeyInterpolated:
    def __init__(self, order, discount=0.75):
        self._impl = _RustKneserNeyInterpolated(order, discount)

    def fit(self, sentences):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    @property
    def order(self):
        return self._impl.order()

    @property
    def fitted(self):
        return self._impl.fitted()


class WittenBellInterpolated:
    def __init__(self, order):
        self._impl = _RustWittenBellInterpolated(order)

    def fit(self, sentences):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    @property
    def order(self):
        return self._impl.order()

    @property
    def fitted(self):
        return self._impl.fitted()


class StupidBackoff:
    """Stupid backoff LM — Rust-accelerated."""
    def __init__(self, order, alpha=0.4):
        self._impl = _RustStupidBackoff(order, alpha)

    def fit(self, sentences):
        self._impl.fit(sentences)

    def score(self, word, context=None):
        return self._impl.score(word, context)

    @property
    def order(self):
        return self._impl.order()

    @property
    def fitted(self):
        return self._impl.fitted()


# ── NLTK re-exports for API compatibility ─────

AbsoluteDiscountingInterpolated = _nltk_lm.AbsoluteDiscountingInterpolated
NgramCounter = _nltk_lm.NgramCounter
Vocabulary = _nltk_lm.Vocabulary

api = _nltk_lm.api
counter = _nltk_lm.counter
models = _nltk_lm.models
preprocessing = _nltk_lm.preprocessing
smoothing = _nltk_lm.smoothing
util = _nltk_lm.util
vocabulary = _nltk_lm.vocabulary
