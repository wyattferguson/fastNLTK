"""
fastnltk.lm — Drop-in replacement for nltk.lm.

Rust-accelerated LM models:
  - MLE, Lidstone, Laplace — compiled Rust via rustling, 11-39x faster
  - KneserNeyInterpolated — compiled Rust
  - WittenBellInterpolated, StupidBackoff — fall back to NLTK (no Rust crate)
"""

import warnings

from nltk.lm import StupidBackoff, WittenBellInterpolated
from nltk.lm.preprocessing import (
    everygrams,
    pad_both_ends,
    pad_sequence,
    padded_everygrams,
)
from nltk.lm.util import log_base2

_rust_available = False
try:
    from fastnltk._rust import (
        MLE as _RustMLE,
    )
    from fastnltk._rust import (
        KneserNeyInterpolated as _RustKneserNeyInterpolated,
    )
    from fastnltk._rust import (
        Laplace as _RustLaplace,
    )
    from fastnltk._rust import (
        Lidstone as _RustLidstone,
    )
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK LM"
    )

__all__ = [
    "MLE", "Lidstone", "Laplace",
    "KneserNeyInterpolated",
    "StupidBackoff", "WittenBellInterpolated",
    "padded_everygrams",
    "everygrams",
    "pad_both_ends",
    "pad_sequence",
    "log_base2",
]


class MLE:
    def __init__(self, order):
        if _rust_available:
            self._impl = _RustMLE(order)
        else:
            from nltk.lm import MLE as _NltkMLE
            self._impl = _NltkMLE(order)

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
        if _rust_available:
            self._impl = _RustLidstone(order, gamma)
        else:
            from nltk.lm import Lidstone as _NltkLidstone
            self._impl = _NltkLidstone(order, gamma)

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
        if _rust_available:
            self._impl = _RustLaplace(order)
        else:
            from nltk.lm import Laplace as _NltkLaplace
            self._impl = _NltkLaplace(order)

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
        if _rust_available:
            self._impl = _RustKneserNeyInterpolated(order, discount)
        else:
            from nltk.lm import KneserNeyInterpolated as _NltkKNI
            self._impl = _NltkKNI(order)

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
