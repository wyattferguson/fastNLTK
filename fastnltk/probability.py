"""
fastnltk.probability — Drop-in replacement for nltk.probability.
"""

import nltk.probability as _nltk_probability
from nltk.probability import (
    ConditionalProbDist,
    KneserNeyProbDist,
    LaplaceProbDist,
    LidstoneProbDist,
    MLEProbDist,
    ProbDistI,
    SimpleGoodTuringProbDist,
    WittenBellProbDist,
)

from fastnltk._rust import (
    ConditionalFreqDist as _RustConditionalFreqDist,
)
from fastnltk._rust import (
    FreqDist as _RustFreqDist,
)

__all__ = [
    "FreqDist",
    "ConditionalFreqDist",
    "ProbDistI",
    "MLEProbDist",
    "LaplaceProbDist",
    "LidstoneProbDist",
    "WittenBellProbDist",
    "SimpleGoodTuringProbDist",
    "KneserNeyProbDist",
    "ConditionalProbDist",
]


class FreqDist:
    """Frequency distribution — Rust-accelerated."""
    def __init__(self, samples=None):
        self._impl = _RustFreqDist(samples or [])

    def N(self):
        return self._impl.N()

    def B(self):
        return self._impl.B()

    def freq(self, sample):
        return self._impl.freq(sample)

    def max(self):
        return self._impl.max()

    def hapaxes(self):
        return self._impl.hapaxes()

    def samples(self):
        return self._impl.samples()

    def update(self, samples):
        self._impl.update(samples)

    def most_common(self, n=None):
        return self._impl.most_common(n)

    def __getitem__(self, sample):
        return self._impl[sample]

    def __len__(self):
        return len(self._impl)

    def __contains__(self, sample):
        return self._impl.__contains__(sample)

    def __add__(self, other):
        return self._impl + other

    def copy(self):
        return self._impl.copy()

    def keys(self):
        return self._impl.keys()

    def values(self):
        return self._impl.values()

    def items(self):
        return self._impl.items()

    def __iter__(self):
        return iter(self._impl)

    def tabulate(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).tabulate(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).plot(*args, **kwargs)


class ConditionalFreqDist:
    """Conditional frequency distribution — Rust-accelerated."""
    def __init__(self):
        self._impl = _RustConditionalFreqDist()

    def __getitem__(self, condition):
        return self._impl.__getitem__(condition)

    def conditions(self):
        return self._impl.conditions()

    def N(self):
        return self._impl.N()

    def inc(self, condition, sample):
        self._impl.inc(condition, sample)

    def __len__(self):
        return self._impl.__len__()

    def tabulate(self, *args, **kwargs):
        return _nltk_probability.ConditionalFreqDist().tabulate(*args, **kwargs)


# ── NLTK re-exports ─────
ConditionalProbDistI = _nltk_probability.ConditionalProbDistI
CrossValidationProbDist = _nltk_probability.CrossValidationProbDist
DictionaryConditionalProbDist = _nltk_probability.DictionaryConditionalProbDist
DictionaryProbDist = _nltk_probability.DictionaryProbDist
ELEProbDist = _nltk_probability.ELEProbDist
HeldoutProbDist = _nltk_probability.HeldoutProbDist
ImmutableProbabilisticMixIn = _nltk_probability.ImmutableProbabilisticMixIn
MutableProbDist = _nltk_probability.MutableProbDist
ProbabilisticMixIn = _nltk_probability.ProbabilisticMixIn
RandomProbDist = _nltk_probability.RandomProbDist
UniformProbDist = _nltk_probability.UniformProbDist
add_logs = _nltk_probability.add_logs
demo = _nltk_probability.demo
entropy = _nltk_probability.entropy
gt_demo = _nltk_probability.gt_demo
log_likelihood = _nltk_probability.log_likelihood
sum_logs = _nltk_probability.sum_logs
