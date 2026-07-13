"""
fastnltk.probability — Drop-in replacement for nltk.probability.
"""

from fastnltk._rust import (
    FreqDist as _RustFreqDist,
)

import nltk.probability as _nltk_probability
from nltk.probability import (
    ProbDistI,
    UniformProbDist,
    DictionaryProbDist,
    MLEProbDist,
    LidstoneProbDist,
    LaplaceProbDist,
    ELEProbDist,
    HeldoutProbDist,
    CrossValidationProbDist,
    WittenBellProbDist,
    SimpleGoodTuringProbDist,
    MutableProbDist,
    KneserNeyProbDist,
    ConditionalFreqDist,
    ConditionalProbDistI,
    ConditionalProbDist,
    ImmutableProbabilisticMixIn,
    ProbabilisticMixIn,
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
    """Rust-accelerated frequency distribution."""
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
        return self._impl.add(other)

    def copy(self):
        return self._impl.copy()

    def keys(self):
        return self._impl.keys()

    def values(self):
        return self._impl.values()

    def items(self):
        return self._impl.items()

    def __iter__(self):
        return self._impl.__iter__()

    def tabulate(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).tabulate(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).plot(*args, **kwargs)

    def pformat(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).pformat(*args, **kwargs)

    def pprint(self, *args, **kwargs):
        return _nltk_probability.FreqDist(self).pprint(*args, **kwargs)
