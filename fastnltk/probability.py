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
        if isinstance(samples, str):
            self._impl = _RustFreqDist(list(samples))
        else:
            self._impl = _RustFreqDist(samples or [])

    def N(self) -> int:
        return self._impl.N()

    def B(self) -> int:
        return self._impl.B()

    def freq(self, sample: str) -> int:
        return self._impl.freq(sample)

    def max(self) -> str | None:
        return self._impl.max()

    def hapaxes(self) -> list[str]:
        return self._impl.hapaxes()

    def samples(self) -> list[str]:
        return self._impl.samples()

    def update(self, samples: any) -> None:
        self._impl.update(samples)

    def most_common(self, n: int | None = None) -> list[tuple[str, int]]:
        return self._impl.most_common(n)

    def __getitem__(self, sample):
        return self._impl[sample]

    def get(self, sample: str, default: int = 0) -> int:
        try:
            if sample in self._impl:
                return self._impl[sample]
            return default
        except Exception:
            return default

    def __setitem__(self, sample: str, count: int) -> None:
        current = self._impl[sample] if sample in self._impl else 0
        if count >= current:
            self._impl.inc(sample, count - current)

    def __len__(self):
        return len(self._impl)

    def __contains__(self, sample):
        return self._impl.__contains__(sample)

    def __add__(self, other):
        return self._impl + other

    def copy(self) -> any:
        return self._impl.copy()

    def keys(self) -> list[str]:
        return self._impl.samples()

    def values(self) -> list[int]:
        return self._impl.values()

    def items(self) -> list[tuple[str, int]]:
        return self._impl.items()

    def inc(self, sample: str, count: int = 1) -> None:
        self._impl.inc(sample, count)

    def __iter__(self):
        return iter(self._impl.samples())

    def tabulate(self, *args: any, **kwargs: any) -> None:
        return _nltk_probability.FreqDist(self).tabulate(*args, **kwargs)

    def plot(self, *args: any, **kwargs: any) -> None:
        return _nltk_probability.FreqDist(self).plot(*args, **kwargs)


class ConditionalFreqDist:
    """Conditional frequency distribution — Rust-accelerated."""

    def __init__(self):
        self._impl = _RustConditionalFreqDist()

    def __getitem__(self, condition):
        # Check if condition exists by trying to fetch it.
        # PyO3 __contains__ is not exposed, so fallback to checking __getitem__.
        result = self._impl[condition]
        if result is None:
            # Auto-create: add a dummy sample to initialize the condition.
            self._impl.inc(condition, "")
            result = self._impl[condition]
        wrapped = FreqDist.__new__(FreqDist)
        wrapped._impl = result
        return wrapped

    def conditions(self) -> list[str]:
        return self._impl.conditions()

    def N(self):
        return self._impl.N()

    def inc(self, condition: str, sample: str) -> None:
        self._impl.inc(condition, sample)

    def __len__(self):
        return self._impl.__len__()

    def tabulate(self, *args: any, **kwargs: any) -> None:
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
