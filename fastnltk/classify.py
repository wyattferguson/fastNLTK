"""
fastnltk.classify — Drop-in replacement for nltk.classify.
"""

_rust_available = False
try:
    from fastnltk._rust import NaiveBayesClassifier as _RustNaiveBayesClassifier
    _rust_available = True
except ImportError:
    pass

import nltk.classify as _nltk_classify
from nltk.classify import (
    ClassifierI,
    MultiClassifierI,
    DecisionTreeClassifier,
    MaxentClassifier,
    BinaryMaxentFeatureEncoding,
    TypedMaxentFeatureEncoding,
    SklearnClassifier,
    TextCat,
    PositiveNaiveBayesClassifier,
    call_megam,
    config_megam,
)

from nltk.classify.util import accuracy, apply_features, log_likelihood

__all__ = [
    "NaiveBayesClassifier",
    "ClassifierI",
    "DecisionTreeClassifier",
    "MaxentClassifier",
    "TextCat",
    "accuracy",
    "apply_features",
    "log_likelihood",
]


class NaiveBayesClassifier:
    """Naive Bayes classifier — Rust-accelerated when available."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustNaiveBayesClassifier()
        else:
            self._impl = None  # set by train()

    @classmethod
    def train(cls, labeled_featuresets, estimator=None, **kwargs):
        """Train a Naive Bayes classifier."""
        inst = cls()
        if _rust_available:
            inst._impl.train(labeled_featuresets)
        else:
            inst._impl = _nltk_classify.NaiveBayesClassifier.train(
                labeled_featuresets, estimator, **kwargs
            )
        return inst

    def classify(self, features):
        if _rust_available:
            return self._impl.classify(features)
        return self._impl.classify(features)

    def labels(self):
        if _rust_available:
            return self._impl.labels()
        return self._impl.labels()

    def prob_classify(self, features):
        return self._impl.prob_classify(features)

    def show_most_informative_features(self, n=10):
        return self._impl.show_most_informative_features(n)
