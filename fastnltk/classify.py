"""
fastnltk.classify — Drop-in replacement for nltk.classify.
"""

from fastnltk._rust import (
    NaiveBayesClassifier as _RustNaiveBayesClassifier,
)

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
    """Rust-accelerated Naive Bayes classifier."""
    def __init__(self):
        self._impl = _RustNaiveBayesClassifier()

    @classmethod
    def train(cls, labeled_featuresets, estimator=None, **kwargs):
        """Train a Naive Bayes classifier."""
        inst = cls()
        inst._impl.train(labeled_featuresets)
        return inst

    def classify(self, features):
        return self._impl.classify(features)

    def labels(self):
        return self._impl.labels()

    def prob_classify(self, features):
        """Return a probability distribution for labels."""
        try:
            return self._impl.prob_classify(features)
        except (ValueError, RuntimeError):
            return _nltk_classify.NaiveBayesClassifier.train([]).prob_classify(features)

    def show_most_informative_features(self, n=10):
        """Show the most informative features."""
        try:
            return self._impl.show_most_informative_features(n)
        except (ValueError, RuntimeError):
            pass
