"""
fastnltk.classify — Drop-in replacement for nltk.classify.
"""

import warnings

import nltk.classify as _nltk_classify
from nltk.classify import ClassifierI, DecisionTreeClassifier
from nltk.classify.util import accuracy, apply_features, log_likelihood

_rust_available = False
try:
    from fastnltk._rust import MaxentClassifier as _RustMaxentClassifier
    from fastnltk._rust import NaiveBayesClassifier as _RustNaiveBayesClassifier
    from fastnltk._rust import TextCat as _RustTextCat
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK classifiers"
    )

__all__ = [
    "NaiveBayesClassifier",
    "PositiveNaiveBayesClassifier",
    "ClassifierI",
    "DecisionTreeClassifier",
    "MaxentClassifier",
    "TextCat",
    "accuracy",
    "apply_features",
    "log_likelihood",
]


class TextCat:
    """Language detection — Rust-accelerated via whatlang."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustTextCat()
        else:
            self._impl = _nltk_classify.TextCat()

    def guess_language(self, text):
        return self._impl.guess_language(text)

    def guess_language_scores(self, text):
        return self._impl.guess_language_scores(text)

    @staticmethod
    def supported_languages():
        return _RustTextCat.supported_languages() if _rust_available else ["unknown"]


class MaxentClassifier:
    """Maximum Entropy classifier — Rust-accelerated GIS training."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustMaxentClassifier()
        else:
            self._impl = None  # set by train()

    @classmethod
    def train(cls, labeled_featuresets, max_iter=100, algorithm="gis", **kwargs):
        """Train a Maxent classifier using GIS."""
        inst = cls()
        if _rust_available:
            sigma = kwargs.get("gaussian_prior_sigma", 0.0)
            inst._impl.train(labeled_featuresets, max_iter, sigma)
        else:
            inst._impl = _nltk_classify.MaxentClassifier.train(
                labeled_featuresets, algorithm, max_iter=max_iter, **kwargs
            )
        return inst

    def classify(self, features):
        return self._impl.classify(features)

    def labels(self):
        return self._impl.labels()

    def prob_classify(self, features):
        return self._impl.prob_classify(features)

    def show_most_informative_features(self, n=10):
        return self._impl.show_most_informative_features(n)


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


class PositiveNaiveBayesClassifier:
    """Positive Naive Bayes for positive + unlabeled data."""
    @staticmethod
    def train(positive_featuresets, unlabeled_featuresets):
        """Train from positive and unlabeled feature sets.

        Treats unlabeled as negative. Uses Rust NB when available.
        """
        from nltk.classify import PositiveNaiveBayesClassifier as _NltkPositiveNB

        if _rust_available:
            from fastnltk.classify import NaiveBayesClassifier

            labeled = [(feats, "pos") for feats in positive_featuresets] + [
                (feats, "neg") for feats in unlabeled_featuresets
            ]
            return NaiveBayesClassifier.train(labeled)
        return _NltkPositiveNB.train(positive_featuresets, unlabeled_featuresets)


# ── NLTK re-exports for API compatibility ─────

# ── NLTK re-exports for API compatibility ─────

BinaryMaxentFeatureEncoding = _nltk_classify.BinaryMaxentFeatureEncoding
ConditionalExponentialClassifier = _nltk_classify.ConditionalExponentialClassifier
MultiClassifierI = _nltk_classify.MultiClassifierI
SklearnClassifier = _nltk_classify.SklearnClassifier
TypedMaxentFeatureEncoding = _nltk_classify.TypedMaxentFeatureEncoding
WekaClassifier = _nltk_classify.WekaClassifier
call_megam = _nltk_classify.call_megam
config_megam = _nltk_classify.config_megam
config_weka = _nltk_classify.config_weka
tadm = _nltk_classify.tadm

api = _nltk_classify.api
decisiontree = _nltk_classify.decisiontree
maxent = _nltk_classify.maxent
megam = _nltk_classify.megam
naivebayes = _nltk_classify.naivebayes
positivenaivebayes = _nltk_classify.positivenaivebayes
rte_classifier = _nltk_classify.rte_classifier
rte_classify = _nltk_classify.rte_classify
rte_features = _nltk_classify.rte_features
scikitlearn = _nltk_classify.scikitlearn
senna = _nltk_classify.senna
textcat = _nltk_classify.textcat
util = _nltk_classify.util
weka = _nltk_classify.weka
RTEFeatureExtractor = _nltk_classify.RTEFeatureExtractor
Senna = _nltk_classify.Senna
