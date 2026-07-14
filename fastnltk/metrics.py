"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.
"""

import warnings

import nltk.metrics as _nltk_metrics

# ── NLTK fallback imports (gracefully handle version differences) ────

try:
    from nltk.metrics import (
        BigramAssocMeasures,
        ConfusionMatrix,
        QuadgramAssocMeasures,
        TrigramAssocMeasures,
        f_measure,
        precision,
        recall,
    )
except ImportError:
    BigramAssocMeasures = getattr(_nltk_metrics, "BigramAssocMeasures", None)
    TrigramAssocMeasures = getattr(_nltk_metrics, "TrigramAssocMeasures", None)
    QuadgramAssocMeasures = getattr(_nltk_metrics, "QuadgramAssocMeasures", None)
    ConfusionMatrix = getattr(_nltk_metrics, "ConfusionMatrix", None)
    precision = getattr(_nltk_metrics, "precision", None)
    recall = getattr(_nltk_metrics, "recall", None)
    f_measure = getattr(_nltk_metrics, "f_measure", None)

try:
    from nltk.metrics.distance import (
        binary_distance,
        custom_distance,
        interval_distance,
        jaccard_distance,
        masi_distance,
    )
except ImportError:
    jaccard_distance = binary_distance = masi_distance = None
    interval_distance = custom_distance = None

# Re-export NLTK names for API compatibility
AnnotationTask = getattr(_nltk_metrics, "AnnotationTask", None)
ContingencyMeasures = getattr(_nltk_metrics, "ContingencyMeasures", None)
approxrand = getattr(_nltk_metrics, "approxrand", None)
log_likelihood = getattr(_nltk_metrics, "log_likelihood", None)
windowdiff = getattr(_nltk_metrics, "windowdiff", None)
pk = getattr(_nltk_metrics, "pk", None)
bcubed = getattr(_nltk_metrics, "bcubed", None)

_rust_available = False
try:
    from fastnltk._rust import binary_distance as _rust_binary_distance
    from fastnltk._rust import dice_similarity as _rust_dice_similarity
    from fastnltk._rust import edit_distance as _rust_edit_distance_fn
    from fastnltk._rust import jaccard_distance as _rust_jaccard_distance
    from fastnltk._rust import jaro_similarity as _rust_jaro_similarity
    from fastnltk._rust import jaro_winkler_similarity as _rust_jaro_winkler_similarity
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK metrics"
    )

__all__ = [
    "edit_distance",
    "jaccard_distance",
    "binary_distance",
    "masi_distance",
    "BigramAssocMeasures",
    "TrigramAssocMeasures",
    "ConfusionMatrix",
    "AnnotationTask",
    "ContingencyMeasures",
    "approxrand",
    "log_likelihood",
    "windowdiff",
    "pk",
    "bcubed",
    "precision",
    "recall",
    "f_measure",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "dice_similarity",
]


def jaccard_distance(s1, s2):
    if _rust_available:
        return _rust_jaccard_distance(s1, s2)
    from nltk.metrics.distance import jaccard_distance as _nltk_j

    return _nltk_j(set(s1 or []), set(s2 or []))


def binary_distance(s1, s2):
    if _rust_available:
        return _rust_binary_distance(s1, s2)
    from nltk.metrics.distance import binary_distance as _nltk_b

    return _nltk_b(set(s1 or []), set(s2 or []))


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    if _rust_available:
        return _rust_edit_distance_fn(s1, s2, substitution_cost, transpositions)
    from nltk.metrics.distance import edit_distance as _nltk_edit_distance

    return _nltk_edit_distance(s1, s2, substitution_cost, transpositions)


def jaro_similarity(x, y):
    if _rust_available:
        return _rust_jaro_similarity(x, y)
    from nltk.metrics.distance import jaro_similarity as _nltk_jaro

    return _nltk_jaro(x, y)


def jaro_winkler_similarity(x, y, p=0.1, max_l=4):
    if _rust_available:
        return _rust_jaro_winkler_similarity(x, y, p, max_l)
    from nltk.metrics.distance import jaro_winkler_similarity as _nltk_jw

    return _nltk_jw(x, y, p, max_l)


def dice_similarity(x, y):
    if _rust_available:
        return _rust_dice_similarity(x, y)
    from nltk.metrics.distance import jaccard_distance

    return 1.0 - jaccard_distance(set(x.split()), set(y.split()))


alignment_error_rate = _nltk_metrics.alignment_error_rate
