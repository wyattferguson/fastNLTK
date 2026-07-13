"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.
"""

_rust_available = False
try:
    from fastnltk._rust import (
        edit_distance as _rust_edit_distance_fn,
        jaro_similarity as _rust_jaro_similarity,
        jaro_winkler_similarity as _rust_jaro_winkler_similarity,
        dice_similarity as _rust_dice_similarity,
        jaccard_distance as _rust_jaccard_distance,
        binary_distance as _rust_binary_distance,
        masi_distance as _rust_masi_distance,
    )
    _rust_available = True
except ImportError:
    pass

# Import from nltk.metrics — gracefully handle version differences
import nltk.metrics as _nltk_metrics

# Safe imports
def _safe_import(name):
    return getattr(_nltk_metrics, name, None)

try:
    from nltk.metrics import (
        BigramAssocMeasures,
        TrigramAssocMeasures,
        QuadgramAssocMeasures,
        ContingencyMeasures,
        ConfusionMatrix,
        AnnotationTask,
        precision,
        recall,
        f_measure,
        log_likelihood,
        approxrand,
    )
except ImportError:
    BigramAssocMeasures = _safe_import("BigramAssocMeasures")
    TrigramAssocMeasures = _safe_import("TrigramAssocMeasures")
    QuadgramAssocMeasures = _safe_import("QuadgramAssocMeasures")
    ConfusionMatrix = _safe_import("ConfusionMatrix")
    precision = _safe_import("precision")
    recall = _safe_import("recall")
    f_measure = _safe_import("f_measure")

try:
    from nltk.metrics.distance import (
        jaccard_distance,
        binary_distance,
        masi_distance,
        interval_distance,
        custom_distance,
    )
except ImportError:
    jaccard_distance = binary_distance = masi_distance = None
    interval_distance = custom_distance = None

windowdiff = _safe_import("windowdiff")
pk = _safe_import("pk")
bcubed = _safe_import("bcubed")

__all__ = [
    "edit_distance",
    "jaccard_distance",
    "binary_distance",
    "masi_distance",
    "BigramAssocMeasures",
    "TrigramAssocMeasures",
    "ConfusionMatrix",
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
