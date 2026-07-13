"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.
"""

_rust_available = False
try:
    from fastnltk._rust import edit_distance as _rust_edit_distance_fn
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
]


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """Edit distance — Rust-accelerated."""
    if _rust_available:
        return _rust_edit_distance_fn(s1, s2, substitution_cost, transpositions)
    from nltk.metrics.distance import edit_distance as _nltk_edit_distance
    return _nltk_edit_distance(s1, s2, substitution_cost, transpositions)
