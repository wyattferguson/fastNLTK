"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.
"""

_rust_available = False
try:
    from fastnltk._rust import (
        edit_distance as _rust_edit_distance_fn,
    )
    _rust_available = True
except ImportError:
    pass

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
    windowdiff,
    pk,
    bcubed,
)

from nltk.metrics.distance import (
    jaccard_distance,
    binary_distance,
    masi_distance,
    interval_distance,
    custom_distance,
    presence,
    fractional_presence,
)

from nltk.metrics import (
    precision as _nltk_precision,
    recall as _nltk_recall,
    f_measure as _nltk_f_measure,
)

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
    """Edit distance — Rust-accelerated when available."""
    if _rust_available:
        return _rust_edit_distance_fn(s1, s2, substitution_cost, transpositions)
    from nltk.metrics.distance import edit_distance as _nltk_edit_distance
    return _nltk_edit_distance(s1, s2, substitution_cost, transpositions)
