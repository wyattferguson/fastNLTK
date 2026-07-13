"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.
"""

from fastnltk._rust import (
    edit_distance as _rust_edit_distance,
    jaro_similarity,
    jaro_winkler_similarity,
    dice_similarity,
)

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

from nltk.metrics.segmentation import windowdiff as _nltk_windowdiff
from nltk.metrics.scores import (
    precision as _nltk_precision,
    recall as _nltk_recall,
    f_measure as _nltk_f_measure,
)

__all__ = [
    "edit_distance",
    "jaccard_distance",
    "binary_distance",
    "masi_distance",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "dice_similarity",
    "BigramAssocMeasures",
    "TrigramAssocMeasures",
    "ConfusionMatrix",
    "precision",
    "recall",
    "f_measure",
]


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """Rust-accelerated edit distance."""
    return _rust_edit_distance(s1, s2, substitution_cost, transpositions)
