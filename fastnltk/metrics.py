"""
fastnltk.metrics — Drop-in replacement for nltk.metrics.

All metric functions are Rust-accelerated via the compiled `_rust` extension.
"""

import nltk.metrics as _nltk_metrics

from fastnltk._rust import (
    BigramAssocMeasures as _RustBigramAssocMeasures,
)
from fastnltk._rust import (
    alpha as _rust_alpha,
)
from fastnltk._rust import (
    binary_distance as _rust_binary_distance,
)
from fastnltk._rust import (
    dice_similarity as _rust_dice_similarity,
)
from fastnltk._rust import (
    edit_distance as _rust_edit_distance_fn,
)
from fastnltk._rust import (
    jaccard_distance as _rust_jaccard_distance,
)
from fastnltk._rust import (
    jaro_similarity as _rust_jaro_similarity,
)
from fastnltk._rust import (
    jaro_winkler_similarity as _rust_jaro_winkler_similarity,
)
from fastnltk._rust import (
    kappa as _rust_kappa,
)
from fastnltk._rust import (
    pi as _rust_pi,
)
from fastnltk._rust import (
    pk as _rust_pk,
)
from fastnltk._rust import (
    spearman as _rust_spearman,
)
from fastnltk._rust import (
    windowdiff as _rust_windowdiff,
)

# NLTK re-exports for API compatibility
try:
    from nltk.metrics import (
        ConfusionMatrix,
        QuadgramAssocMeasures,
        TrigramAssocMeasures,
        f_measure,
        precision,
        recall,
    )
except ImportError:
    ConfusionMatrix = getattr(_nltk_metrics, "ConfusionMatrix", None)
    TrigramAssocMeasures = getattr(_nltk_metrics, "TrigramAssocMeasures", None)
    QuadgramAssocMeasures = getattr(_nltk_metrics, "QuadgramAssocMeasures", None)
    precision = getattr(_nltk_metrics, "precision", None)
    recall = getattr(_nltk_metrics, "recall", None)
    f_measure = getattr(_nltk_metrics, "f_measure", None)

try:
    from nltk.metrics.distance import (
        custom_distance,
        interval_distance,
        masi_distance,
    )
except ImportError:
    masi_distance = interval_distance = custom_distance = None

AnnotationTask = getattr(_nltk_metrics, "AnnotationTask", None)
ContingencyMeasures = getattr(_nltk_metrics, "ContingencyMeasures", None)
approxrand = getattr(_nltk_metrics, "approxrand", None)
log_likelihood = getattr(_nltk_metrics, "log_likelihood", None)
bcubed = getattr(_nltk_metrics, "bcubed", None)

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
    # NLTK accepts both sets and strings
    if isinstance(s1, set) and isinstance(s2, set):
        if not s1 and not s2:
            return 0.0
        return 1 - len(s1 & s2) / len(s1 | s2)
    return _rust_jaccard_distance(s1, s2)


def binary_distance(s1, s2):
    # NLTK accepts both sets and strings
    if isinstance(s1, set) and isinstance(s2, set):
        return 0.0 if s1 == s2 else 1.0
    return _rust_binary_distance(s1, s2)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    return _rust_edit_distance_fn(s1, s2, substitution_cost, transpositions)


def jaro_similarity(x, y):
    return _rust_jaro_similarity(x, y)


def jaro_winkler_similarity(x, y, p=0.1, max_l=4):
    return _rust_jaro_winkler_similarity(x, y, p, max_l)


def dice_similarity(x, y):
    return _rust_dice_similarity(x, y)


# Rust exports with set/list→str conversion for NLTK compat
def windowdiff(reference, hypothesis, k=3, boundary="1"):
    if isinstance(reference, list):
        reference = "".join(str(b) for b in reference)
    if isinstance(hypothesis, list):
        hypothesis = "".join(str(b) for b in hypothesis)
    return _rust_windowdiff(reference, hypothesis, k, boundary)


def pk(reference, hypothesis, k=None, boundary="1"):
    if isinstance(reference, list):
        reference = "".join(str(b) for b in reference)
    if isinstance(hypothesis, list):
        hypothesis = "".join(str(b) for b in hypothesis)
    return _rust_pk(
        reference, hypothesis, k if k is not None else max(1, len(reference) // 10), boundary
    )


kappa = _rust_kappa
pi = _rust_pi
alpha = _rust_alpha
spearman = _rust_spearman
BigramAssocMeasures = _RustBigramAssocMeasures

# NLTK re-exports
alignment_error_rate = _nltk_metrics.alignment_error_rate
