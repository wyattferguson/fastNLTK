"""
fastnltk.lm — Drop-in replacement for nltk.lm.
"""

from nltk.lm import (
    MLE,
    KneserNey,
    Laplace,
    Lidstone,
    Smoothing,
    StupidBackoff,
    Vocabulary,
    WittenBell,
)
from nltk.lm.counter import NgramCounter
from nltk.lm.preprocessing import (
    everygrams,
    pad_both_ends,
    pad_sequence,
    padded_everygrams,
)
from nltk.lm.util import log_base2

__all__ = [
    "Vocabulary",
    "MLE",
    "Lidstone",
    "Laplace",
    "KneserNey",
    "StupidBackoff",
    "WittenBell",
    "NgramCounter",
    "Smoothing",
    "padded_everygrams",
    "everygrams",
    "pad_both_ends",
    "pad_sequence",
    "log_base2",
]
