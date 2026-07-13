"""
fastnltk.lm — Drop-in replacement for nltk.lm.
"""

import nltk.lm as _nltk_lm
from nltk.lm import (
    Vocabulary,
    MLE,
    Lidstone,
    Laplace,
    KneserNey,
    StupidBackoff,
    WittenBell,
    Smoothing,
)

from nltk.lm.preprocessing import (
    padded_everygrams,
    pad_both_ends,
    everygrams,
    everygram_pad,
    pad_sequence,
)
from nltk.lm.counter import NgramCounter
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
