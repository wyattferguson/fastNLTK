"""
fastnltk.data — Drop-in replacement for nltk.data.

Resolves and loads NLTK data files (models, corpora, tokenizers, taggers).
"""

import pickle as _pickle

from nltk.data import *  # noqa: F403
from nltk.data import find as _find

# ── Rust model loading helpers ───────────────────────────


def load_punkt_model(language="english"):
    """Load a Punkt tokenizer model from nltk_data and return as dict."""
    path = _find(f"tokenizers/punkt/{language}.pickle")
    with open(path, "rb") as f:
        return _pickle.load(f)


def load_perceptron_tagger():
    """Load averaged perceptron tagger weights from nltk_data."""
    import os as _os

    path = _find("taggers/averaged_perceptron_tagger/")
    weights = {}
    for fname in ["weights.pickle", "tagdict.pickle", "classes.pickle"]:
        fpath = _os.path.join(path, fname)
        if _os.path.exists(fpath):
            with open(fpath, "rb") as f:
                key = fname.replace(".pickle", "")
                weights[key] = _pickle.load(f)
    return weights
