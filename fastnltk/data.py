"""
fastnltk.data — Drop-in replacement for nltk.data.

Resolves and loads NLTK data files (models, corpora, tokenizers, taggers).
find() uses Rust-accelerated path resolution.
"""

import pickle as _pickle

from nltk.data import *  # noqa: F403

try:
    from fastnltk._rust import find as _rust_find
except ImportError:
    _rust_find = None


def find(resource_name, paths=None):
    """Find an NLTK resource file by name — Rust-accelerated path resolution.

    Searches standard nltk_data directories (NLTK_DATA env, ~/nltk_data,
    /usr/share/nltk_data, etc.) for the given resource.
    """
    if _rust_find is not None:
        try:
            return _rust_find(resource_name)
        except LookupError:
            pass
    # Fall back to NLTK for broader search paths (sys.path, etc.)
    from nltk.data import find as _nltk_find

    return _nltk_find(resource_name, paths)


# ── Rust model loading helpers ───────────────────────────


def load_punkt_model(language="english"):
    """Load a Punkt tokenizer model from nltk_data and return as dict."""
    path = find(f"tokenizers/punkt/{language}.pickle")
    with open(path, "rb") as f:
        return _pickle.load(f)


def load_perceptron_tagger():
    """Load averaged perceptron tagger weights from nltk_data."""
    import os as _os

    path = find("taggers/averaged_perceptron_tagger/")
    weights = {}
    for fname in ["weights.pickle", "tagdict.pickle", "classes.pickle"]:
        fpath = _os.path.join(path, fname)
        if _os.path.exists(fpath):
            with open(fpath, "rb") as f:
                key = fname.replace(".pickle", "")
                weights[key] = _pickle.load(f)
    return weights
