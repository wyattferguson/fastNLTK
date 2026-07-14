"""
fastnltk.data — Drop-in replacement for nltk.data.

Resolves and loads NLTK data files (models, corpora, tokenizers, taggers).
"""

import os
import pickle

import nltk.data as _nltk_data

# ── Data path resolution ─────────────────────────────────

def find(resource_name):
    """Find the path to an NLTK resource file.

    Uses the same search order as nltk.data.find.
    """
    return _nltk_data.find(resource_name)


def load(resource_name, format="auto"):
    """Load an NLTK resource file.

    Supports: pickle, json, raw, auto.
    """
    return _nltk_data.load(resource_name, format)


def path(resource_name):
    """Get the absolute file path for a resource name."""
    return _nltk_data.path(resource_name)


def show_cfg(resource_name):
    """Show the configuration for a resource."""
    _nltk_data.show_cfg(resource_name)


# ── Rust model loading ───────────────────────────────────

def load_punkt_model(language="english"):
    """Load a Punkt tokenizer model from nltk_data and return as dict."""
    path = find(f"tokenizers/punkt/{language}.pickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_perceptron_tagger():
    """Load averaged perceptron tagger weights from nltk_data."""
    path = find("taggers/averaged_perceptron_tagger/")
    weights = {}
    for fname in ["weights.pickle", "tagdict.pickle", "classes.pickle"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                key = fname.replace(".pickle", "")
                weights[key] = pickle.load(f)
    return weights


def data_dirs():
    """Return list of NLTK data directories."""
    return _nltk_data.path
