"""
fastnltk.tag — Drop-in replacement for nltk.tag.

Delegates to compiled Rust extension where available,
falls back to original nltk.tag for unimplemented pieces.
"""

import functools
import os
import pickle
import warnings

import nltk.tag as _nltk_tag
from nltk.data import find
from nltk.tag import (
    AffixTagger,
    BigramTagger,
    BrillTagger,
    BrillTaggerTrainer,
    ClassifierBasedPOSTagger,
    ClassifierBasedTagger,
    ContextTagger,
    CRFTagger,
    DefaultTagger,
    HiddenMarkovModelTagger,
    HiddenMarkovModelTrainer,
    NgramTagger,
    RegexpTagger,
    SequentialBackoffTagger,
    TaggerI,
    TrigramTagger,
    UnigramTagger,
    map_tag,
)

_rust_available = False
try:
    from fastnltk._rust import PerceptronTagger as _RustPerceptronTagger
    from fastnltk._rust import TnT as _RustTnT
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK tagger"
    )


def _load_tagger_model(tagger):
    """Load NLTK perceptron tagger weights into a Rust or NLTK tagger."""
    path = find("taggers/averaged_perceptron_tagger/")
    pickle_path = os.path.join(str(path), "averaged_perceptron_tagger.pickle")
    with open(pickle_path, "rb") as f:
        model_data = pickle.load(f)
    tagger.load(model_data[0], model_data[1], sorted(model_data[2]))

__all__ = [
    "pos_tag",
    "pos_tag_sents",
    "TaggerI",
    "PerceptronTagger",
    "DefaultTagger",
    "NgramTagger",
    "UnigramTagger",
    "BigramTagger",
    "TrigramTagger",
    "AffixTagger",
    "RegexpTagger",
    "TnT",
    "BrillTagger",
    "BrillTaggerTrainer",
    "HiddenMarkovModelTagger",
    "HiddenMarkovModelTrainer",
    "CRFTagger",
    "SequentialBackoffTagger",
    "ContextTagger",
    "ClassifierBasedTagger",
    "ClassifierBasedPOSTagger",
]


@functools.lru_cache(maxsize=1)
def _get_tagger():
    """Lazy-loaded Rust perceptron tagger with NLTK weights."""
    tagger = _RustPerceptronTagger()
    try:
        _load_tagger_model(tagger)
    except (LookupError, FileNotFoundError, OSError) as e:
        raise RuntimeError(
            "NLTK perceptron tagger data not found. "
            "Run: python -m nltk.downloader averaged_perceptron_tagger"
        ) from e
    return tagger


def pos_tag(tokens, tagset=None, lang="eng"):
    """POS tagging (Rust-accelerated).

    Returns a list of (word, tag) tuples.
    Falls back to NLTK if Rust tagger is unavailable.
    """
    if not _rust_available:
        return _nltk_tag.pos_tag(tokens, tagset, lang)
    try:
        tagger = _get_tagger()
        result = tagger.tag(tokens)
        if tagset:
            return [
                (word, map_tag("en-ptb", tagset, tag) if tag else tag)
                for word, tag in result
            ]
        return result
    except (ValueError, LookupError, RuntimeError):
        return _nltk_tag.pos_tag(tokens, tagset, lang)


def pos_tag_sents(sentences, tagset=None, lang="eng"):
    """POS tagging for multiple sentences (Rust-accelerated)."""
    if not _rust_available:
        return _nltk_tag.pos_tag_sents(sentences, tagset, lang)
    try:
        tagger = _get_tagger()
        results = tagger.tag_sents(sentences)
        if tagset:
            return [
                [(word, map_tag("en-ptb", tagset, tag) if tag else tag)
                 for word, tag in sent]
                for sent in results
            ]
        return results
    except (ValueError, LookupError, RuntimeError):
        return _nltk_tag.pos_tag_sents(sentences, tagset, lang)


class PerceptronTagger:
    """Rust-accelerated averaged perceptron tagger."""
    def __init__(self, load_model=True):
        if not _rust_available:
            self._impl = _nltk_tag.PerceptronTagger()
            return
        self._impl = _RustPerceptronTagger()
        if load_model:
            try:
                _load_tagger_model(self._impl)
            except (LookupError, FileNotFoundError, OSError):
                pass

    def tag(self, tokens):
        return self._impl.tag(tokens)

    def tag_sents(self, sentences):
        return self._impl.tag_sents(sentences)


class TnT:
    """TnT trigram HMM tagger — Rust-accelerated."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustTnT()
        else:
            self._impl = _nltk_tag.TnT()

    def train(self, sentences):
        self._impl.train(sentences)

    def tag(self, words):
        return self._impl.tag(words)

    def tag_sents(self, sentences):
        return self._impl.tag_sents(sentences)
