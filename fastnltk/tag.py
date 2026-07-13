"""
fastnltk.tag — Drop-in replacement for nltk.tag.

Delegates to compiled Rust extension where available,
falls back to original nltk.tag for unimplemented pieces.
"""

import functools
import os
import pickle

_rust_available = False
try:
    from fastnltk._rust import PerceptronTagger as _RustPerceptronTagger
    _rust_available = True
except ImportError:
    pass

import nltk.tag as _nltk_tag
from nltk.tag import (
    TaggerI,
    SequentialBackoffTagger,
    ContextTagger,
    DefaultTagger,
    NgramTagger,
    UnigramTagger,
    BigramTagger,
    TrigramTagger,
    AffixTagger,
    RegexpTagger,
    ClassifierBasedTagger,
    ClassifierBasedPOSTagger,
    TnT,
    HiddenMarkovModelTagger,
    HiddenMarkovModelTrainer,
    BrillTagger,
    BrillTaggerTrainer,
    HunposTagger,
    StanfordTagger,
    StanfordPOSTagger,
    StanfordNERTagger,
    SennaTagger,
    SennaChunkTagger,
    SennaNERTagger,
    CRFTagger,
    str2tuple,
    tuple2str,
    untag,
    tagset_mapping,
    map_tag,
)

from nltk.data import load, find

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
        path = find("taggers/averaged_perceptron_tagger/")
        pickle_path = os.path.join(str(path), "averaged_perceptron_tagger.pickle")
        with open(pickle_path, "rb") as f:
            model_data = pickle.load(f)

        weights_dict = model_data[0]
        tagdict = model_data[1]
        classes = sorted(model_data[2])

        tagger.load(weights_dict, tagdict, classes)
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
                path = find("taggers/averaged_perceptron_tagger/")
                pickle_path = os.path.join(str(path), "averaged_perceptron_tagger.pickle")
                with open(pickle_path, "rb") as f:
                    model_data = pickle.load(f)
                self._impl.load(model_data[0], model_data[1], sorted(model_data[2]))
            except (LookupError, FileNotFoundError, OSError):
                pass

    def tag(self, tokens):
        return self._impl.tag(tokens)

    def tag_sents(self, sentences):
        return self._impl.tag_sents(sentences)
