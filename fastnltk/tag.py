"""
fastnltk.tag — Drop-in replacement for nltk.tag.
"""

import functools

from fastnltk._rust import (
    PerceptronTagger as _RustPerceptronTagger,
    PosTagger as _RustPosTagger,
)

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
    """Lazy-loaded perceptron tagger."""
    return _RustPerceptronTagger()


def pos_tag(tokens, tagset=None, lang="eng"):
    """POS tagging (Rust-accelerated).

    Returns a list of (word, tag) tuples.
    Falls back to NLTK if Rust tagger is unavailable.
    """
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
    try:
        tagger = _get_tagger()
        results = tagger.tag_sents(sentences)
        if tagset:
            return [
                [(word, map_tag("en-ptb", tagset, tag) if tag else tag) for word, tag in sent]
                for sent in results
            ]
        return results
    except (ValueError, LookupError, RuntimeError):
        return _nltk_tag.pos_tag_sents(sentences, tagset, lang)


class PerceptronTagger:
    """Rust-accelerated averaged perceptron tagger."""
    def __init__(self):
        self._impl = _RustPerceptronTagger()

    def tag(self, tokens):
        return self._impl.tag(tokens)

    def tag_sents(self, sentences):
        return self._impl.tag_sents(sentences)
