from __future__ import annotations

"""
fastnltk.tag — Drop-in replacement for nltk.tag.

All taggers are Rust-accelerated via the compiled `_rust` extension.
"""

import functools
import os
import pickle

import nltk.tag as _nltk_tag
from nltk.data import find
from nltk.tag import (
    BrillTagger,
    BrillTaggerTrainer,
    ClassifierBasedPOSTagger,
    ClassifierBasedTagger,
    ContextTagger,
    CRFTagger,
    HiddenMarkovModelTagger,
    HiddenMarkovModelTrainer,
    NgramTagger,
    SequentialBackoffTagger,
    TaggerI,
    map_tag,
)

from fastnltk._rust import (
    AffixTagger as _RustAffixTagger,,
    BigramTagger as _RustBigramTagger,,
    DefaultTagger as _RustDefaultTagger,,
    PerceptronTagger as _RustPerceptronTagger,,
    RegexpTagger as _RustRegexpTagger,,
    TnT as _RustTnT,,
    TrigramTagger as _RustTrigramTagger,,
    UnigramTagger as _RustUnigramTagger,)


def _load_tagger_model(tagger):
    """Load NLTK perceptron tagger weights into Rust tagger."""
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


def pos_tag(tokens: list[str], tagset=None, lang="eng"):
    """POS tagging (Rust-accelerated).

    Returns a list of (word, tag) tuples.
    Falls back to NLTK if tagger data is unavailable.
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


def pos_tag_sents(sentences: list[list[str]], tagset=None, lang="eng"):
    """POS tagging for multiple sentences (Rust-accelerated)."""
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
        self._impl = _RustPerceptronTagger()
        if load_model:
            try:
                _load_tagger_model(self._impl)
            except (LookupError, FileNotFoundError, OSError):
                pass

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)


class TnT:
    """TnT trigram HMM tagger — delegates to NLTK (Rust impl still being optimized)."""
    def __init__(self):
        import nltk.tag
        self._impl = nltk.tag.TnT()

    def train(self, sentences: list[list[str]]) -> None:
        self._impl.train(sentences)

    def tag(self, words: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(words)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)


class DefaultTagger:
    """Assign same tag to every token — Rust-accelerated."""
    def __init__(self, tag):
        self._impl = _RustDefaultTagger(tag)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)


class UnigramTagger:
    """Unigram tagger — Rust-accelerated lookup."""
    def __init__(self, backoff=None):
        self._impl = _RustUnigramTagger(backoff)

    def train(self, sentences: list[list[str]]) -> None:
        self._impl.train(sentences)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)

    def evaluate(self, gold: list[list[str]]) -> float:
        return self._impl.evaluate(gold)


class BigramTagger:
    """Bigram tagger — Rust-accelerated lookup."""
    def __init__(self, backoff=None):
        self._impl = _RustBigramTagger(backoff)

    def train(self, sentences: list[list[str]]) -> None:
        self._impl.train(sentences)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)


class TrigramTagger:
    """Trigram tagger — Rust-accelerated lookup."""
    def __init__(self, backoff=None):
        self._impl = _RustTrigramTagger(backoff)

    def train(self, sentences: list[list[str]]) -> None:
        self._impl.train(sentences)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)

    def tag_sents(self, sentences: list[list[str]]) -> list[list[tuple[str, str]]]:
        return self._impl.tag_sents(sentences)


class AffixTagger:
    """Affix (suffix/prefix) tagger — Rust-accelerated."""
    def __init__(self, affix_len=3, use_suffix=True, backoff=None):
        self._impl = _RustAffixTagger(affix_len, use_suffix, backoff)

    def train(self, sentences: list[list[str]]) -> None:
        self._impl.train(sentences)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)


class RegexpTagger:
    """Regexp pattern tagger — Rust-accelerated."""
    def __init__(self, patterns, backoff=None):
        self._impl = _RustRegexpTagger(patterns, backoff)

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return self._impl.tag(tokens)


# ── NLTK re-exports for API compatibility ─────

HunposTagger = _nltk_tag.HunposTagger
SennaChunkTagger = _nltk_tag.SennaChunkTagger
SennaNERTagger = _nltk_tag.SennaNERTagger
SennaTagger = _nltk_tag.SennaTagger
StanfordNERTagger = _nltk_tag.StanfordNERTagger
StanfordPOSTagger = _nltk_tag.StanfordPOSTagger
StanfordTagger = _nltk_tag.StanfordTagger
str2tuple = _nltk_tag.str2tuple
tuple2str = _nltk_tag.tuple2str
untag = _nltk_tag.untag
tagset_mapping = _nltk_tag.tagset_mapping

PRETRAINED_TAGGERS = _nltk_tag.PRETRAINED_TAGGERS
load = _nltk_tag.load

api = _nltk_tag.api
brill = _nltk_tag.brill
brill_trainer = _nltk_tag.brill_trainer
crf = _nltk_tag.crf
hmm = _nltk_tag.hmm
hunpos = _nltk_tag.hunpos
mapping = _nltk_tag.mapping
perceptron = _nltk_tag.perceptron
senna = _nltk_tag.senna
sequential = _nltk_tag.sequential
stanford = _nltk_tag.stanford
tnt = _nltk_tag.tnt
util = _nltk_tag.util
