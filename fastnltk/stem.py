from __future__ import annotations

"""fastnltk.stem — Drop-in replacement for nltk.stem."""

import nltk.stem as _nltk_stem

from fastnltk._rust import (
    ARLSTem as _RustARLSTem,,
    ARLSTem2 as _RustARLSTem2,,
    Cistem as _RustCistem,,
    ISRIStemmer as _RustISRIStemmer,,
    LancasterStemmer as _RustLancasterStemmer,,
    PorterStemmer as _RustPorterStemmer,,
    RegexpStemmer as _RustRegexpStemmer,,
    RSLPStemmer as _RustRSLPStemmer,,
    SnowballStemmer as _RustSnowballStemmer,,
    WordNetLemmatizer as _RustWordNetLemmatizer,)

__all__ = [
    "SnowballStemmer", "PorterStemmer", "LancasterStemmer", "RegexpStemmer",
    "ISRIStemmer", "Cistem", "RSLPStemmer", "WordNetLemmatizer",
    "ARLSTem", "ARLSTem2",
]


class SnowballStemmer:
    def __init__(self, language="english"):
        self._impl = _RustSnowballStemmer(language)
        self._language = language

    def stem(self, word: str) -> str:
        return self._impl.stem(word)

    @property
    def language(self):
        return self._language


class PorterStemmer:
    def __init__(self):
        self._impl = _RustPorterStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class LancasterStemmer:
    def __init__(self):
        self._impl = _RustLancasterStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class RegexpStemmer:
    def __init__(self, min_length=0):
        self._impl = _RustRegexpStemmer(min_length)

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class ISRIStemmer:
    def __init__(self):
        self._impl = _RustISRIStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class Cistem:
    def __init__(self):
        self._impl = _RustCistem()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class RSLPStemmer:
    def __init__(self):
        self._impl = _RustRSLPStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class WordNetLemmatizer:
    """WordNet lemmatizer — Rust-accelerated morphy algorithm."""
    def __init__(self):
        self._impl = _RustWordNetLemmatizer()

    def lemmatize(self, word, pos="n"):
        return self._impl.lemmatize(word, pos)


class ARLSTem:
    """Arabic stemmer — Rust-accelerated."""
    def __init__(self):
        self._impl = _RustARLSTem()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class ARLSTem2:
    """Arabic stemmer v2 — Rust-accelerated."""
    def __init__(self):
        self._impl = _RustARLSTem2()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


# ── NLTK re-exports ─────
StemmerI = _nltk_stem.StemmerI
api = _nltk_stem.api
arlstem = _nltk_stem.arlstem
arlstem2 = _nltk_stem.arlstem2
cistem = _nltk_stem.cistem
isri = _nltk_stem.isri
lancaster = _nltk_stem.lancaster
porter = _nltk_stem.porter
regexp = _nltk_stem.regexp
rslp = _nltk_stem.rslp
snowball = _nltk_stem.snowball
util = _nltk_stem.util
wordnet = _nltk_stem.wordnet
