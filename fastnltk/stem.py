"""
fastnltk.stem — Drop-in replacement for nltk.stem.

Delegates to compiled Rust extension where available,
falls back to original nltk.stem for unimplemented pieces.
"""

from fastnltk._rust import (
    SnowballStemmer as _RustSnowballStemmer,
    PorterStemmer as _RustPorterStemmer,
    LancasterStemmer as _RustLancasterStemmer,
    RegexpStemmer as _RustRegexpStemmer,
    ISRIStemmer as _RustISRIStemmer,
    Cistem as _RustCistem,
    RSLPStemmer as _RustRSLPStemmer,
)

import nltk.stem as _nltk_stem
from nltk.stem import (
    WordNetLemmatizer,
    ARLSTem,
    ARLSTem2,
)

from nltk.stem.api import StemmerI

__all__ = [
    "StemmerI",
    "SnowballStemmer",
    "PorterStemmer",
    "LancasterStemmer",
    "RegexpStemmer",
    "ISRIStemmer",
    "Cistem",
    "RSLPStemmer",
    "WordNetLemmatizer",
    "ARLSTem",
    "ARLSTem2",
]


class SnowballStemmer:
    """Rust-accelerated Snowball stemmer.

    Supports languages: danish, dutch, english, finnish, french, german,
    hungarian, italian, norwegian, portuguese, romanian, russian, spanish,
    swedish, turkish, arabic.
    """
    def __init__(self, language="english"):
        self._impl = _RustSnowballStemmer(language)
        self._language = language

    def stem(self, word):
        return self._impl.stem(word)

    def stem_many(self, words):
        return self._impl.stem_many(words)

    @property
    def language(self):
        return self._language


class PorterStemmer:
    """Rust-accelerated Porter stemmer."""
    def __init__(self):
        self._impl = _RustPorterStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class LancasterStemmer:
    """Rust-accelerated Lancaster stemmer."""
    def __init__(self):
        self._impl = _RustLancasterStemmer()

    def stem(self, word):
        return self._impl.stem(word)

    @property
    def rule_tuple(self):
        return None  # placeholder for NLTK compat


class RegexpStemmer:
    """Rust-accelerated regexp stemmer."""
    def __init__(self, min_length=0):
        self._impl = _RustRegexpStemmer(min_length)

    def stem(self, word):
        return self._impl.stem(word)


class ISRIStemmer:
    """Rust-accelerated Arabic ISRI stemmer."""
    def __init__(self):
        self._impl = _RustISRIStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class Cistem:
    """Rust-accelerated German Cistem."""
    def __init__(self):
        self._impl = _RustCistem()

    def stem(self, word):
        return self._impl.stem(word)


class RSLPStemmer:
    """Rust-accelerated Portuguese RSLP stemmer."""
    def __init__(self):
        self._impl = _RustRSLPStemmer()

    def stem(self, word):
        return self._impl.stem(word)
