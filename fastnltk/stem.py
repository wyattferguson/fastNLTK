"""
fastnltk.stem — Drop-in replacement for nltk.stem.

Delegates to compiled Rust extension where available,
falls back to original nltk.stem for unimplemented pieces.
"""

_rust_available = False
try:
    from fastnltk._rust import (
        SnowballStemmer as _RustSnowballStemmer,
        PorterStemmer as _RustPorterStemmer,
    )
    _rust_available = True
except ImportError:
    pass

import nltk.stem as _nltk_stem
from nltk.stem import (
    LancasterStemmer,
    RegexpStemmer,
    WordNetLemmatizer,
    ISRIStemmer,
    RSLPStemmer,
    Cistem,
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
    "WordNetLemmatizer",
    "ISRIStemmer",
    "RSLPStemmer",
    "Cistem",
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
        if _rust_available:
            self._impl = _RustSnowballStemmer(language)
        else:
            self._impl = _nltk_stem.SnowballStemmer(language)
        self._language = language

    def stem(self, word):
        return self._impl.stem(word)

    @property
    def language(self):
        return self._language


class PorterStemmer:
    """Rust-accelerated Porter stemmer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustPorterStemmer()
        else:
            self._impl = _nltk_stem.PorterStemmer()

    def stem(self, word):
        return self._impl.stem(word)
