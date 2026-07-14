"""fastnltk.stem — Drop-in replacement for nltk.stem."""

import warnings

import nltk.stem as _nltk_stem

_rust_available = False
try:
    from fastnltk._rust import (
        Cistem as _RustCistem,
    )
    from fastnltk._rust import (
        ISRIStemmer as _RustISRIStemmer,
    )
    from fastnltk._rust import (
        LancasterStemmer as _RustLancasterStemmer,
    )
    from fastnltk._rust import (
        PorterStemmer as _RustPorterStemmer,
    )
    from fastnltk._rust import (
        RegexpStemmer as _RustRegexpStemmer,
    )
    from fastnltk._rust import (
        RSLPStemmer as _RustRSLPStemmer,
    )
    from fastnltk._rust import (
        SnowballStemmer as _RustSnowballStemmer,
    )
    from fastnltk._rust import (
        WordNetLemmatizer as _RustWordNetLemmatizer,
    )
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK stemmers"
    )

__all__ = [
    "SnowballStemmer", "PorterStemmer", "LancasterStemmer", "RegexpStemmer",
    "ISRIStemmer", "Cistem", "RSLPStemmer", "WordNetLemmatizer",
]


class SnowballStemmer:
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
    def __init__(self):
        if _rust_available:
            self._impl = _RustPorterStemmer()
        else:
            self._impl = _nltk_stem.PorterStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class LancasterStemmer:
    def __init__(self):
        if _rust_available:
            self._impl = _RustLancasterStemmer()
        else:
            self._impl = _nltk_stem.LancasterStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class RegexpStemmer:
    def __init__(self, min_length=0):
        if _rust_available:
            self._impl = _RustRegexpStemmer(min_length)
        else:
            self._impl = _nltk_stem.RegexpStemmer(min_length)

    def stem(self, word):
        return self._impl.stem(word)


class ISRIStemmer:
    def __init__(self):
        if _rust_available:
            self._impl = _RustISRIStemmer()
        else:
            self._impl = _nltk_stem.ISRIStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class Cistem:
    def __init__(self):
        if _rust_available:
            self._impl = _RustCistem()
        else:
            self._impl = _nltk_stem.Cistem()

    def stem(self, word):
        return self._impl.stem(word)


class RSLPStemmer:
    def __init__(self):
        if _rust_available:
            self._impl = _RustRSLPStemmer()
        else:
            self._impl = _nltk_stem.RSLPStemmer()

    def stem(self, word):
        return self._impl.stem(word)


class WordNetLemmatizer:
    """WordNet lemmatizer — Rust-accelerated morphy algorithm."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustWordNetLemmatizer()
        else:
            self._impl = _nltk_stem.WordNetLemmatizer()

    def lemmatize(self, word, pos="n"):
        return self._impl.lemmatize(word, pos)
