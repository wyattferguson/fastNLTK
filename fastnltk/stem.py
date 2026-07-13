"""fastnltk.stem — Drop-in replacement for nltk.stem."""

_rust_available = False
try:
    from fastnltk._rust import (
        SnowballStemmer as _RustSnowballStemmer,
        PorterStemmer as _RustPorterStemmer,
        LancasterStemmer as _RustLancasterStemmer,
        ISRIStemmer as _RustISRIStemmer,
        Cistem as _RustCistem,
        RSLPStemmer as _RustRSLPStemmer,
        RegexpStemmer as _RustRegexpStemmer,
    )
    _rust_available = True
except ImportError:
    pass

import nltk.stem as _nltk_stem

__all__ = [
    "SnowballStemmer", "PorterStemmer", "LancasterStemmer", "RegexpStemmer",
    "ISRIStemmer", "Cistem", "RSLPStemmer",
]

def _make_stemmer(rust_cls, nltk_cls, *init_args):
    if _rust_available:
        return rust_cls(*init_args)
    return nltk_cls(*init_args)

class SnowballStemmer:
    def __init__(self, language="english"):
        if _rust_available: self._impl = _RustSnowballStemmer(language)
        else: self._impl = _nltk_stem.SnowballStemmer(language)
        self._language = language
    def stem(self, word): return self._impl.stem(word)
    @property
    def language(self): return self._language

class PorterStemmer:
    def __init__(self):
        if _rust_available: self._impl = _RustPorterStemmer()
        else: self._impl = _nltk_stem.PorterStemmer()
    def stem(self, word): return self._impl.stem(word)

class LancasterStemmer:
    def __init__(self):
        if _rust_available: self._impl = _RustLancasterStemmer()
        else: self._impl = _nltk_stem.LancasterStemmer()
    def stem(self, word): return self._impl.stem(word)

class RegexpStemmer:
    def __init__(self, min_length=0):
        if _rust_available: self._impl = _RustRegexpStemmer(min_length)
        else: self._impl = _nltk_stem.RegexpStemmer(min_length)
    def stem(self, word): return self._impl.stem(word)

class ISRIStemmer:
    def __init__(self):
        if _rust_available: self._impl = _RustISRIStemmer()
        else: self._impl = _nltk_stem.ISRIStemmer()
    def stem(self, word): return self._impl.stem(word)

class Cistem:
    def __init__(self):
        if _rust_available: self._impl = _RustCistem()
        else: self._impl = _nltk_stem.Cistem()
    def stem(self, word): return self._impl.stem(word)

class RSLPStemmer:
    def __init__(self):
        if _rust_available: self._impl = _RustRSLPStemmer()
        else: self._impl = _nltk_stem.RSLPStemmer()
    def stem(self, word): return self._impl.stem(word)
