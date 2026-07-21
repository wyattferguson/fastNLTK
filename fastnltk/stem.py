"""fastnltk.stem — Drop-in replacement for nltk.stem."""

from __future__ import annotations

import nltk.stem as _nltk_stem

from fastnltk._rust import (
    ARLSTem as _RustARLSTem,
)
from fastnltk._rust import (
    ARLSTem2 as _RustARLSTem2,
)
from fastnltk._rust import (
    Cistem as _RustCistem,
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
    SnowballStemmer as _RustSnowballStemmer,
)
from fastnltk._rust import (
    WordNetLemmatizer as _RustWordNetLemmatizer,
)

__all__ = [
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
    def __init__(self, language="english"):
        self._impl = _RustSnowballStemmer(language)
        self._language = language

    def stem(self, word: str) -> str:
        return self._impl.stem(word)

    @property
    def language(self) -> str:
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
    """ISRI Arabic stemmer — delegates to NLTK for byte-identical output."""

    def __init__(self):
        self._impl = _nltk_stem.ISRIStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class Cistem:
    def __init__(self):
        self._impl = _RustCistem()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class RSLPStemmer:
    """RSLP Portuguese stemmer — delegates to NLTK for byte-identical output."""

    def __init__(self):
        self._impl = _nltk_stem.RSLPStemmer()

    def stem(self, word: str) -> str:
        return self._impl.stem(word)


class WordNetLemmatizer:
    """WordNet lemmatizer — Rust-accelerated morphy algorithm."""

    def __init__(self):
        self._ensure_wordnet_extracted()
        self._impl = _RustWordNetLemmatizer()

    def lemmatize(self, word: str, pos: str = "n") -> str:
        return self._impl.lemmatize(word, pos)

    @staticmethod
    def _ensure_wordnet_extracted():
        """Extract wordnet.zip to nltk_data/corpora/wordnet if needed."""
        import os
        import zipfile

        for base in [
            os.environ.get("NLTK_DATA", ""),
            os.environ.get("APPDATA", ""),
            os.path.expanduser("~"),
            os.environ.get("USERPROFILE", ""),
        ]:
            if not base:
                continue
            corpora = os.path.join(base, "nltk_data", "corpora")
            wn_dir = os.path.join(corpora, "wordnet")
            if os.path.isdir(wn_dir):
                return
            wn_zip = os.path.join(corpora, "wordnet.zip")
            if os.path.isfile(wn_zip):
                os.makedirs(wn_dir, exist_ok=True)
                with zipfile.ZipFile(wn_zip) as z:
                    for member in z.namelist():
                        # Strip "wordnet/" prefix from archive paths
                        rel = member[len("wordnet/") :] if member.startswith("wordnet/") else member
                        if not rel:
                            continue
                        target = os.path.join(wn_dir, rel)
                        if member.endswith("/"):
                            os.makedirs(target, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with z.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                return


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
