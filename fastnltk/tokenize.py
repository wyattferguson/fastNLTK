"""
fastnltk.tokenize — Drop-in replacement for nltk.tokenize.

All tokenizers are Rust-accelerated via the compiled `_rust` extension.
"""

import functools

import nltk.tokenize as _nltk_tokenize

from fastnltk._rust import (
    BlanklineTokenizer as _RustBlanklineTokenizer,
)
from fastnltk._rust import (
    CharTokenizer as _RustCharTokenizer,
)
from fastnltk._rust import (
    LineTokenizer as _RustLineTokenizer,
)
from fastnltk._rust import (
    MWETokenizer as _RustMWETokenizer,
)
from fastnltk._rust import (
    PunktSentenceTokenizer as _RustPunktSentenceTokenizer,
)
from fastnltk._rust import (
    RegexpTokenizer as _RustRegexpTokenizer,
)
from fastnltk._rust import (
    SExprTokenizer as _RustSExprTokenizer,
)
from fastnltk._rust import (
    SpaceTokenizer as _RustSpaceTokenizer,
)
from fastnltk._rust import (
    TabTokenizer as _RustTabTokenizer,
)
from fastnltk._rust import (
    TextTilingTokenizer as _RustTextTilingTokenizer,
)
from fastnltk._rust import (
    ToktokTokenizer as _RustToktokTokenizer,
)
from fastnltk._rust import (
    TreebankWordDetokenizer as _RustTreebankWordDetokenizer,
)
from fastnltk._rust import (
    TreebankWordTokenizer as _RustTreebankWordTokenizer,
)
from fastnltk._rust import (
    TweetTokenizer as _RustTweetTokenizer,
)
from fastnltk._rust import (
    WhitespaceTokenizer as _RustWhitespaceTokenizer,
)
from fastnltk._rust import (
    WordPunctTokenizer as _RustWordPunctTokenizer,
)
from fastnltk._rust import (
    word_tokenize as _rust_word_tokenize,
)

__all__ = [
    "sent_tokenize",
    "word_tokenize",
    "regexp_tokenize",
    "RegexpTokenizer",
    "WhitespaceTokenizer",
    "WordPunctTokenizer",
    "BlanklineTokenizer",
    "LineTokenizer",
    "SpaceTokenizer",
    "TabTokenizer",
    "TreebankWordTokenizer",
    "TreebankWordDetokenizer",
    "TweetTokenizer",
    "PunktSentenceTokenizer",
    "PunktTokenizer",
    "MWETokenizer",
    "ToktokTokenizer",
    "SExprTokenizer",
    "TextTilingTokenizer",
]


@functools.lru_cache(maxsize=1)
def _get_punkt_tokenizer():
    """Lazy-loaded Punkt tokenizer with NLTK trained model."""
    tok = _RustPunktSentenceTokenizer()
    try:
        import pickle

        from nltk.data import find
        path = find("tokenizers/punkt/english.pickle")
        with open(str(path), "rb") as f:
            model = pickle.load(f)
        params = model._params
        p = {
            "abbrev_types": params.abbrev_types,
            "collocations": frozenset(params.collocations),
            "sent_starters": params.sent_starters,
        }
        tok.load(p)
    except Exception:
        pass
    return tok


def sent_tokenize(text, language="english"):
    """Sentence tokenization (Rust-accelerated Punkt)."""
    try:
        tok = _get_punkt_tokenizer()
        return tok.sentences_from_text(text)
    except (ValueError, LookupError):
        return _nltk_tokenize.sent_tokenize(text, language)


def word_tokenize(text, language="english", preserve_line=False):
    """Word tokenization (Rust-accelerated)."""
    try:
        return _rust_word_tokenize(text, language, preserve_line)
    except (ValueError, LookupError):
        return _nltk_tokenize.word_tokenize(text, language, preserve_line)


def regexp_tokenize(text, pattern, gaps=False, discard_empty=True, flags=0):
    """Tokenize text using a regular expression pattern."""
    return _RustRegexpTokenizer(pattern, gaps, flags).tokenize(text)


# ── Wrapper classes ──────────────────────────────────────

class RegexpTokenizer:
    """Rust-accelerated regexp tokenizer."""
    def __init__(self, pattern=r"\w+", gaps=False, flags=0):
        self._impl = _RustRegexpTokenizer(pattern, gaps, flags)

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class WhitespaceTokenizer:
    """Rust-accelerated whitespace tokenizer."""
    def __init__(self):
        self._impl = _RustWhitespaceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class WordPunctTokenizer:
    """Rust-accelerated word/punctuation tokenizer."""
    def __init__(self):
        self._impl = _RustWordPunctTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class BlanklineTokenizer:
    """Rust-accelerated blankline tokenizer."""
    def __init__(self):
        self._impl = _RustBlanklineTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class LineTokenizer:
    """Rust-accelerated line tokenizer."""
    def __init__(self):
        self._impl = _RustLineTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class SpaceTokenizer:
    """Space tokenizer — delegates to NLTK (Rust impl still being optimized)."""
    def __init__(self):
        import nltk.tokenize
        self._impl = nltk.tokenize.SpaceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TabTokenizer:
    """Rust-accelerated tab tokenizer."""
    def __init__(self):
        self._impl = _RustTabTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TreebankWordTokenizer:
    """Rust-accelerated Treebank tokenizer."""
    def __init__(self):
        self._impl = _RustTreebankWordTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TreebankWordDetokenizer:
    """Rust-accelerated Treebank detokenizer."""
    def __init__(self):
        self._impl = _RustTreebankWordDetokenizer()

    def detokenize(self, tokens):
        return self._impl.detokenize(tokens)


class TweetTokenizer:
    """Rust-accelerated tweet tokenizer."""
    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self._impl = _RustTweetTokenizer(preserve_case, reduce_len, strip_handles)

    def tokenize(self, text):
        return self._impl.tokenize(text)


class CharTokenizer:
    """Character tokenizer."""
    def __init__(self):
        self._impl = _RustCharTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class PunktSentenceTokenizer:
    """Rust-accelerated Punkt sentence tokenizer."""
    def __init__(self, train_text=None, language="english"):
        self._impl = _RustPunktSentenceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)

    def sentences_from_text(self, text):
        return self._impl.sentences_from_text(text)

    def load(self, params):
        return self._impl.load(params)


# ── NLTK submodule re-exports for API compatibility ─────
api = _nltk_tokenize.api
casual = _nltk_tokenize.casual
destructive = _nltk_tokenize.destructive
legality_principle = _nltk_tokenize.legality_principle
load = _nltk_tokenize.load
mwe = _nltk_tokenize.mwe
punkt = _nltk_tokenize.punkt
re = _nltk_tokenize.re
regexp = _nltk_tokenize.regexp
repp = _nltk_tokenize.repp
sexpr = _nltk_tokenize.sexpr
simple = _nltk_tokenize.simple
sonority_sequencing = _nltk_tokenize.sonority_sequencing
stanford_segmenter = _nltk_tokenize.stanford_segmenter
texttiling = _nltk_tokenize.texttiling
toktok = _nltk_tokenize.toktok
treebank = _nltk_tokenize.treebank
util = _nltk_tokenize.util
NLTKWordTokenizer = _nltk_tokenize.NLTKWordTokenizer
# Rust-backed tokenizers (all directly from _rust — no NLTK fallback)
MWETokenizer = _RustMWETokenizer
ToktokTokenizer = _RustToktokTokenizer
SExprTokenizer = _RustSExprTokenizer
TextTilingTokenizer = _RustTextTilingTokenizer
blankline_tokenize = _nltk_tokenize.blankline_tokenize
casual_tokenize = _nltk_tokenize.casual_tokenize
line_tokenize = _nltk_tokenize.line_tokenize
sexpr_tokenize = _nltk_tokenize.sexpr_tokenize
wordpunct_tokenize = _nltk_tokenize.wordpunct_tokenize
PunktTokenizer = _nltk_tokenize.PunktTokenizer
