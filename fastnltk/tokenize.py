"""
fastnltk.tokenize — Drop-in replacement for nltk.tokenize.

Delegates to compiled Rust extension where available,
falls back to original nltk.tokenize for unimplemented pieces.
"""

import functools
import warnings

import nltk.tokenize as _nltk_tokenize

_rust_available = False
try:
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
        PunktSentenceTokenizer as _RustPunktSentenceTokenizer,
    )
    from fastnltk._rust import (
        RegexpTokenizer as _RustRegexpTokenizer,
    )
    from fastnltk._rust import (
        SpaceTokenizer as _RustSpaceTokenizer,
    )
    from fastnltk._rust import (
        TabTokenizer as _RustTabTokenizer,
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
    from fastnltk._rust import (
        SExprTokenizer as _RustSExprTokenizer,
    )
    from fastnltk._rust import (
        ToktokTokenizer as _RustToktokTokenizer,
    )
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK tokenizers"
    )

# Import what's available in NLTK — some symbols vary by version
try:
    from nltk.tokenize import MWETokenizer
except ImportError:
    MWETokenizer = None
try:
    from nltk.tokenize import ToktokTokenizer
except ImportError:
    ToktokTokenizer = None
try:
    from nltk.tokenize import SExprTokenizer
except ImportError:
    SExprTokenizer = None
try:
    from nltk.tokenize import PunktSentenceTokenizer, PunktTokenizer
except ImportError:
    PunktSentenceTokenizer = PunktTokenizer = None
try:
    from nltk.tokenize import LegalitySyllableTokenizer
except ImportError:
    LegalitySyllableTokenizer = None
try:
    from nltk.tokenize import SyllableTokenizer
except ImportError:
    SyllableTokenizer = None
try:
    from nltk.tokenize import TextTilingTokenizer
except ImportError:
    TextTilingTokenizer = None
try:
    from nltk.tokenize import StanfordSegmenter
except ImportError:
    StanfordSegmenter = None
try:
    from nltk.tokenize import ReppTokenizer
except ImportError:
    ReppTokenizer = None
try:
    from nltk.tokenize import regexp_span_tokenize, string_span_tokenize
except ImportError:
    regexp_span_tokenize = string_span_tokenize = None

__all__ = [
    "sent_tokenize",
    "word_tokenize",
    "regexp_tokenize",
    "regexp_span_tokenize",
    "string_span_tokenize",
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
]
if MWETokenizer is not None:
    __all__.append("MWETokenizer")
if ToktokTokenizer is not None:
    __all__.append("ToktokTokenizer")
if SExprTokenizer is not None:
    __all__.append("SExprTokenizer")
if PunktSentenceTokenizer is not None:
    __all__.extend(["PunktSentenceTokenizer", "PunktTokenizer"])
if LegalitySyllableTokenizer is not None:
    __all__.append("LegalitySyllableTokenizer")
if SyllableTokenizer is not None:
    __all__.append("SyllableTokenizer")
if TextTilingTokenizer is not None:
    __all__.append("TextTilingTokenizer")
if StanfordSegmenter is not None:
    __all__.append("StanfordSegmenter")
if ReppTokenizer is not None:
    __all__.append("ReppTokenizer")


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
    if _rust_available:
        try:
            tok = _get_punkt_tokenizer()
            return tok.sentences_from_text(text)
        except (ValueError, LookupError):
            pass
    return _nltk_tokenize.sent_tokenize(text, language)


def word_tokenize(text, language="english", preserve_line=False):
    """Word tokenization (Rust-accelerated)."""
    if _rust_available:
        try:
            return _rust_word_tokenize(text, language, preserve_line)
        except (ValueError, LookupError):
            pass
    return _nltk_tokenize.word_tokenize(text, language, preserve_line)


def regexp_tokenize(text, pattern, gaps=False, discard_empty=True, flags=0):
    """Tokenize text using a regular expression pattern."""
    if _rust_available:
        return _RustRegexpTokenizer(pattern, gaps, flags).tokenize(text)
    return _nltk_tokenize.regexp_tokenize(text, pattern, gaps, discard_empty, flags)


# ── Wrapper classes ──────────────────────────────────────

class RegexpTokenizer:
    """Rust-accelerated regexp tokenizer."""
    def __init__(self, pattern=r"\w+", gaps=False, flags=0):
        if _rust_available:
            self._impl = _RustRegexpTokenizer(pattern, gaps, flags)
        else:
            self._impl = _nltk_tokenize.RegexpTokenizer(pattern, gaps)

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class WhitespaceTokenizer:
    """Rust-accelerated whitespace tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustWhitespaceTokenizer()
        else:
            self._impl = _nltk_tokenize.WhitespaceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class WordPunctTokenizer:
    """Rust-accelerated word/punctuation tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustWordPunctTokenizer()
        else:
            self._impl = _nltk_tokenize.WordPunctTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class BlanklineTokenizer:
    """Rust-accelerated blankline tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustBlanklineTokenizer()
        else:
            self._impl = _nltk_tokenize.BlanklineTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class LineTokenizer:
    """Rust-accelerated line tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustLineTokenizer()
        else:
            self._impl = _nltk_tokenize.LineTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class SpaceTokenizer:
    """Rust-accelerated space tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustSpaceTokenizer()
        else:
            self._impl = _nltk_tokenize.SpaceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TabTokenizer:
    """Rust-accelerated tab tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustTabTokenizer()
        else:
            self._impl = _nltk_tokenize.TabTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TreebankWordTokenizer:
    """Rust-accelerated Treebank tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustTreebankWordTokenizer()
        else:
            self._impl = _nltk_tokenize.TreebankWordTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)


class TreebankWordDetokenizer:
    """Rust-accelerated Treebank detokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustTreebankWordDetokenizer()
        else:
            self._impl = _nltk_tokenize.TreebankWordDetokenizer()

    def detokenize(self, tokens):
        return self._impl.detokenize(tokens)


class TweetTokenizer:
    """Rust-accelerated tweet tokenizer."""
    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        if _rust_available:
            self._impl = _RustTweetTokenizer(preserve_case, reduce_len, strip_handles)
        else:
            self._impl = _nltk_tokenize.TweetTokenizer(preserve_case, reduce_len, strip_handles)

    def tokenize(self, text):
        return self._impl.tokenize(text)


class CharTokenizer:
    """Character tokenizer."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustCharTokenizer()
        else:
            self._impl = _nltk_tokenize.CharTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class PunktSentenceTokenizer:
    """Rust-accelerated Punkt sentence tokenizer."""
    def __init__(self, train_text=None, language="english"):
        if _rust_available:
            self._impl = _RustPunktSentenceTokenizer()
        else:
            self._impl = _nltk_tokenize.PunktSentenceTokenizer(train_text, language)

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def span_tokenize(self, text):
        return self._impl.span_tokenize(text)

    def sentences_from_text(self, text):
        return self._impl.sentences_from_text(text)

    def load(self, params):
        return self._impl.load(params)


# ── NLTK submodule re-exports for API compatibility ─────
# Submodule pass-through: from fastnltk.tokenize import punkt, treebank, etc.
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
blankline_tokenize = _nltk_tokenize.blankline_tokenize
casual_tokenize = _nltk_tokenize.casual_tokenize
line_tokenize = _nltk_tokenize.line_tokenize
sexpr_tokenize = _nltk_tokenize.sexpr_tokenize
wordpunct_tokenize = _nltk_tokenize.wordpunct_tokenize

# Also export PunktTokenizer (alias)
if PunktSentenceTokenizer is not None:
    PunktTokenizer = _nltk_tokenize.PunktTokenizer
