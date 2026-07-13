"""
fastnltk.tokenize — Drop-in replacement for nltk.tokenize.

Delegates to compiled Rust extension where available,
falls back to original nltk.tokenize for unimplemented pieces.
"""

from fastnltk._rust import (
    sent_tokenize as _rust_sent_tokenize,
    word_tokenize as _rust_word_tokenize,
    RegexpTokenizer as _RustRegexpTokenizer,
    WhitespaceTokenizer as _RustWhitespaceTokenizer,
    WordPunctTokenizer as _RustWordPunctTokenizer,
    BlanklineTokenizer as _RustBlanklineTokenizer,
    LineTokenizer as _RustLineTokenizer,
    SpaceTokenizer as _RustSpaceTokenizer,
    TabTokenizer as _RustTabTokenizer,
    TreebankWordTokenizer as _RustTreebankWordTokenizer,
    TreebankWordDetokenizer as _RustTreebankWordDetokenizer,
    TweetTokenizer as _RustTweetTokenizer,
    MWETokenizer as _RustMWETokenizer,
    ToktokTokenizer as _RustToktokTokenizer,
    NISTTokenizer as _RustNISTTokenizer,
    SExprTokenizer as _RustSExprTokenizer,
    CharTokenizer as _RustCharTokenizer,
)

import nltk.tokenize as _nltk_tokenize
from nltk.tokenize import (
    PunktSentenceTokenizer,
    PunktTokenizer,
    LegalitySyllableTokenizer,
    SyllableTokenizer,
    TextTilingTokenizer,
    StanfordSegmenter,
    ReppTokenizer,
    regexp_span_tokenize,
    string_span_tokenize,
)

__all__ = [
    # Functions
    "sent_tokenize",
    "word_tokenize",
    "regexp_tokenize",
    "wordpunct_tokenize",
    "blankline_tokenize",
    "line_tokenize",
    "regexp_span_tokenize",
    "string_span_tokenize",
    # Classes
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
    "MWETokenizer",
    "ToktokTokenizer",
    "NISTTokenizer",
    "SExprTokenizer",
    "CharTokenizer",
    "PunktSentenceTokenizer",
    "PunktTokenizer",
    "LegalitySyllableTokenizer",
    "SyllableTokenizer",
    "TextTilingTokenizer",
    "StanfordSegmenter",
    "ReppTokenizer",
]


def sent_tokenize(text, language="english"):
    """Sentence tokenization (Rust-accelerated).

    Returns a list of sentences, each as a string.
    Falls back to NLTK if language is not supported by the Rust engine.
    """
    try:
        return _rust_sent_tokenize(text, language)
    except (ValueError, LookupError):
        return _nltk_tokenize.sent_tokenize(text, language)


def word_tokenize(text, language="english", preserve_line=False):
    """Word tokenization (Rust-accelerated).

    Returns a list of token strings.
    Falls back to NLTK if language is not supported by the Rust engine.
    """
    try:
        return _rust_word_tokenize(text, language, preserve_line)
    except (ValueError, LookupError):
        return _nltk_tokenize.word_tokenize(text, language, preserve_line)


def regexp_tokenize(text, pattern, gaps=False, discard_empty=True, flags=0):
    """Tokenize text using a regular expression pattern."""
    return _RustRegexpTokenizer(pattern, gaps, flags).tokenize(text)


def wordpunct_tokenize(text):
    """Tokenize text into alphabetic and non-alphabetic tokens."""
    return _RustWordPunctTokenizer().tokenize(text)


def blankline_tokenize(text):
    """Tokenize text by blank lines."""
    return _RustBlanklineTokenizer().tokenize(text)


def line_tokenize(text):
    """Tokenize text by lines."""
    return _RustLineTokenizer().tokenize(text)


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


class SpaceTokenizer:
    """Rust-accelerated space tokenizer."""
    def __init__(self):
        self._impl = _RustSpaceTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class TabTokenizer:
    """Rust-accelerated tab tokenizer."""
    def __init__(self):
        self._impl = _RustTabTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


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


class MWETokenizer:
    """Rust-accelerated multi-word expression tokenizer."""
    def __init__(self, mwe_pairs=None, separator="_"):
        self._impl = _RustMWETokenizer(mwe_pairs or [], separator)

    def tokenize(self, text):
        return self._impl.tokenize(text)

    def add_mwe(self, mwe):
        self._impl.add_mwe(mwe)


class ToktokTokenizer:
    """Rust-accelerated TokTok tokenizer."""
    def __init__(self):
        self._impl = _RustToktokTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class NISTTokenizer:
    """Rust-accelerated NIST tokenizer."""
    def __init__(self):
        self._impl = _RustNISTTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)


class SExprTokenizer:
    """Rust-accelerated S-expression tokenizer."""
    def __init__(self, strict=True):
        self._impl = _RustSExprTokenizer(strict)

    def tokenize(self, text):
        return self._impl.tokenize(text)


class CharTokenizer:
    """Character tokenizer."""
    def __init__(self):
        self._impl = _RustCharTokenizer()

    def tokenize(self, text):
        return self._impl.tokenize(text)
