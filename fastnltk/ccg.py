"""
fastnltk.ccg — Drop-in replacement for nltk.ccg.

Combinatory Categorial Grammar parsing.
Full Rust-accelerated stack: Category types, combinator rules,
lexicon loading, and chart parser.
"""

from fastnltk._rust import from_string as _from_string

fromstring = _from_string

__all__ = [
    "fromstring",
]
