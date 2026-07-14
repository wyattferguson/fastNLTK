"""
fastnltk.ccg — Drop-in replacement for nltk.ccg.

Combinatory Categorial Grammar parsing.
Full Rust-accelerated stack: Category types, combinator rules,
lexicon loading, and chart parser.
"""

from fastnltk._rust import Category, from_string, CCGLexicon, CCGChartParser

# Re-export fromstring alias for NLTK compatibility
fromstring = from_string

# Re-export all NLTK CCG names for API compatibility
from nltk.ccg import (  # noqa: F401
    PrimitiveCategory,
    FunctionalCategory,
    CCGVar,
    Direction,
    AbstractCCGCategory,
)
