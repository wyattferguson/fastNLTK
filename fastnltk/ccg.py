"""
fastnltk.ccg — Drop-in replacement for nltk.ccg.

Combinatory Categorial Grammar parsing.
Category types and combinators are Rust-accelerated;
full chart parsing falls back to NLTK.
"""

from nltk.ccg import *  # noqa: F403

_rust_available = False
try:
    from fastnltk._rust import Category
    from fastnltk._rust import from_string as fromstring
    _rust_available = True
except ImportError:
    pass
