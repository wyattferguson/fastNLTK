"""
fastnltk.chat — Drop-in replacement for nltk.chat.

Rust-accelerated Chat class with compiled regex pattern matching.
"""

import warnings

from nltk.chat import *  # noqa: F403

_rust_available = False
try:
    from fastnltk._rust import Chat as _RustChat
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to NLTK chat"
    )


class Chat:
    """Rust-accelerated pattern-matching chatbot."""
    def __init__(self, pairs):
        if _rust_available:
            self._impl = _RustChat(pairs)
        else:
            from nltk.chat.util import Chat as _NltkChat
            self._impl = _NltkChat(pairs)

    def respond(self, text):
        return self._impl.respond(text)

    def converse(self, text):
        return self._impl.converse(text)
