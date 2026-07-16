"""
fastnltk.chat — Drop-in replacement for nltk.chat.

Rust-accelerated Chat class with compiled regex pattern matching.
"""

from nltk.chat import *  # noqa: F403

from fastnltk._rust import Chat as _RustChat


class Chat:
    """Rust-accelerated pattern-matching chatbot."""

    def __init__(self, pairs):
        self._impl = _RustChat(pairs)

    def respond(self, text):
        return self._impl.respond(text)

    def converse(self, text):
        return self._impl.converse(text)
