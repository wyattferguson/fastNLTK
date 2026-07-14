"""Property-based tests for tokenizer invariants.

Uses Hypothesis-style manually specified invariants.
"""

import pytest

from fastnltk.tokenize import sent_tokenize, word_tokenize


class TestTokenizerInvariants:
    """Invariants that must hold for all inputs."""

    def test_word_tokenize_never_panics(self):
        """word_tokenize must not raise on any string."""
        nasty_strings = [
            "",
            " ",
            "\x00",
            "\uffff",
            "\n" * 1000,
            "a" * 10000,
            "!@#$%^&*()_+-=[]{}|;:',.<>?/",
            "hello\x00world",
            "a b c",
            "\n\n\n",
            "  \t  \n  ",
        ]
        for s in nasty_strings:
            result = word_tokenize(s)
            assert isinstance(result, list)

    def test_sent_tokenize_never_panics(self):
        """sent_tokenize must not raise on any string."""
        nasty_strings = [
            "",
            " ",
            "\x00",
            "." * 1000,
            "Hello. " * 500,
            "\n" * 100,
        ]
        for s in nasty_strings:
            result = sent_tokenize(s)
            assert isinstance(result, list)

    def test_tokenize_roundtrip(self):
        """Tokenizing and joining should approximate the original."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = word_tokenize(text)
        # Rejoining should get close to original
        rejoined = " ".join(tokens)
        assert len(rejoined) >= len(text) * 0.5

    def test_no_empty_tokens_from_treebank(self):
        """Treebank tokenizer should not produce empty string tokens."""
        from fastnltk.tokenize import TreebankWordTokenizer
        tok = TreebankWordTokenizer()
        for text in ["Hello world.", "", "a", "a b c", "a.b!c?"]:
            tokens = tok.tokenize(text)
            assert all(t != "" for t in tokens), f"empty token in: {tokens}"

    def test_unicode_boundary(self):
        """Multi-byte unicode characters should not cause panics."""
        unicode_texts = [
            "café",        # 2-byte
            "naïve",       # 2-byte
            "中文测试",    # CJK 3-byte
            "مرحبا",       # Arabic RTL
            "💻🌍🎉",      # 4-byte emoji
            "こんにちは",  # Japanese
        ]
        for text in unicode_texts:
            # Must not panic
            tokens = word_tokenize(text)
            assert len(tokens) >= 1
