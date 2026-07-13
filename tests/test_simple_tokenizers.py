"""Tests for simple tokenizers (SpaceTokenizer, TabTokenizer, LineTokenizer, CharTokenizer).

These tests verify that fastNLTK's output matches NLTK's output exactly.
"""

import pytest

import nltk
from fastnltk.tokenize import SpaceTokenizer, TabTokenizer, LineTokenizer


class TestSpaceTokenizer:
    def test_matches_nltk_basic(self):
        text = "a b c"
        expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
        result = SpaceTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_multiple_spaces(self):
        text = "a  b   c"
        expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
        result = SpaceTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_leading_trailing(self):
        text = "  a b  "
        expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
        result = SpaceTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
        result = SpaceTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_single(self):
        text = "hello"
        expected = nltk.tokenize.SpaceTokenizer().tokenize(text)
        result = SpaceTokenizer().tokenize(text)
        assert result == expected

    def test_span_tokenize_basic(self):
        text = "a b c"
        expected = nltk.tokenize.SpaceTokenizer().span_tokenize(text)
        result = SpaceTokenizer().span_tokenize(text)
        assert result == expected


class TestTabTokenizer:
    def test_matches_nltk_basic(self):
        text = "a\tb\tc"
        expected = nltk.tokenize.TabTokenizer().tokenize(text)
        result = TabTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        expected = nltk.tokenize.TabTokenizer().tokenize(text)
        result = TabTokenizer().tokenize(text)
        assert result == expected


class TestLineTokenizer:
    def test_matches_nltk_basic(self):
        text = "a\nb\nc"
        expected = nltk.tokenize.LineTokenizer().tokenize(text)
        result = LineTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        expected = nltk.tokenize.LineTokenizer().tokenize(text)
        result = LineTokenizer().tokenize(text)
        assert result == expected

    def test_span_tokenize(self):
        text = "ab\ncd\nef"
        expected = nltk.tokenize.LineTokenizer().span_tokenize(text)
        result = LineTokenizer().span_tokenize(text)
        assert result == expected
