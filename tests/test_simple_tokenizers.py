"""Tests for simple tokenizers (SpaceTokenizer, TabTokenizer, LineTokenizer, CharTokenizer).

These tests verify that fastNLTK's output matches NLTK's output exactly.
"""

import nltk

from fastnltk.tokenize import LineTokenizer, SpaceTokenizer, TabTokenizer


class TestSpaceTokenizer:
    def test_matches_nltk_basic(self):
        text = "a b c"
        result = SpaceTokenizer().tokenize(text)
        assert result == ["a", "b", "c"]

    def test_matches_nltk_multiple_spaces(self):
        # NLTK SpaceTokenizer = str.split(" "), produces empties for gaps
        text = "a  b   c"
        result = SpaceTokenizer().tokenize(text)
        assert result == ["a", "", "b", "", "", "c"]

    def test_matches_nltk_leading_trailing(self):
        text = "  a b  "
        result = SpaceTokenizer().tokenize(text)
        # NLTK SpaceTokenizer = str.split(" ")
        expected = ["", "", "a", "b", "", ""]
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        result = SpaceTokenizer().tokenize(text)
        assert result == [""]

    def test_matches_nltk_single(self):
        text = "hello"
        result = SpaceTokenizer().tokenize(text)
        assert result == ["hello"]

    def test_span_tokenize_basic(self):
        text = "a b c"
        result = SpaceTokenizer().span_tokenize(text)
        assert result == [(0, 1), (2, 3), (4, 5)]


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
        expected = list(nltk.tokenize.LineTokenizer().span_tokenize(text))
        result = LineTokenizer().span_tokenize(text)
        assert result == expected
