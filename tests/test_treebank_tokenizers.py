"""Tests for Rust-accelerated tokenizers — NLTK compatibility."""

import nltk

from fastnltk.tokenize import (
    BlanklineTokenizer,
    RegexpTokenizer,
    TreebankWordDetokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    WordPunctTokenizer,
)


class TestTreebankWordTokenizer:
    def test_matches_nltk_basic(self):
        text = "Hello world."
        expected = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        result = TreebankWordTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_contractions(self):
        text = "can't won't I'll he'd she's"
        expected = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        result = TreebankWordTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_punctuation(self):
        text = "Hello (world), isn't it?"
        expected = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        result = TreebankWordTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        expected = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        result = TreebankWordTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_single_word(self):
        text = "hello"
        expected = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
        result = TreebankWordTokenizer().tokenize(text)
        assert result == expected

    def test_span_tokenize_basic(self):
        text = "Hello world."
        expected = list(nltk.tokenize.TreebankWordTokenizer().span_tokenize(text))
        result = TreebankWordTokenizer().span_tokenize(text)
        assert result == expected


class TestTreebankWordDetokenizer:
    def test_matches_nltk_basic(self):
        tokens = ["Hello", ",", "world", "."]
        expected = nltk.tokenize.TreebankWordDetokenizer().detokenize(tokens)
        result = TreebankWordDetokenizer().detokenize(tokens)
        assert result == expected

    def test_matches_nltk_contraction(self):
        tokens = ["ca", "n't"]
        expected = nltk.tokenize.TreebankWordDetokenizer().detokenize(tokens)
        result = TreebankWordDetokenizer().detokenize(tokens)
        assert result == expected


class TestTweetTokenizer:
    def test_matches_nltk_basic(self):
        text = "Hello world #hashtag"
        expected = nltk.tokenize.TweetTokenizer().tokenize(text)
        result = TweetTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_preserve_case(self):
        text = "@user Hello"
        expected = nltk.tokenize.TweetTokenizer(preserve_case=True).tokenize(text)
        result = TweetTokenizer(preserve_case=True).tokenize(text)
        assert result == expected

    def test_matches_nltk_reduce_len(self):
        text = "soooo coool!"
        expected = nltk.tokenize.TweetTokenizer(reduce_len=True).tokenize(text)
        result = TweetTokenizer(reduce_len=True).tokenize(text)
        assert result == expected

    def test_matches_nltk_strip_handles(self):
        text = "@user hello"
        expected = nltk.tokenize.TweetTokenizer(strip_handles=True).tokenize(text)
        result = TweetTokenizer(strip_handles=True).tokenize(text)
        assert result == expected


class TestRegexpTokenizer:
    def test_matches_nltk_default(self):
        text = "Hello world."
        pattern = r"\w+|[^\w\s]+"
        expected = nltk.tokenize.RegexpTokenizer(pattern).tokenize(text)
        result = RegexpTokenizer(pattern).tokenize(text)
        assert result == expected

    def test_matches_nltk_gaps(self):
        text = "Hello world."
        expected = nltk.tokenize.RegexpTokenizer(r"\s+", gaps=True).tokenize(text)
        result = RegexpTokenizer(r"\s+", gaps=True).tokenize(text)
        assert result == expected

    def test_matches_nltk_custom_pattern(self):
        text = "123 abc 456"
        expected = nltk.tokenize.RegexpTokenizer(r"\d+").tokenize(text)
        result = RegexpTokenizer(r"\d+").tokenize(text)
        assert result == expected


class TestWordPunctTokenizer:
    def test_matches_nltk_basic(self):
        text = "Hello, world!"
        expected = nltk.tokenize.WordPunctTokenizer().tokenize(text)
        result = WordPunctTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_empty(self):
        text = ""
        expected = nltk.tokenize.WordPunctTokenizer().tokenize(text)
        result = WordPunctTokenizer().tokenize(text)
        assert result == expected


class TestBlanklineTokenizer:
    def test_matches_nltk_basic(self):
        text = "Hello\n\nworld"
        expected = nltk.tokenize.BlanklineTokenizer().tokenize(text)
        result = BlanklineTokenizer().tokenize(text)
        assert result == expected

    def test_matches_nltk_single_paragraph(self):
        text = "Just one paragraph"
        expected = nltk.tokenize.BlanklineTokenizer().tokenize(text)
        result = BlanklineTokenizer().tokenize(text)
        assert result == expected
