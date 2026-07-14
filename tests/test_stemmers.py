"""Tests for Rust-accelerated stemmers — NLTK compatibility."""


import nltk

from fastnltk.stem import (
    LancasterStemmer,
    PorterStemmer,
    RegexpStemmer,
    SnowballStemmer,
)


class TestPorterStemmer:
    def test_matches_nltk_basic(self):
        words = ["running", "flies", "happily", "dogs", "cats", "fishing"]
        for w in words:
            assert PorterStemmer().stem(w) == nltk.stem.PorterStemmer().stem(w)

    def test_matches_nltk_edge_cases(self):
        words = ["", "a", "the", "nationality"]
        for w in words:
            assert PorterStemmer().stem(w) == nltk.stem.PorterStemmer().stem(w)


class TestLancasterStemmer:
    def test_matches_nltk_basic(self):
        words = ["running", "flies", "happily", "dogs", "cats"]
        for w in words:
            assert LancasterStemmer().stem(w) == nltk.stem.LancasterStemmer().stem(w)

    def test_matches_nltk_edge_cases(self):
        for w in ["", "a", "maximum"]:
            assert LancasterStemmer().stem(w) == nltk.stem.LancasterStemmer().stem(w)


class TestSnowballStemmer:
    def test_matches_nltk_english(self):
        words = ["running", "flies", "happily", "generalizations"]
        for w in words:
            assert SnowballStemmer("english").stem(w) == nltk.stem.SnowballStemmer("english").stem(w)

    def test_matches_nltk_french(self):
        words = ["courir", "manger", "parlant"]
        for w in words:
            assert SnowballStemmer("french").stem(w) == nltk.stem.SnowballStemmer("french").stem(w)

    def test_matches_nltk_spanish(self):
        words = ["corriendo", "comiendo", "hablando"]
        for w in words:
            assert SnowballStemmer("spanish").stem(w) == nltk.stem.SnowballStemmer("spanish").stem(w)


class TestRegexpStemmer:
    def test_matches_nltk_default(self):
        words = ["running", "flies"]
        for w in words:
            assert RegexpStemmer().stem(w) == nltk.stem.RegexpStemmer("ing$|ed$|s$").stem(w)

    def test_matches_nltk_min_length(self):
        r = RegexpStemmer(min_length=5)
        n = nltk.stem.RegexpStemmer("ing$|ed$|s$", min_length=5)
        for w in ["running", "a"]:
            assert r.stem(w) == n.stem(w)
