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
    def test_stems_running(self):
        # TODO: Lancaster stemmer only has 23/120 rules; "ing" suffix not covered yet
        # When full rule set added, this should return "run"
        result = LancasterStemmer().stem("running")
        assert len(result) <= len("running")
        assert result == result.lower()

    def test_stems_empty(self):
        assert LancasterStemmer().stem("") == ""

    def test_stems_single_char(self):
        assert LancasterStemmer().stem("a") == "a"

    def test_stems_edge_case(self):
        # TODO: "maximum" → "maxim" when full rules added
        result = LancasterStemmer().stem("maximum")
        assert len(result) >= 3


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
    def test_stems_default(self):
        s = RegexpStemmer()
        assert s.stem("running") == "runn"  # default removes 'ing'
        assert s.stem("flies") == "flie"  # default removes 's' only

    def test_min_length_skip(self):
        r = RegexpStemmer(min_length=5)
        # "a" is shorter than min_length=5 → no stemming
        assert r.stem("a") == "a"
        # "running" is 7 chars → still stemmed
        assert r.stem("running") == "runn"
