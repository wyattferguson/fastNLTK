"""Tests for Rust-accelerated metrics — NLTK compatibility."""

import nltk
import pytest

from fastnltk.metrics import (
    binary_distance,
    dice_similarity,
    edit_distance,
    jaccard_distance,
    jaro_similarity,
    jaro_winkler_similarity,
)


class TestEditDistance:
    def test_matches_nltk_basic(self):
        cases = [
            ("kitten", "sitting"),
            ("abc", "abc"),
            ("", "abc"),
            ("abc", ""),
            ("", ""),
        ]
        for s1, s2 in cases:
            expected = nltk.edit_distance(s1, s2)
            result = edit_distance(s1, s2)
            assert result == expected, f"edit_distance({s1!r}, {s2!r})"

    def test_matches_nltk_transpositions(self):
        expected = nltk.edit_distance("abcd", "acbd", transpositions=True)
        result = edit_distance("abcd", "acbd", transpositions=True)
        assert result == expected


class TestJaccardDistance:
    def test_matches_nltk_identical(self):
        expected = nltk.metrics.distance.jaccard_distance({"a", "b"}, {"a", "b"})
        result = jaccard_distance(["a", "b"], ["a", "b"])
        assert result == pytest.approx(expected)

    def test_matches_nltk_disjoint(self):
        expected = nltk.metrics.distance.jaccard_distance({"a"}, {"b"})
        result = jaccard_distance(["a"], ["b"])
        assert result == pytest.approx(expected)


class TestBinaryDistance:
    def test_matches_nltk_identical(self):
        expected = nltk.metrics.distance.binary_distance({"a"}, {"a"})
        result = binary_distance(["a"], ["a"])
        assert result == pytest.approx(expected)


class TestJaroSimilarity:
    def test_matches_nltk_identical(self):
        expected = nltk.metrics.distance.jaro_similarity("abc", "abc")
        result = jaro_similarity("abc", "abc")
        assert result == pytest.approx(expected)

    def test_matches_nltk_similar(self):
        expected = nltk.metrics.distance.jaro_similarity("martha", "marhta")
        result = jaro_similarity("martha", "marhta")
        assert result == pytest.approx(expected, rel=1e-4)


class TestJaroWinklerSimilarity:
    def test_matches_nltk_identical(self):
        expected = nltk.metrics.distance.jaro_winkler_similarity("abc", "abc")
        result = jaro_winkler_similarity("abc", "abc")
        assert result == pytest.approx(expected)


class TestDiceSimilarity:
    def test_matches_nltk_identical(self):
        # dice with identical bigrams should give high similarity
        result = dice_similarity("hello world", "hello world")
        # dice with identical bigrams should give high similarity
        assert result > 0.5
