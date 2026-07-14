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
        expected = nltk.jaccard_distance({"a", "b"}, {"a", "b"})
        result = jaccard_distance(["a", "b"], ["a", "b"])
        assert result == pytest.approx(expected)

    def test_matches_nltk_disjoint(self):
        expected = nltk.jaccard_distance({"a"}, {"b"})
        result = jaccard_distance(["a"], ["b"])
        assert result == pytest.approx(expected)


class TestBinaryDistance:
    def test_matches_nltk_identical(self):
        expected = nltk.binary_distance({"a"}, {"a"})
        result = binary_distance(["a"], ["a"])
        assert result == pytest.approx(expected)


class TestJaroSimilarity:
    def test_identical(self):
        assert jaro_similarity("abc", "abc") == pytest.approx(1.0)

    def test_similar(self):
        # martha vs marhta: expected ~0.944
        result = jaro_similarity("martha", "marhta")
        assert result == pytest.approx(0.944444, rel=1e-4)

    def test_completely_different(self):
        assert jaro_similarity("abc", "xyz") == pytest.approx(0.0)


class TestJaroWinklerSimilarity:
    def test_identical(self):
        assert jaro_winkler_similarity("abc", "abc") == pytest.approx(1.0)

    def test_similar_prefix(self):
        # "martha" vs "marhta" with common prefix "mar" → Winkler boost
        result = jaro_winkler_similarity("martha", "marhta")
        assert result > 0.95


class TestDiceSimilarity:
    def test_identical(self):
        result = dice_similarity("hello world", "hello world")
        assert result == pytest.approx(1.0)

    def test_similar(self):
        result = dice_similarity("hello world", "hello")
        assert 0.0 < result < 1.0
