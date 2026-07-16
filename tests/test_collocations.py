"""Tests for Rust-accelerated collocations — NLTK compatibility."""

from nltk.collocations import BigramAssocMeasures, QuadgramAssocMeasures, TrigramAssocMeasures

from fastnltk.collocations import (
    BigramCollocationFinder,
    QuadgramCollocationFinder,
    TrigramCollocationFinder,
)


class TestBigramCollocationFinder:
    def test_from_words(self):
        words = "the cat sat on the mat the cat sat".split()
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(BigramAssocMeasures.raw_freq)
        assert len(scored) > 0
        assert isinstance(scored[0], tuple)
        assert len(scored[0]) == 2

    def test_nbest(self):
        words = "the cat sat on the mat the cat sat".split()
        finder = BigramCollocationFinder.from_words(words)
        top = finder.nbest(BigramAssocMeasures.raw_freq, 3)
        assert len(top) <= 3

    def test_score_ngrams_pmi(self):
        words = "the cat sat on the mat the cat sat the cat".split()
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(BigramAssocMeasures.pmi)
        assert len(scored) > 0

    def test_score_ngrams_chi_sq(self):
        words = "the cat sat on the mat the cat sat".split()
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(BigramAssocMeasures.chi_sq)
        assert len(scored) > 0

    def test_score_ngrams_likelihood_ratio(self):
        words = "the cat sat on the mat the cat sat".split()
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
        assert len(scored) > 0


class TestTrigramCollocationFinder:
    def test_from_words(self):
        words = "the cat sat on the mat the cat sat on".split()
        finder = TrigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(TrigramAssocMeasures.raw_freq)
        assert len(scored) > 0

    def test_nbest(self):
        words = "the cat sat on the mat the cat sat on".split()
        finder = TrigramCollocationFinder.from_words(words)
        top = finder.nbest(TrigramAssocMeasures.raw_freq, 3)
        assert len(top) <= 3


class TestQuadgramCollocationFinder:
    def test_from_words(self):
        words = "the cat sat on the mat the cat sat on the mat".split()
        finder = QuadgramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(QuadgramAssocMeasures.raw_freq)
        assert len(scored) > 0

    def test_nbest(self):
        words = "the cat sat on the mat the cat sat on the mat".split()
        finder = QuadgramCollocationFinder.from_words(words)
        top = finder.nbest(QuadgramAssocMeasures.raw_freq, 3)
        assert len(top) <= 3
