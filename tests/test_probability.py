"""Tests for Rust-accelerated probability — NLTK compatibility."""

import nltk
import pytest

from fastnltk.probability import ConditionalFreqDist, FreqDist


class TestFreqDist:
    def test_matches_nltk_counts(self):
        samples = ["a", "b", "a", "c", "b", "a"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        for sample in set(samples):
            assert fd[sample] == nltk_fd[sample]
            assert fd.freq(sample) == pytest.approx(nltk_fd.freq(sample))

    def test_matches_nltk_N(self):
        samples = ["a", "b", "a"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        assert fd.N() == nltk_fd.N()

    def test_matches_nltk_B(self):
        samples = ["a", "b", "a", "c"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        assert fd.B() == nltk_fd.B()

    def test_matches_nltk_max(self):
        samples = ["a", "b", "a", "c", "b", "a"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        assert fd.max() == nltk_fd.max()

    def test_matches_nltk_most_common(self):
        samples = ["a", "b", "a", "c", "b", "a"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        assert fd.most_common(2) == nltk_fd.most_common(2)

    def test_matches_nltk_hapaxes(self):
        samples = ["a", "b", "a", "c"]
        fd = FreqDist(samples)
        nltk_fd = nltk.probability.FreqDist(samples)
        assert set(fd.hapaxes()) == set(nltk_fd.hapaxes())

    def test_matches_nltk_empty(self):
        fd = FreqDist()
        nltk_fd = nltk.probability.FreqDist()
        assert fd.N() == nltk_fd.N()
        assert fd.B() == nltk_fd.B()

    def test_len(self):
        fd = FreqDist(["a", "b", "a"])
        assert len(fd) == 2


class TestConditionalFreqDist:
    def test_matches_nltk_basic(self):
        cfd = ConditionalFreqDist()
        nltk_cfd = nltk.probability.ConditionalFreqDist()
        cfd.inc("A", "x")
        cfd.inc("A", "x")
        cfd.inc("A", "y")
        cfd.inc("B", "z")
        nltk_cfd.inc("A", "x")
        nltk_cfd.inc("A", "x")
        nltk_cfd.inc("A", "y")
        nltk_cfd.inc("B", "z")
        assert cfd.N() == nltk_cfd.N()
        assert set(cfd.conditions()) == set(nltk_cfd.conditions())
