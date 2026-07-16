"""Tests for Rust-accelerated translation metrics — NLTK compatibility."""

from fastnltk.translate import bleu_score, corpus_bleu


class TestBleuScore:
    def test_perfect_match(self):
        candidate = "the cat sat on the mat".split()
        reference = "the cat sat on the mat".split()
        score = bleu_score(candidate, reference)
        assert score > 0.9

    def test_no_match(self):
        candidate = "dog".split()
        reference = "cat".split()
        score = bleu_score(candidate, reference)
        assert score < 0.5

    def test_empty_candidate(self):
        score = bleu_score([], ["a"])
        assert score == 0.0


class TestCorpusBleu:
    def test_perfect_match(self):
        candidates = ["the cat sat on the mat".split()]
        references = ["the cat sat on the mat".split()]
        score = corpus_bleu(candidates, references)
        assert score > 0.9
