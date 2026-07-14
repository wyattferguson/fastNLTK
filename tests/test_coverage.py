"""Tests for PerceptronTagger, TnT, and LM variants."""

import pytest

from fastnltk.tag import PerceptronTagger, TnT, pos_tag_sents
from fastnltk.lm import StupidBackoff, WittenBellInterpolated
from fastnltk.parse import EarleyChartParser, CFG


class TestPerceptronTagger:
    def test_basic(self):
        tagger = PerceptronTagger()
        result = tagger.tag(["The", "cat", "runs"])
        assert len(result) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_single_word(self):
        tagger = PerceptronTagger()
        result = tagger.tag(["Hello"])
        assert len(result) == 1

    def test_empty(self):
        tagger = PerceptronTagger()
        result = tagger.tag([])
        assert result == []

    def test_tag_sents(self):
        tagger = PerceptronTagger()
        result = tagger.tag_sents([["The", "cat"], ["A", "dog"]])
        assert len(result) == 2
        assert all(len(s) == 2 for s in result)


class TestTnT:
    def test_basic(self):
        tagger = TnT()
        tagger.train([
            [("the", "DT"), ("cat", "NN")],
            [("the", "DT"), ("dog", "NN")],
        ])
        result = tagger.tag(["the", "cat"])
        assert len(result) == 2

    def test_single_word_train(self):
        tagger = TnT()
        tagger.train([[("hello", "UH")]])
        result = tagger.tag(["hello"])
        assert result[0][1] == "UH"

    def test_empty_train(self):
        tagger = TnT()
        tagger.train([])
        result = tagger.tag(["word"])
        assert len(result) == 1


class TestPosTagSents:
    def test_basic(self):
        result = pos_tag_sents([["The", "cat"], ["A", "dog"]])
        assert len(result) == 2
        assert all(isinstance(t, tuple) for s in result for t in s)


class TestLMVariants:
    def test_witten_bell(self):
        lm = WittenBellInterpolated(order=3)
        lm.fit([["the", "cat", "runs"], ["the", "dog", "barks"]])
        assert lm.fitted
        assert lm.order == 3
        score = lm.score("cat")
        assert isinstance(score, float)

    def test_stupid_backoff(self):
        lm = StupidBackoff(order=3)
        lm.fit([["the", "cat", "runs"], ["the", "dog", "barks"]])
        assert lm.fitted
        score = lm.score("cat")
        assert isinstance(score, float)

    def test_witten_bell_empty(self):
        lm = WittenBellInterpolated(order=3)
        lm.fit([])
        assert lm.fitted


class TestEarleyTrace:
    def test_parse_sents_batch(self):
        grammar = "S -> 'cat' 'runs'\n"
        cfg = CFG.from_string(grammar)
        parser = EarleyChartParser()
        results = parser.parse_sents(cfg, [["cat", "runs"], ["cat", "runs"]])
        assert len(results) == 2

    def test_parse_three_words(self):
        grammar = "S -> 'the' 'cat' 'runs'\n"
        cfg = CFG.from_string(grammar)
        parser = EarleyChartParser()
        results = parser.parse(cfg, ["the", "cat", "runs"])
        assert len(results) > 0
