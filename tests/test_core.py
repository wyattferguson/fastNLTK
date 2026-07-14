"""Tests for language models, clustering, chat, and parsing."""

import pytest

from fastnltk.chat import Chat
from fastnltk.cluster import KMeansClusterer
from fastnltk.lm import (
    MLE,
    KneserNeyInterpolated,
    Laplace,
    Lidstone,
    StupidBackoff,
    WittenBellInterpolated,
)
from fastnltk.parse import CFG, EarleyChartParser


class TestLanguageModels:
    """Test all language model variants."""

    def test_mle_fit_and_score(self):
        lm = MLE(order=2)
        lm.fit([["the", "cat", "runs"], ["the", "dog", "barks"]])
        score = lm.score("cat", context=["the"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_mle_generate(self):
        lm = MLE(order=2)
        lm.fit([["the", "cat", "runs"]])
        generated = lm.generate(3)
        assert len(generated) == 3
        assert all(isinstance(w, str) for w in generated)

    def test_mle_order_property(self):
        lm = MLE(order=3)
        assert lm.order == 3

    def test_mle_not_fitted_initially(self):
        lm = MLE(order=2)
        assert not lm.fitted

    def test_mle_fitted_after_fit(self):
        lm = MLE(order=2)
        lm.fit([["a", "b"]])
        assert lm.fitted

    def test_lidstone(self):
        lm = Lidstone(order=2, gamma=0.1)
        lm.fit([["the", "cat", "runs"]])
        score = lm.score("cat", context=["the"])
        assert isinstance(score, float)

    def test_laplace(self):
        lm = Laplace(order=2)
        lm.fit([["the", "cat", "runs"]])
        score = lm.score("cat", context=["the"])
        assert isinstance(score, float)

    def test_kneser_ney(self):
        lm = KneserNeyInterpolated(order=3)
        lm.fit([["the", "cat", "runs"], ["the", "dog", "barks"]])
        score = lm.score("cat")
        assert isinstance(score, float)
        assert lm.order == 3

    def test_witten_bell(self):
        lm = WittenBellInterpolated(order=3)
        lm.fit([["the", "cat", "runs"]])
        score = lm.score("cat")
        assert isinstance(score, float)

    def test_stupid_backoff(self):
        lm = StupidBackoff(order=3)
        lm.fit([["the", "cat", "runs"]])
        score = lm.score("cat")
        assert isinstance(score, float)

    def test_fit_empty(self):
        lm = MLE(order=2)
        lm.fit([])
        # Should not crash

    def test_logscore(self):
        lm = MLE(order=2)
        lm.fit([["a", "b"]])
        ls = lm.logscore("b", context=["a"])
        assert ls <= 0.0  # log prob is <= 0

    def test_generate_zero(self):
        lm = MLE(order=2)
        lm.fit([["a", "b"]])
        gen = lm.generate(0)
        assert gen == []


class TestKMeans:
    def test_cluster_basic(self):
        clusterer = KMeansClusterer(num_clusters=2)
        vectors = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
        labels = clusterer.cluster(vectors)
        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_classify_after_cluster(self):
        clusterer = KMeansClusterer(num_clusters=2)
        vectors = [[0.0, 0.0], [10.0, 10.0]]
        clusterer.cluster(vectors)
        label = clusterer.classify([5.0, 5.0])
        assert label in (0, 1)

    def test_classify_before_cluster(self):
        clusterer = KMeansClusterer(num_clusters=2)
        with pytest.raises(Exception):
            clusterer.classify([1.0, 2.0])

    def test_centroids(self):
        clusterer = KMeansClusterer(num_clusters=2)
        vectors = [[0.0, 0.0], [10.0, 10.0]]
        clusterer.cluster(vectors)
        centroids = clusterer.centroids()
        assert len(centroids) == 2

    def test_single_vector(self):
        clusterer = KMeansClusterer(num_clusters=1)
        labels = clusterer.cluster([[1.0, 2.0, 3.0]])
        assert labels == [0]

    def test_empty_vectors(self):
        clusterer = KMeansClusterer(num_clusters=3)
        labels = clusterer.cluster([])
        assert labels == []


class TestChat:
    def test_respond_match(self):
        chat = Chat([(r"hello|hi", ["Hello!", "Hi there!"])])
        resp = chat.respond("hello")
        assert resp in ("Hello!", "Hi there!")

    def test_respond_no_match(self):
        chat = Chat([(r"hello", ["Hi"])])
        resp = chat.respond("goodbye")
        assert "don't understand" in resp.lower() or resp == "I don't understand."

    def test_converse(self):
        chat = Chat([(r"hi", ["Hello"]), (r"bye", ["Goodbye"])])
        resp, idx = chat.converse("bye")
        assert resp == "Goodbye"
        assert idx == 1

    def test_converse_no_match(self):
        chat = Chat([(r"hi", ["Hello"])])
        resp, idx = chat.converse("unknown")
        assert "don't understand" in resp.lower()
        assert idx == -1

    def test_multiple_patterns(self):
        chat = Chat([(r"hello|hi", ["Hi"]), (r"bye", ["Bye"])])
        assert chat.respond("hi") == "Hi"
        assert chat.respond("bye") == "Bye"

    def test_empty_pairs(self):
        chat = Chat([])
        resp = chat.respond("hello")
        assert isinstance(resp, str)


class TestCFG:
    def test_from_string(self):
        grammar = """
            S -> NP VP
            NP -> Det N
            VP -> V
            Det -> 'the'
            N -> 'cat'
            V -> 'runs'
        """
        cfg = CFG.from_string(grammar)
        assert cfg.start() == "S"
        assert len(cfg) >= 5  # at least 5 productions

    def test_productions(self):
        cfg = CFG("S", [("S", ["NP", "VP"])])
        prods = cfg.productions()
        assert len(prods) == 1
        assert prods[0] == ("S", ["NP", "VP"])

    def test_nonterminals(self):
        cfg = CFG.from_string("S -> NP VP\nNP -> Det N")
        nt = cfg.nonterminals()
        assert "S" in nt
        assert "NP" in nt

    def test_empty_grammar(self):
        with pytest.raises(Exception):
            CFG.from_string("")

    def test_str(self):
        cfg = CFG.from_string("S -> NP VP")
        s = str(cfg)
        assert "S" in s


class TestEarleyParser:
    def test_parse_simple(self):
        grammar = "S -> 'cat' 'runs'\n"
        cfg = CFG.from_string(grammar)
        parser = EarleyChartParser()
        results = parser.parse(cfg, ["cat", "runs"])
        assert len(results) > 0

    def test_parse_no_match(self):
        grammar = "S -> 'cat'"
        cfg = CFG.from_string(grammar)
        parser = EarleyChartParser()
        with pytest.raises(Exception):
            parser.parse(cfg, ["dog"])

    def test_parse_empty(self):
        cfg = CFG.from_string("S -> 'a'")
        parser = EarleyChartParser()
        with pytest.raises(Exception):
            parser.parse(cfg, [])

    def test_parse_sents(self):
        cfg = CFG.from_string("S -> 'cat'")
        parser = EarleyChartParser()
        results = parser.parse_sents(cfg, [["cat"], ["cat"]])
        assert len(results) == 2
