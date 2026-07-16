"""Tests for classify, CCG, corpus, inference, and robustness."""

import pytest

from fastnltk._rust import (
    CCGChartParser,
    CCGLexicon,
    ClosedWorldReasoner,
    DefaultReasoner,
    DefaultRule,
    DiscourseThread,
    ResolutionProver,
    TableauProver,
)
from fastnltk.ccg import fromstring as ccg_from_string
from fastnltk.classify import TextCat


class TestClassifyExtra:
    def test_textcat_guess(self):
        tc = TextCat()
        lang = tc.guess_language("the quick brown fox jumps over the lazy dog")
        assert lang is not None

    def test_textcat_scores(self):
        tc = TextCat()
        scores = tc.guess_language_scores("bonjour le monde")
        assert len(scores) > 0

    def test_textcat_supported_langs(self):
        langs = TextCat.supported_languages()
        assert len(langs) > 0

    def test_textcat_empty(self):
        tc = TextCat()
        lang = tc.guess_language("")
        assert isinstance(lang, str) or lang is None


class TestCCG:
    def test_parse_simple(self):
        cat = ccg_from_string("NP/N")
        assert str(cat) == "NP/N"

    def test_parse_complex(self):
        cat = ccg_from_string("(S\\NP)/NP")
        assert "S" in str(cat)
        assert "NP" in str(cat)

    def test_parse_primitive(self):
        cat = ccg_from_string("S")
        assert cat.is_primitive() is True

    def test_parse_invalid(self):
        with pytest.raises(Exception):
            ccg_from_string("")

    def test_ccg_lexicon(self):
        lex = CCGLexicon([("the", "NP/N"), ("cat", "N")])
        assert len(lex) == 2
        cats = lex.lookup("the")
        assert len(cats) == 1
        assert str(cats[0]) == "NP/N"

    def test_ccg_lexicon_missing(self):
        lex = CCGLexicon([("the", "NP/N")])
        cats = lex.lookup("unknown")
        assert cats == []

    def test_ccg_chart_parse(self):
        lex = CCGLexicon(
            [
                ("the", "NP/N"),
                ("cat", "N"),
                ("ran", "S\\NP"),
            ]
        )
        parser = CCGChartParser(lex)
        results = parser.parse(["the", "cat", "ran"])
        assert len(results) > 0


class TestInference:
    def test_tableau_excluded_middle(self):
        prover = TableauProver()
        result = prover.prove("P | ~P")
        assert result.success is True

    def test_resolution_excluded_middle(self):
        prover = ResolutionProver()
        result = prover.prove("P | ~P")
        assert result.success is True

    def test_tableau_false(self):
        prover = TableauProver()
        result = prover.prove("P & ~P")
        assert result.success is False

    def test_discourse_add_and_merge(self):
        thread = DiscourseThread()
        thread.add_drs("([x],[dog(x)])")
        thread.add_drs("([y],[cat(y)])")
        assert len(thread.get_drss()) == 2
        merged = thread.merge()
        assert "dog" in merged
        assert "cat" in merged

    def test_nonmonotonic_default(self):
        rule = DefaultRule("", "bird", "flies", "")
        reasoner = DefaultReasoner([rule])
        extensions = reasoner.extensions()
        assert len(extensions) > 0

    def test_nonmonotonic_closed_world(self):
        reasoner = ClosedWorldReasoner(["bird"])
        result = reasoner.query("dog")
        assert isinstance(result, bool)


class TestRobustness:
    """Stress-test with bad inputs — nothing should crash."""

    def test_empty_string_everywhere(self):
        from fastnltk.tokenize import sent_tokenize, word_tokenize

        sent_tokenize("")
        word_tokenize("")

    def test_very_long_single_word(self):
        from fastnltk.tokenize import word_tokenize

        long_word = "supercalifragilisticexpialidocious" * 100
        tokens = word_tokenize(long_word)
        assert len(tokens) >= 1

    def test_only_whitespace(self):
        from fastnltk.tokenize import sent_tokenize, word_tokenize

        sent_tokenize("    \n\n   ")
        word_tokenize("    \t\n  ")

    def test_null_bytes(self):
        from fastnltk.tokenize import word_tokenize

        text = "hello\x00world"
        tokens = word_tokenize(text)
        assert len(tokens) > 0

    def test_all_punctuation(self):
        from fastnltk.tokenize import word_tokenize

        tokens = word_tokenize("!@#$%^&*()_+-=[]{}|;:',.<>?/")
        assert len(tokens) > 0
