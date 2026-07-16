"""Tests for taggers — POS tagging, sequential taggers, edge cases."""

import pytest

from fastnltk.tag import (
    AffixTagger,
    BigramTagger,
    DefaultTagger,
    RegexpTagger,
    TrigramTagger,
    UnigramTagger,
    pos_tag,
)


class TestDefaultTagger:
    def test_single_tag(self):
        tagger = DefaultTagger("NN")
        result = tagger.tag(["the", "cat", "runs"])
        assert result == [("the", "NN"), ("cat", "NN"), ("runs", "NN")]

    def test_empty_tokens(self):
        tagger = DefaultTagger("NN")
        result = tagger.tag([])
        assert result == []

    def test_sents(self):
        tagger = DefaultTagger("NN")
        result = tagger.tag_sents([["the", "cat"], ["a", "dog"]])
        assert len(result) == 2
        assert result[0] == [("the", "NN"), ("cat", "NN")]

    def test_unicode(self):
        tagger = DefaultTagger("NN")
        result = tagger.tag(["café", "naïve", "💻"])
        assert result == [("café", "NN"), ("naïve", "NN"), ("💻", "NN")]

    def test_single_token(self):
        tagger = DefaultTagger("VB")
        result = tagger.tag(["run"])
        assert result == [("run", "VB")]


class TestUnigramTagger:
    def test_train_and_tag(self):
        tagger = UnigramTagger()
        tagger.train(
            [
                [("the", "DT"), ("cat", "NN")],
                [("the", "DT"), ("dog", "NN")],
            ]
        )
        result = tagger.tag(["the", "cat"])
        assert result[0] == ("the", "DT")
        assert result[1] == ("cat", "NN")

    def test_empty_train(self):
        tagger = UnigramTagger()
        tagger.train([])
        # Should not crash — uses most-common or backoff
        result = tagger.tag(["unknown"])
        assert len(result) == 1

    def test_tag_sents(self):
        tagger = UnigramTagger()
        tagger.train(
            [
                [("the", "DT"), ("cat", "NN")],
            ]
        )
        result = tagger.tag_sents([["the"], ["cat"]])
        assert result == [[("the", "DT")], [("cat", "NN")]]


class TestBigramTagger:
    def test_train_and_tag(self):
        tagger = BigramTagger()
        tagger.train(
            [
                [("the", "DT"), ("cat", "NN")],
                [("the", "DT"), ("dog", "NN")],
            ]
        )
        result = tagger.tag(["the", "cat"])
        assert result[0] == ("the", "DT")

    def test_empty_tokens(self):
        tagger = BigramTagger()
        tagger.train([[("the", "DT")]])
        result = tagger.tag([])
        assert result == []


class TestTrigramTagger:
    def test_train_and_tag(self):
        tagger = TrigramTagger()
        tagger.train(
            [
                [("the", "DT"), ("big", "JJ"), ("cat", "NN")],
            ]
        )
        result = tagger.tag(["the", "big", "cat"])
        assert result[0] == ("the", "DT")


class TestAffixTagger:
    def test_suffix(self):
        tagger = AffixTagger(affix_len=3, use_suffix=True)
        tagger.train(
            [
                [("running", "VBG"), ("walking", "VBG")],
                [("jumped", "VBD"), ("walked", "VBD")],
            ]
        )
        result = tagger.tag(["running"])
        assert result[0][1] == "VBG"

    def test_prefix(self):
        tagger = AffixTagger(affix_len=3, use_suffix=False)
        tagger.train(
            [
                [("unhappy", "JJ"), ("unclear", "JJ")],
            ]
        )
        result = tagger.tag(["unhappy"])
        assert result[0][1] == "JJ"

    def test_empty_train(self):
        tagger = AffixTagger()
        tagger.train([])
        result = tagger.tag(["word"])
        assert len(result) == 1


class TestRegexpTagger:
    def test_basic_patterns(self):
        patterns = [
            (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),
            (r".*ing$", "VBG"),
            (r".*ed$", "VBD"),
        ]
        tagger = RegexpTagger(patterns)
        result = tagger.tag(["running", "walked", "42", "unknown"])
        assert result[0][1] == "VBG"
        assert result[1][1] == "VBD"
        assert result[2][1] == "CD"

    def test_empty_patterns(self):
        tagger = RegexpTagger([])
        result = tagger.tag(["word"])
        assert result[0][1] is None or result[0][1] == ""


class TestPosTag:
    def test_basic(self):
        tokens = ["The", "cat", "runs"]
        result = pos_tag(tokens)
        assert len(result) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
        assert all(isinstance(tag, str) for _, tag in result)

    def test_single_word(self):
        result = pos_tag(["The"])
        assert len(result) == 1
        assert result[0][0] == "The"

    def test_unicode(self):
        result = pos_tag(["café", "naïve"])
        assert len(result) == 2

    def test_matches_nltk(self):
        import nltk

        tokens = ["The", "cat", "runs", "fast"]
        fast = pos_tag(tokens)
        try:
            nltk_tags = nltk.pos_tag(tokens)
        except LookupError:
            pytest.skip("requires NLTK data")
        assert fast == nltk_tags
