"""Edge case tests — Unicode, large inputs, empty/null, special characters."""

import pytest

from fastnltk.stem import (
    ISRIStemmer,
    PorterStemmer,
    RSLPStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from fastnltk.tokenize import (
    RegexpTokenizer,
    SpaceTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    WordPunctTokenizer,
    sent_tokenize,
    word_tokenize,
)


class TestEmptyInputs:
    """Every public function should handle empty inputs gracefully."""

    def test_tokenize_empty(self):
        assert sent_tokenize("") in ([], [""])  # behavior varies
        assert word_tokenize("") == []
        assert SpaceTokenizer().tokenize("") == [""]
        assert TreebankWordTokenizer().tokenize("") == []
        assert TweetTokenizer().tokenize("") == []
        assert RegexpTokenizer(r"\w+").tokenize("") == []
        assert WordPunctTokenizer().tokenize("") == []

    def test_stem_empty(self):
        assert PorterStemmer().stem("") == ""
        assert SnowballStemmer("english").stem("") == ""

    def test_lemmatize_empty(self):
        assert WordNetLemmatizer().lemmatize("") == ""


class TestUnicode:
    """All tokenizers should handle Unicode text gracefully."""

    def test_cjk_chinese(self):
        text = "你好世界这是测试"
        tokens = word_tokenize(text)
        assert len(tokens) > 0

    def test_emoji(self):
        text = "Hello 👋 world 🌍"
        tokens = word_tokenize(text)
        assert "world" in tokens

    def test_rtl_arabic(self):
        text = "مرحبا بالعالم"
        tokens = word_tokenize(text)
        assert len(tokens) > 0

    def test_mixed_scripts(self):
        text = "English русский 日本語 한글"
        tokens = word_tokenize(text)
        assert len(tokens) >= 4

    def test_accents(self):
        text = "café naïve über señor"
        tokens = word_tokenize(text)
        assert "café" in tokens
        assert "naïve" in tokens

    def test_tweet_unicode(self):
        text = "Hello @user #hashtag café 💻"
        tokens = TweetTokenizer().tokenize(text)
        assert len(tokens) > 0

    def test_stem_unicode_french(self):
        stemmer = SnowballStemmer("french")
        result = stemmer.stem("courant")
        assert len(result) > 0
        assert result != "courant"  # should actually stem

    def test_stem_non_latin(self):
        stemmer = ISRIStemmer()
        result = stemmer.stem("المكتبات")
        assert len(result) > 0


class TestLargeInputs:
    """Performance-related: large inputs should not panic or hang."""

    def test_large_sent_tokenize(self):
        text = "Hello world. " * 1000
        sentences = sent_tokenize(text)
        assert len(sentences) == 1000

    def test_large_word_tokenize(self):
        text = "the cat runs " * 5000
        tokens = word_tokenize(text)
        assert len(tokens) > 10000

    def test_large_regexp_tokenize(self):
        text = "word " * 10000
        tokens = RegexpTokenizer(r"\w+").tokenize(text)
        assert len(tokens) == 10000

    def test_many_stems(self):
        stemmer = PorterStemmer()
        for w in ["running", "flies", "cats", "dogs"] * 1000:
            result = stemmer.stem(w)
            assert isinstance(result, str)
            assert len(result) > 0


class TestSpecialCharacters:
    """Special and unusual characters should be handled."""

    def test_newlines(self):
        text = "line1\nline2\n\nline3"
        sentences = sent_tokenize(text)
        assert len(sentences) > 0

    def test_only_punctuation(self):
        text = "!!! ??? ... ---"
        tokens = word_tokenize(text)
        assert len(tokens) > 0  # should get individual punct tokens

    def test_urls(self):
        text = "Visit https://example.com today"
        tokens = word_tokenize(text)
        assert "https://example.com" in tokens or any("https" in t for t in tokens)

    def test_html_entities(self):
        text = "Hello &amp; world"
        tokens = word_tokenize(text)
        assert len(tokens) >= 2

    def test_control_characters(self):
        text = "Hello\x00world\x01"
        tokens = word_tokenize(text)
        assert len(tokens) > 0


class TestStemmerEdgeCases:
    """Stemmer-specific edge cases."""

    def test_porter_very_short(self):
        s = PorterStemmer()
        assert s.stem("a") == "a"
        assert s.stem("I") == "i"  # porter lowercases

    def test_rslp_portuguese(self):
        s = RSLPStemmer()
        result = s.stem("correndo")
        assert len(result) > 0
        assert isinstance(result, str)

    def test_wordnet_lemmatizer_noun(self):
        lemmatizer = WordNetLemmatizer()
        result = lemmatizer.lemmatize("cats", pos="n")
        assert result in ("cat", "cats")  # falls back if no WordNet data

    def test_wordnet_lemmatizer_verb(self):
        lemmatizer = WordNetLemmatizer()
        result = lemmatizer.lemmatize("running", pos="v")
        assert isinstance(result, str)

    def test_snowball_multiple_languages(self):
        for lang in ["english", "french", "spanish", "german", "italian"]:
            stemmer = SnowballStemmer(lang)
            result = stemmer.stem("test")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_snowball_invalid_language(self):
        with pytest.raises(Exception):
            SnowballStemmer("klingon")
