"""Tests for Rust-accelerated sentiment — NLTK compatibility."""


from fastnltk.sentiment import SentimentIntensityAnalyzer


class TestSentimentIntensityAnalyzer:
    def test_positive_sentiment(self):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("This is wonderful and amazing!")
        assert "compound" in scores
        assert scores["pos"] > 0

    def test_negative_sentiment(self):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("This is terrible and awful!")
        assert "compound" in scores
        assert scores["neg"] > 0

    def test_neutral_sentiment(self):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("The book is on the table.")
        assert "compound" in scores

    def test_empty_text(self):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("")
        assert "compound" in scores
        assert scores["compound"] == 0.0
