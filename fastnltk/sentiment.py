"""fastnltk.sentiment — Drop-in replacement for nltk.sentiment."""

import nltk.sentiment as _nltk_sentiment
from nltk.sentiment import SentimentAnalyzer

from fastnltk._rust import SentimentIntensityAnalyzer as _RustSentimentIntensityAnalyzer

__all__ = ["SentimentAnalyzer", "SentimentIntensityAnalyzer"]


class SentimentIntensityAnalyzer:
    """VADER sentiment intensity analyzer — Rust-accelerated."""

    def __init__(self):
        self._impl = _RustSentimentIntensityAnalyzer()

    def polarity_scores(self, text: str) -> dict[str, float]:
        return self._impl.polarity_scores(text)


# ── NLTK re-exports ─────
sentiment_analyzer = _nltk_sentiment.sentiment_analyzer
vader = _nltk_sentiment.vader
