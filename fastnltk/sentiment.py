"""fastnltk.sentiment — Drop-in replacement for nltk.sentiment."""

_rust_available = False
try:
    from fastnltk._rust import SentimentIntensityAnalyzer as _RustSentimentIntensityAnalyzer
    _rust_available = True
except ImportError:
    pass

import nltk.sentiment as _nltk_sentiment
from nltk.sentiment import SentimentAnalyzer

__all__ = ["SentimentAnalyzer", "SentimentIntensityAnalyzer"]


class SentimentIntensityAnalyzer:
    """VADER sentiment intensity analyzer — Rust-accelerated."""
    def __init__(self):
        if _rust_available:
            self._impl = _RustSentimentIntensityAnalyzer()
        else:
            self._impl = _nltk_sentiment.SentimentIntensityAnalyzer()

    def polarity_scores(self, text):
        return self._impl.polarity_scores(text)
