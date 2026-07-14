"""fastnltk.sentiment — Drop-in replacement for nltk.sentiment."""

import warnings

import nltk.sentiment as _nltk_sentiment
from nltk.sentiment import SentimentAnalyzer

_rust_available = False
try:
    from fastnltk._rust import SentimentIntensityAnalyzer as _RustSentimentIntensityAnalyzer
    _rust_available = True
except ImportError:
    warnings.warn(
        "fastnltk._rust extension not available; falling back to pure-NLTK sentiment"
    )

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


# ── NLTK re-exports for API compatibility ─────

# Submodule pass-through
sentiment_analyzer = _nltk_sentiment.sentiment_analyzer
vader = _nltk_sentiment.vader
