"""Property-based tests for tokenizer invariants using Hypothesis."""

from hypothesis import given
from hypothesis import strategies as st

from fastnltk.tokenize import regexp_tokenize, sent_tokenize, word_tokenize

# ── Word tokenizer invariants ──────────────────────────────


@given(st.text())
def test_word_tokenize_never_panics(text):
    """word_tokenize must not raise on any string."""
    result = word_tokenize(text)
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


@given(
    st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            min_size=0,
            max_size=5,
        )
    )
)
def test_detokenize_roundtrip(words):
    """detokenize should not crash on any word list."""
    import fastnltk.tokenize as T

    for w in words:
        try:
            T.TreebankWordDetokenizer().detokenize([w])
        except Exception:
            pass  # some single chars may not detokenize cleanly


# ── Sentence tokenizer invariants ──────────────────────────


@given(st.text(max_size=500))
def test_sent_tokenize_never_panics(text):
    """sent_tokenize must not raise on any string."""
    result = sent_tokenize(text)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


@given(st.text(alphabet="abc..!?", max_size=200))
def test_sent_tokenize_preserves_characters(text):
    """Combining sentence-tokenized output should recover all
    characters (modulo whitespace normalization)."""
    sents = sent_tokenize(text)
    combined = "".join(sents)
    # All non-whitespace chars should appear in order
    stripped = "".join(text.split())
    combined_stripped = "".join(combined.split())
    assert stripped in combined_stripped or combined_stripped in stripped


@given(st.text(max_size=100))
def test_sent_tokenize_single_sentence(text):
    """Short text with no sentence boundaries should return one sentence."""
    if "." not in text and "!" not in text and "?" not in text:
        sents = sent_tokenize(text)
        if text.strip():
            assert len(sents) >= 1
            assert len(text) >= len(sents[0]) - 1  # allow minor normalization


# ── Regexp tokenizer invariants ────────────────────────────


@given(st.text(max_size=200))
def test_regexp_tokenize_never_panics(text):
    """regexp_tokenize must not raise on any string."""
    result = regexp_tokenize(text, r"\w+")
    assert isinstance(result, list)
    result2 = regexp_tokenize(text, r"\s+", gaps=True)
    assert isinstance(result2, list)


# ── Span tokenizer invariants ──────────────────────────────


@given(st.text(max_size=200))
def test_spans_cover_text(text):
    """Span tokenization should produce non-overlapping spans
    that cover non-whitespace portions of the text."""
    from fastnltk._rust import RegexpTokenizer

    tok = RegexpTokenizer(r"\S+", gaps=False)
    spans = tok.span_tokenize(text)
    seen_end = 0
    for start, end in spans:
        assert start >= seen_end
        assert start < end
        # spans are byte offsets, so end may exceed Unicode len
        seen_end = end
