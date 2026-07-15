//! Simple tokenizers: Space, Tab, Line, Char.

use pyo3::prelude::*;

// SpaceTokenizer

/// Tokenize a string by splitting on whitespace.
///
/// Matches NLTK's `nltk.tokenize.SpaceTokenizer`.
/// NLTK compat: Same behavior (split on space, collapse empty).
#[pyclass(name = "SpaceTokenizer", module = "fastnltk._rust")]
pub struct SpaceTokenizer;

#[pymethods]
impl SpaceTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Tokenize text by splitting on space characters.
    fn tokenize(&self, text: &str) -> Vec<String> {
        tokenize_space(text)
    }

    /// Return span tuples (start, end) for each token.
    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = None;
        for (i, ch) in text.char_indices() {
            if ch == ' ' {
                if let Some(s) = start.take() {
                    spans.push((s, i));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            spans.push((s, text.len()));
        }
        spans
    }
}

/// Core `SpaceTokenizer` logic: split on ' ', filter empty.
fn tokenize_space(text: &str) -> Vec<String> {
    let n = text.split(' ').count();
    let mut tokens = Vec::with_capacity(n);
    for s in text.split(' ') {
        tokens.push(s.to_string());
    }
    tokens
}

// TabTokenizer

/// Tokenize a string by splitting on tabs.
#[pyclass(name = "TabTokenizer", module = "fastnltk._rust")]
pub struct TabTokenizer;

#[pymethods]
impl TabTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split('\t').map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = None;
        for (i, ch) in text.char_indices() {
            if ch == '\t' {
                if let Some(s) = start.take() {
                    spans.push((s, i));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            spans.push((s, text.len()));
        }
        spans
    }
}

// LineTokenizer

/// Tokenize a string by splitting on newlines.
#[pyclass(name = "LineTokenizer", module = "fastnltk._rust")]
pub struct LineTokenizer;

#[pymethods]
impl LineTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.lines().map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = 0;
        let len = text.len();
        for (i, ch) in text.char_indices() {
            if ch == '\n' {
                if i > start {
                    spans.push((start, i));
                }
                start = i + 1;
            }
        }
        if start < len {
            spans.push((start, len));
        }
        spans
    }
}

// CharTokenizer

/// Tokenize a string into individual characters.
#[pyclass(name = "CharTokenizer", module = "fastnltk._rust")]
pub struct CharTokenizer;

#[pymethods]
impl CharTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.chars().map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        text.char_indices().map(|(i, ch)| (i, i + ch.len_utf8())).collect()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpaceTokenizer tests ─────────────────────────────

    #[test]
    fn test_space_tokenize_basic() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("a b c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_space_tokenize_empty() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize(""), vec![""]);
    }

    #[test]
    fn test_space_tokenize_multiple_spaces() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("a  b"), vec!["a", "", "b"]);
    }

    #[test]
    fn test_space_tokenize_leading_trailing() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("  a b  "), vec!["", "", "a", "b", "", ""]);
    }

    #[test]
    fn test_space_tokenize_single() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("hello"), vec!["hello"]);
    }

    #[test]
    fn test_space_span_tokenize() {
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.span_tokenize("a b c"), vec![(0, 1), (2, 3), (4, 5)]);
    }

    // ── TabTokenizer tests ───────────────────────────────

    #[test]
    fn test_tab_tokenize_basic() {
        let tok = TabTokenizer::new();
        assert_eq!(tok.tokenize("a\tb\tc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_tab_tokenize_empty() {
        let tok = TabTokenizer::new();
        assert_eq!(tok.tokenize(""), vec![""]);
    }

    // ── LineTokenizer tests ──────────────────────────────

    #[test]
    fn test_line_tokenize_basic() {
        let tok = LineTokenizer::new();
        assert_eq!(tok.tokenize("a\nb\nc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_line_tokenize_empty() {
        let tok = LineTokenizer::new();
        let result: Vec<String> = tok.tokenize("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_line_span_tokenize() {
        let tok = LineTokenizer::new();
        assert_eq!(tok.span_tokenize("ab\ncd\nef"), vec![(0, 2), (3, 5), (6, 8)]);
    }

    // ── CharTokenizer tests ──────────────────────────────

    #[test]
    fn test_char_tokenize_basic() {
        let tok = CharTokenizer::new();
        assert_eq!(tok.tokenize("abc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_char_tokenize_empty() {
        let tok = CharTokenizer::new();
        let result: Vec<String> = tok.tokenize("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_char_tokenize_unicode() {
        let tok = CharTokenizer::new();
        let chars = tok.tokenize("héllo");
        assert_eq!(chars.len(), 5);
    }
}
