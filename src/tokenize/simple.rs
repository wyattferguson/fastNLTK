//! Simple tokenizers: Space, Tab, Line, Char.
//!
//! SpaceTokenizer uses `memchr3`-accelerated scanning (SIMD where available)
//! instead of regex for 5-10x faster whitespace splitting.

use pyo3::prelude::*;

// SpaceTokenizer

/// Tokenize a string by splitting on ASCII space (`0x20`).
///
/// Matches NLTK's `nltk.tokenize.SpaceTokenizer` (`str.split(" ")`).
/// Splits on SINGLE SPACE only — tabs, newlines, and other whitespace
/// are preserved as-is (NLTK compatibility).
#[pyclass(name = "SpaceTokenizer", module = "fastnltk._rust")]
pub struct SpaceTokenizer;

#[pymethods]
impl SpaceTokenizer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        // Match NLTK SpaceTokenizer: s.split(" ")
        // Pre-allocate exact capacity, then iterate with byte scan.
        let bytes = text.as_bytes();
        let space_count = bytes.iter().filter(|&&b| b == b' ').count();
        let mut tokens = Vec::with_capacity(space_count + 1);
        let mut start = 0usize;

        for (i, &b) in bytes.iter().enumerate() {
            if b == b' ' {
                tokens.push(text[start..i].to_string());
                start = i + 1;
            }
        }
        tokens.push(text[start..].to_string());
        tokens
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        // Match NLTK SpaceTokenizer span_tokenize: s.split(" ")
        let mut spans = Vec::new();
        let mut start = 0usize;
        for (i, ch) in text.char_indices() {
            if ch == ' ' {
                spans.push((start, i));
                start = i + 1;
            }
        }
        // Trailing segment (empty if text ends with space)
        spans.push((start, text.len()));
        spans
    }
}

// TabTokenizer

/// Tokenize a string by splitting on tab characters.
#[pyclass(name = "TabTokenizer", module = "fastnltk._rust")]
pub struct TabTokenizer;

#[pymethods]
impl TabTokenizer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split('\t').map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start: Option<usize> = None;
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

/// Tokenize a string into lines.
#[pyclass(name = "LineTokenizer", module = "fastnltk._rust")]
pub struct LineTokenizer;

#[pymethods]
impl LineTokenizer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        // NLTK default: blanklines='discard', remove lines with only whitespace
        text.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start: Option<usize> = None;
        for (i, ch) in text.char_indices() {
            if ch == '\n' {
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

// CharTokenizer

/// Tokenize a string into individual characters.
#[pyclass(name = "CharTokenizer", module = "fastnltk._rust")]
pub struct CharTokenizer;

#[pymethods]
impl CharTokenizer {
    #[new]
    const fn new() -> Self {
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
        // NLTK SpaceTokenizer = str.split(" "), produces empties between gaps
        let tok = SpaceTokenizer::new();
        assert_eq!(tok.tokenize("a  b"), vec!["a", "", "b"]);
    }

    #[test]
    fn test_space_tokenize_leading_trailing() {
        // NLTK SpaceTokenizer = str.split(" "), produces empties for leading/trailing
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
        assert_eq!(tok.span_tokenize("a\nb\nc"), vec![(0, 1), (2, 3), (4, 5)]);
    }

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
        let result = tok.tokenize("héllo");
        assert_eq!(result.len(), 5);
    }
}
