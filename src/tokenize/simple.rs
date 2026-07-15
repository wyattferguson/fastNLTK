//! Simple tokenizers: Space, Tab, Line, Char.

use pyo3::prelude::*;

// SpaceTokenizer

/// Tokenize a string by splitting on whitespace.
///
/// Matches NLTK's `nltk.tokenize.SpaceTokenizer`.
#[pyclass(name = "SpaceTokenizer", module = "fastnltk._rust")]
pub struct SpaceTokenizer;

#[pymethods]
impl SpaceTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(' ').map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start: Option<usize> = None;
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

// TabTokenizer

/// Tokenize a string by splitting on tab characters.
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
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.lines().map(String::from).collect()
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
