//! Regexp-based tokenizers: RegexpTokenizer, WhitespaceTokenizer,
//! WordPunctTokenizer, BlanklineTokenizer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

use crate::util::regex_cache;

// ═══════════════════════════════════════════════════════════
// RegexpTokenizer
// ═══════════════════════════════════════════════════════════

/// Tokenize a string using a regular expression pattern.
///
/// If ``gaps`` is True, the pattern is used to find separators
/// (splitting the text). Otherwise, the pattern is used to find matches.
#[pyclass(name = "RegexpTokenizer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RegexpTokenizer {
    pattern: String,
    gaps: bool,
    flags: u32,
}

#[pymethods]
impl RegexpTokenizer {
    #[new]
    #[pyo3(signature = (pattern="\\w+", gaps=false, flags=0))]
    fn new(pattern: &str, gaps: bool, flags: u32) -> Self {
        let pattern = pattern.to_string();
        Self { pattern, gaps, flags }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        let re = regex_cache::get_or_compile(&self.pattern, self.flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(if self.gaps {
            re.split(text).filter(|s| !s.is_empty()).map(String::from).collect()
        } else {
            re.find_iter(text).map(|m| m.as_str().to_string()).collect()
        })
    }

    fn span_tokenize(&self, text: &str) -> PyResult<Vec<(usize, usize)>> {
        let re = regex_cache::get_or_compile(&self.pattern, self.flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(if self.gaps {
            let mut spans = Vec::new();
            let mut start = 0;
            for m in re.find_iter(text) {
                if m.start() > start {
                    spans.push((start, m.start()));
                }
                start = m.end();
            }
            if start < text.len() {
                spans.push((start, text.len()));
            }
            spans
        } else {
            re.find_iter(text).map(|m| (m.start(), m.end())).collect()
        })
    }
}

// ═══════════════════════════════════════════════════════════
// WhitespaceTokenizer
// ═══════════════════════════════════════════════════════════

/// Tokenize on whitespace.
#[pyclass(name = "WhitespaceTokenizer", module = "fastnltk._rust")]
pub struct WhitespaceTokenizer;

#[pymethods]
impl WhitespaceTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start: Option<usize> = None;
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
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

// ═══════════════════════════════════════════════════════════
// WordPunctTokenizer
// ═══════════════════════════════════════════════════════════

/// Tokenize into sequences of alphabetic and non-alphabetic characters.
#[pyclass(name = "WordPunctTokenizer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct WordPunctTokenizer {
    re: Regex,
}

#[pymethods]
impl WordPunctTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        let re = Regex::new(r"\w+|[^\w\s]+").map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { re })
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.re.find_iter(text).map(|m| m.as_str().to_string()).collect()
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        self.re.find_iter(text).map(|m| (m.start(), m.end())).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// BlanklineTokenizer
// ═══════════════════════════════════════════════════════════

/// Tokenize on blank lines (empty lines or lines with only whitespace).
#[pyclass(name = "BlanklineTokenizer", module = "fastnltk._rust")]
pub struct BlanklineTokenizer;

#[pymethods]
impl BlanklineTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut paragraphs = Vec::new();
        let mut current = String::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                if !current.is_empty() {
                    paragraphs.push(current.trim().to_string());
                    current = String::new();
                }
            } else {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(line.trim());
            }
        }
        if !current.is_empty() {
            paragraphs.push(current.trim().to_string());
        }
        paragraphs
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regexp_tokenize_words() {
        let tok = RegexpTokenizer::new(r"\w+", false, 0);
        let result = tok.tokenize("Hello, world!").unwrap();
        assert_eq!(result, vec!["Hello", "world"]);
    }

    #[test]
    fn test_regexp_tokenize_gaps() {
        let tok = RegexpTokenizer::new(r"\s+", true, 0);
        let result = tok.tokenize("a b  c").unwrap();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_whitespace_tokenize() {
        let tok = WhitespaceTokenizer::new();
        let result = tok.tokenize("Hello   world!\tTest");
        assert_eq!(result, vec!["Hello", "world!", "Test"]);
    }

    #[test]
    fn test_wordpunct_tokenize() {
        let tok = WordPunctTokenizer::new().unwrap();
        let result = tok.tokenize("Hello, world!");
        assert_eq!(result, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_blankline_tokenize() {
        let tok = BlanklineTokenizer::new();
        let result = tok.tokenize("Para one.\n\nPara two.\n\nPara three.");
        assert_eq!(result, vec!["Para one.", "Para two.", "Para three."]);
    }
}
