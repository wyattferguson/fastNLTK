//! Regexp-based tokenizers: `RegexpTokenizer`, `WhitespaceTokenizer`,

use memchr::memchr3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use smol_str::SmolStr;

use crate::util::regex_cache;

// RegexpTokenizer

/// Tokenize a string using a regular expression pattern.
///
/// If ``gaps`` is True, the pattern is used to find separators
/// (splitting the text). Otherwise, the pattern is used to find matches.
///
/// Fast paths: patterns `\S+` and `\s+` use a manual char scanner
/// instead of the regex engine (~5x faster for whitespace splitting).
#[pyclass(name = "RegexpTokenizer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RegexpTokenizer {
    pattern: String,
    gaps: bool,
    flags: u32,
    /// True if pattern is `\S+` (or `\s+` in gaps mode) — use fast char scanner
    is_simple_whitespace: bool,
}

/// SIMD-accelerated whitespace tokenizer.
/// Uses memchr3 to find space/tab/newline with SSE2/AVX2/NEON.
fn tokenize_whitespace(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        // Skip standalone \r (not in memchr3 set) and other ASCII ws
        if bytes[start].is_ascii_whitespace() {
            start += 1;
            continue;
        }
        match memchr3(b' ', b'\t', b'\n', &bytes[start..]) {
            Some(rel) => {
                let ws_pos = start + rel; // position of whitespace byte
                // Exclude trailing \r before \n from token
                let mut token_end = ws_pos;
                if bytes[ws_pos] == b'\n' && token_end > start && bytes[token_end - 1] == b'\r' {
                    token_end -= 1;
                }
                tokens.push(text[start..token_end].to_string());
                start = ws_pos + 1;
                // Skip consecutive ASCII whitespace
                while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                    start += 1;
                }
            }
            None => {
                tokens.push(text[start..].to_string());
                break;
            }
        }
    }
    tokens
}

/// SIMD-accelerated whitespace span finder.
fn span_tokenize_whitespace(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        if bytes[start].is_ascii_whitespace() {
            start += 1;
            continue;
        }
        match memchr3(b' ', b'\t', b'\n', &bytes[start..]) {
            Some(rel) => {
                let ws_pos = start + rel;
                let mut token_end = ws_pos;
                if bytes[ws_pos] == b'\n' && token_end > start && bytes[token_end - 1] == b'\r' {
                    token_end -= 1;
                }
                spans.push((start, token_end));
                start = ws_pos + 1;
                while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                    start += 1;
                }
            }
            None => {
                spans.push((start, bytes.len()));
                break;
            }
        }
    }
    spans
}

/// Gap tokenizer: same as tokenize_whitespace (both return non-ws runs).
fn split_whitespace_gaps(text: &str) -> Vec<String> {
    tokenize_whitespace(text)
}

/// Gap span finder.
fn span_split_whitespace_gaps(text: &str) -> Vec<(usize, usize)> {
    span_tokenize_whitespace(text)
}

fn is_simple_whitespace_pattern(pattern: &str, gaps: bool) -> bool {
    if gaps {
        pattern == r"\s+" || pattern == "[\\s]+" || pattern == "[ \\t\\n\\r\\f]+"
    } else {
        pattern == r"\S+" || pattern == "[\\S]+" || pattern == "[^ \\t\\n\\r\\f]+"
    }
}

#[pymethods]
impl RegexpTokenizer {
    #[new]
    #[pyo3(signature = (pattern="\\w+", gaps=false, flags=0))]
    fn new(pattern: &str, gaps: bool, flags: u32) -> Self {
        let is_simple = is_simple_whitespace_pattern(pattern, gaps) && flags == 0;
        Self {
            pattern: pattern.to_string(),
            gaps,
            flags,
            is_simple_whitespace: is_simple,
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        if self.is_simple_whitespace {
            return Ok(if self.gaps {
                split_whitespace_gaps(text)
            } else {
                tokenize_whitespace(text)
            });
        }
        let re = regex_cache::get_or_compile(&self.pattern, self.flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(if self.gaps {
            re.split(text).filter(|s| !s.is_empty()).map(|s| SmolStr::new(s).to_string()).collect()
        } else {
            re.find_iter(text).map(|m| SmolStr::new(m.as_str()).to_string()).collect()
        })
    }

    fn span_tokenize(&self, text: &str) -> PyResult<Vec<(usize, usize)>> {
        if self.is_simple_whitespace {
            return Ok(if self.gaps {
                span_split_whitespace_gaps(text)
            } else {
                span_tokenize_whitespace(text)
            });
        }
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

// WhitespaceTokenizer

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

// WordPunctTokenizer

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

// BlanklineTokenizer

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

// Tests

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
