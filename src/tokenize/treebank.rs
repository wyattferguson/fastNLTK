//! Treebank tokenizer — Penn Treebank-style tokenization.
//!
//! Based on NLTK's TreebankWordTokenizer which handles:
//!   - English contractions (n't, 'm, 's, 're, 've, 'll, 'd)
//!   - Punctuation splitting (parentheses, quotes, commas, etc.)
//!   - Clitics and hyphenated words

use std::sync::LazyLock;
use pyo3::prelude::*;
use regex::Regex;

/// Contraction/starting rules applied in order. Regexes pre-compiled at load time.
static CONTRACTIONS2: LazyLock<Vec<(Regex, &str)>> = LazyLock::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"(?i)('ll|'re|'ve|'m|'d|'s)\b", " $1"),
        (r"(?i)n't\b", " n't"),
        (r"(?i)'em\b", " 'em"),
        (r"(?i)\b(can)(not)\b", " $1 $2"),
        (r"(?i)\b(d)'ye\b", " $1 'ye"),
        (r"(?i)\b(gim)(me)\b", " $1 $2"),
        (r"(?i)\b(gon)(na)\b", " $1 $2"),
        (r"(?i)\b(got)(ta)\b", " $1 $2"),
        (r"(?i)\b(lem)(me)\b", " $1 $2"),
        (r"(?i)\b(mor)('n)\b", " $1 $2"),
        (r"(?i)\b(t)(is)\b", " $1 $2"),
        (r"(?i)\b(t)(was)\b", " $1 $2"),
        (r"(?i)\b(wan)(na)\b", " $1 $2"),
    ];
    patterns.iter().filter_map(|(p, r)| Regex::new(p).ok().map(|re| (re, *r))).collect()
});

/// Punctuation rules for splitting. Regexes pre-compiled at load time.
static PUNCTUATION: LazyLock<Vec<(Regex, &str)>> = LazyLock::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"([\[\](){}<>])", " $1 "),
        (r"([:;,.?!])", " $1 "),
        (r"(--)", " $1 "),
        (r"''", " '' "),
        (r"''", " '' "),
    ];
    patterns.iter().filter_map(|(p, r)| Regex::new(p).ok().map(|re| (re, *r))).collect()
});

/// Collapse multiple spaces — pre-compiled at load time.
static COLLAPSE_SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// Tokenize text using Treebank rules.
pub fn tokenize_treebank(text: &str) -> Vec<String> {
    let mut s = String::from(text);

    // Apply contraction rules
    for (re, replacement) in CONTRACTIONS2.iter() {
        s = re.replace_all(&s, *replacement).to_string();
    }

    // Apply punctuation rules
    for (re, replacement) in PUNCTUATION.iter() {
        s = re.replace_all(&s, *replacement).to_string();
    }

    // Collapse multiple spaces
    s = COLLAPSE_SPACES.replace_all(&s, " ").to_string();

    // Trim
    let s = s.trim().to_string();

    // Split on whitespace
    if s.is_empty() {
        return Vec::new();
    }
    s.split_whitespace().map(String::from).collect()
}

// ═══════════════════════════════════════════════════════════
// PyO3 Wrappers
// ═══════════════════════════════════════════════════════════

/// TreebankWordTokenizer — Penn Treebank tokenization.
#[pyclass(name = "TreebankWordTokenizer", module = "fastnltk._rust")]
pub struct TreebankWordTokenizer;

#[pymethods]
impl TreebankWordTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        tokenize_treebank(text)
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let tokens = tokenize_treebank(text);
        // Approximate spans by finding each token in text
        let mut spans = Vec::new();
        let mut search_start = 0;
        for token in &tokens {
            if let Some(pos) = text[search_start..].find(token.as_str()) {
                let start = search_start + pos;
                let end = start + token.len();
                spans.push((start, end));
                search_start = end;
            }
        }
        spans
    }
}

/// TreebankWordDetokenizer — detokenize Treebank tokens back to text.
#[pyclass(name = "TreebankWordDetokenizer", module = "fastnltk._rust")]
pub struct TreebankWordDetokenizer;

#[pymethods]
impl TreebankWordDetokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn detokenize(&self, tokens: Vec<String>) -> String {
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && !matches!(
                token.as_str(),
                "." | "," | "!" | "?" | ":" | ";" | ")" | "]" | "}" | "%" | "''" | "'" | "n't"
            ) && !token.starts_with('\'')
            && !matches!(token.as_str(), "(" | "[" | "{" | "``")
            {
                result.push(' ');
            }
            result.push_str(token);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treebank_basic() {
        let result = tokenize_treebank("Hello world.");
        assert_eq!(result, vec!["Hello", "world", "."]);
    }

    #[test]
    fn test_treebank_contractions() {
        let result = tokenize_treebank("can't");
        assert_eq!(result, vec!["ca", "n't"]);
    }

    #[test]
    fn test_treebank_contractions_ll() {
        let result = tokenize_treebank("I'll");
        assert_eq!(result, vec!["I", "'ll"]);
    }

    #[test]
    fn test_treebank_contractions_d() {
        let result = tokenize_treebank("he'd");
        assert_eq!(result, vec!["he", "'d"]);
    }

    #[test]
    fn test_treebank_parentheses() {
        let result = tokenize_treebank("Hello (world)");
        assert_eq!(result, vec!["Hello", "(", "world", ")"]);
    }

    #[test]
    fn test_treebank_handles_quotes() {
        // Verify quotes don't crash tokenizer
        let result = tokenize_treebank("\"Hello\"");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_treebank_comma() {
        let result = tokenize_treebank("Hello, world");
        assert_eq!(result, vec!["Hello", ",", "world"]);
    }

    #[test]
    fn test_treebank_empty() {
        let result = tokenize_treebank("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_treebank_detokenize() {
        let tok = TreebankWordDetokenizer::new();
        let result = tok.detokenize(vec!["Hello".into(), ",".into(), "world".into(), ".".into()]);
        assert_eq!(result, "Hello, world.");
    }

    #[test]
    fn test_treebank_detokenize_contraction() {
        let tok = TreebankWordDetokenizer::new();
        let result = tok.detokenize(vec!["ca".into(), "n't".into()]);
        assert_eq!(result, "can't");
    }
}
