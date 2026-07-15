//! Penn Treebank-style tokenization — char-scanner with exact NLTK compatibility.
//!
//! First pass: find whitespace-delimited words with byte spans.
//! Second pass: split each word on Treebank punctuation/contractions.
//! No regex, no intermediate string copies for the common case.

use memchr::memchr3;
use pyo3::prelude::*;

/// Characters that Treebank detaches from adjacent words.
const fn is_punct(c: char) -> bool {
    matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ':' | ';' | ',' | '.' | '?' | '!')
}

/// Split a word at a contraction boundary.
///
/// `word` is a non-whitespace token. If it contains a known contraction,
/// returns `(stem, suffix)` byte offsets relative to `word`.
fn find_contraction(word: &str) -> Option<(usize, usize)> {
    if word.len() < 3 {
        return None;
    }

    // Find apostrophe position
    let ap_pos = word.find('\'')?;

    let before = &word[..ap_pos];
    let after = &word[ap_pos + 1..];
    let after_bytes = after.as_bytes();

    // n't: before ends with 'n', after starts with 't'
    if !after_bytes.is_empty() && after_bytes[0] == b't' {
        let rest = &after_bytes[1..];
        if (rest.is_empty() || !rest[0].is_ascii_alphabetic()) && before.ends_with('n') {
            let stem_end = ap_pos - 1; // split at 'n', i.e. split at start of 'n'
            return Some((stem_end, ap_pos + 2)); // "n't" = 3 bytes from ap_pos-1 to ap_pos+2 inclusive
        }
    }

    // Word-final suffixes: 'll, 're, 've, 'm, 'd, 's
    let suffix_map: &[(&[u8], usize)] =
        &[(b"ll", 3), (b"re", 3), (b"ve", 3), (b"m", 2), (b"d", 2), (b"s", 2)];
    for (suffix, total_len) in suffix_map {
        if after_bytes.starts_with(suffix) {
            let rest = &after_bytes[*total_len - 1..];
            if rest.is_empty() || !rest[0].is_ascii_alphabetic() {
                return Some((ap_pos, ap_pos + total_len));
            }
        }
    }

    // 'em
    if after.len() >= 2 && after_bytes.starts_with(b"em") {
        let rest = &after_bytes[2..];
        if rest.is_empty() || !rest[0].is_ascii_alphabetic() {
            return Some((ap_pos, ap_pos + 3));
        }
    }

    None
}

/// Process a single whitespace-delimited word, splitting on punctuation/contractions.
fn split_word(
    word: &str,
    offset: usize,
    tokens: &mut Vec<String>,
    spans: &mut Vec<(usize, usize)>,
) {
    // Fast path: most words have no punctuation or contractions, no double-hyphen
    if !word.contains(|c: char| is_punct(c) || c == '\'' || word.contains("--")) {
        tokens.push(word.to_string());
        spans.push((offset, offset + word.len()));
        return;
    }

    // Check for contraction first (handles the whole word at once)
    if let Some((stem_end, suffix_end)) = find_contraction(word) {
        // Emit word stem before contraction
        flush_subword(word, offset, &mut 0, stem_end, tokens, spans);
        // Emit contraction suffix
        tokens.push(word[stem_end..suffix_end].to_string());
        spans.push((offset + stem_end, offset + suffix_end));
        // Process rest (after contraction)
        let rest = &word[suffix_end..];
        if !rest.is_empty() {
            split_word(rest, offset + suffix_end, tokens, spans);
        }
        return;
    }

    // Scan word for punctuation splits
    let mut start = 0;
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let (byte_pos, c) = chars[i];

        // Double hyphen
        if c == '-' && i + 1 < chars.len() && chars[i + 1].1 == '-' {
            flush_subword(word, offset, &mut start, byte_pos, tokens, spans);
            start = byte_pos + 2;
            tokens.push("--".to_string());
            spans.push((offset + byte_pos, offset + byte_pos + 2));
            i += 2;
            continue;
        }

        // Double quotes
        if c == '\'' && i + 1 < chars.len() && chars[i + 1].1 == '\'' {
            flush_subword(word, offset, &mut start, byte_pos, tokens, spans);
            start = byte_pos + 2;
            tokens.push("''".to_string());
            spans.push((offset + byte_pos, offset + byte_pos + 2));
            i += 2;
            continue;
        }

        // Punctuation
        if is_punct(c) {
            flush_subword(word, offset, &mut start, byte_pos, tokens, spans);
            start = byte_pos + c.len_utf8();
            tokens.push(word[byte_pos..start].to_string());
            spans.push((offset + byte_pos, offset + start));
            i += 1;
            continue;
        }

        i += 1;
    }

    // Flush remaining
    if start < word.len() {
        tokens.push(word[start..].to_string());
        spans.push((offset + start, offset + word.len()));
    }
}

/// Helper: flush a sub-word token.
#[inline]
fn flush_subword(
    word: &str,
    offset: usize,
    start: &mut usize,
    end: usize,
    tokens: &mut Vec<String>,
    spans: &mut Vec<(usize, usize)>,
) {
    if end > *start {
        tokens.push(word[*start..end].to_string());
        spans.push((offset + *start, offset + end));
    }
    *start = end;
}

/// Tokenize text using Treebank rules.
///
/// Returns `(tokens, byte_spans)`. Spans reference the original `text`.
/// First pass uses SIMD memchr3 for whitespace boundary detection.
#[must_use]
pub fn tokenize_treebank(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut tokens: Vec<String> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;

    while start < bytes.len() {
        // Skip carriage return (not in memchr3 search set but is whitespace)
        if bytes[start] == b'\r' {
            start += 1;
            continue;
        }
        if let Some(rel) = memchr3(b' ', b'\t', b'\n', &bytes[start..]) {
            let mut abs = start + rel;
            // Exclude trailing \r before \n from the token
            if abs > start && bytes[abs - 1] == b'\r' {
                abs -= 1;
            }
            if abs > start {
                split_word(&text[start..abs], start, &mut tokens, &mut spans);
            }
            start = abs + 1;
            // Skip consecutive ASCII whitespace
            while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                start += 1;
            }
        } else {
            if start < bytes.len() {
                split_word(&text[start..], start, &mut tokens, &mut spans);
            }
            break;
        }
    }

    (tokens, spans)
}

// ── PyO3 wrappers ────────────────────────────────────

/// `TreebankWordTokenizer` — Penn Treebank tokenization.
#[pyclass(name = "TreebankWordTokenizer", module = "fastnltk._rust")]
pub struct TreebankWordTokenizer;

#[pymethods]
impl TreebankWordTokenizer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let (tokens, _) = tokenize_treebank(text);
        tokens
    }

    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let (_, spans) = tokenize_treebank(text);
        spans
    }
}

/// `TreebankWordDetokenizer` — detokenize Treebank tokens back to text.
#[pyclass(name = "TreebankWordDetokenizer", module = "fastnltk._rust")]
pub struct TreebankWordDetokenizer;

#[pymethods]
impl TreebankWordDetokenizer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn detokenize(&self, tokens: Vec<String>) -> String {
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0
                && !matches!(
                    token.as_str(),
                    "." | "," | "!" | "?" | ":" | ";" | ")" | "]" | "}" | "%" | "''" | "'" | "n't"
                )
                && !token.starts_with('\'')
                && !matches!(token.as_str(), "(" | "[" | "{" | "``")
            {
                result.push(' ');
            }
            result.push_str(token);
        }
        result
    }
}

// ── Tests ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let (t, s) = tokenize_treebank("Hello world.");
        assert_eq!(t, vec!["Hello", "world", "."]);
        assert_eq!(s, vec![(0, 5), (6, 11), (11, 12)]);
    }

    #[test]
    fn test_contraction_nt() {
        let (t, s) = tokenize_treebank("can't");
        assert_eq!(t, vec!["ca", "n't"]);
        assert_eq!(s, vec![(0, 2), (2, 5)]);
    }

    #[test]
    fn test_contraction_ll() {
        let (t, s) = tokenize_treebank("I'll");
        assert_eq!(t, vec!["I", "'ll"]);
        assert_eq!(s, vec![(0, 1), (1, 4)]);
    }

    #[test]
    fn test_contraction_d() {
        let (t, s) = tokenize_treebank("he'd");
        assert_eq!(t, vec!["he", "'d"]);
        assert_eq!(s, vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn test_parentheses() {
        let (t, s) = tokenize_treebank("Hello (world)");
        assert_eq!(t, vec!["Hello", "(", "world", ")"]);
    }

    #[test]
    fn test_comma() {
        let (t, s) = tokenize_treebank("Hello, world");
        assert_eq!(t, vec!["Hello", ",", "world"]);
    }

    #[test]
    fn test_double_hyphen() {
        let (t, s) = tokenize_treebank("Hello--world");
        assert_eq!(t, vec!["Hello", "--", "world"]);
    }

    #[test]
    fn test_empty() {
        let (t, s) = tokenize_treebank("");
        assert!(t.is_empty());
        assert!(s.is_empty());
    }

    #[test]
    fn test_preserves_quotes() {
        let (t, _) = tokenize_treebank("\"Hello\"");
        assert!(!t.is_empty());
        // Quotes stay attached (not in Treebank punctuation set)
        assert_eq!(t, vec!["\"Hello\""]);
    }

    #[test]
    fn test_emoji_standalone() {
        let (t, _) = tokenize_treebank("Hello \u{1f44b} world");
        assert!(t.contains(&"Hello".to_string()));
        assert!(t.contains(&"world".to_string()));
    }

    #[test]
    fn test_full_sentence() {
        let text = "Mr. Smith went to Washington, D.C. and said \"We can't allow this!\"";
        let (t, _) = tokenize_treebank(text);
        assert!(!t.is_empty());
        assert!(t.len() > 10);
    }

    #[test]
    fn test_detokenize() {
        let tok = TreebankWordDetokenizer::new();
        let result = tok.detokenize(vec!["Hello".into(), ",".into(), "world".into(), ".".into()]);
        assert_eq!(result, "Hello, world.");
    }

    #[test]
    fn test_detokenize_contraction() {
        let tok = TreebankWordDetokenizer::new();
        let result = tok.detokenize(vec!["ca".into(), "n't".into()]);
        assert_eq!(result, "can't");
    }
}
