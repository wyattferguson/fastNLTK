//! Tweet tokenizer — handles emoji, hashtags, mentions, URLs, etc.
//!
//! Based on NLTK's TweetTokenizer with regex patterns for
//! social media text.

use pyo3::prelude::*;
use regex::Regex;

/// Regex patterns for tweet tokenization (from NLTK's casual.py)
fn build_patterns() -> (Regex, Regex, Regex, Regex) {
    // URL pattern
    let url_re = Regex::new(
        r"https?://[^\s<>\[\]{}|\\^`]+|www\.[^\s<>\[\]{}|\\^`]+"
    ).unwrap();

    // Emoticon pattern
    let emoticon_re = Regex::new(
        r"[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\][:;=8][<>]?"
    ).unwrap();

    // Phone number pattern
    let phone_re = Regex::new(
        r"(?:(?:\+?1[ \.-]?)?\(?\d{3}\)?[ \.-]?\d{3}[ \.-]?\d{4})"
    ).unwrap();

    // Main split pattern: preserve URL, emoji, hashtags, mentions
    let main_re = Regex::new(
        r"https?://[^\s<>\[\]{}|\\^`]+|www\.[^\s<>\[\]{}|\\^`]+|[\#@]?\w+(?:'\w+)?|[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\][:;=8][<>]?|\S"
    ).unwrap();

    (main_re, url_re, emoticon_re, phone_re)
}

// ═══════════════════════════════════════════════════════════
// TweetTokenizer
// ═══════════════════════════════════════════════════════════

/// Tokenizer for Twitter/text messages.
#[pyclass(name = "TweetTokenizer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct TweetTokenizer {
    preserve_case: bool,
    reduce_len: bool,
    strip_handles: bool,
}

#[pymethods]
impl TweetTokenizer {
    #[new]
    #[pyo3(signature = (preserve_case=true, reduce_len=false, strip_handles=false))]
    fn new(preserve_case: bool, reduce_len: bool, strip_handles: bool) -> Self {
        Self {
            preserve_case,
            reduce_len,
            strip_handles,
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let (main_re, _url_re, _emoticon_re, _phone_re) = build_patterns();
        let mut tokens = Vec::new();

        for m in main_re.find_iter(text) {
            let token = m.as_str().to_string();

            // Handle @mentions
            if self.strip_handles && token.starts_with('@') && token.len() > 1 {
                continue;
            }

            // Handle case
            let token = if self.preserve_case {
                token
            } else {
                token.to_lowercase()
            };

            // Handle repeated characters
            let token = if self.reduce_len {
                reduce_repeated(&token)
            } else {
                token
            };

            tokens.push(token);
        }

        tokens
    }
}

/// Reduce repeated characters (e.g., "hellooooo" → "hellooo")
fn reduce_repeated(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 3 {
        return s.to_string();
    }

    let mut result = String::with_capacity(s.len());
    let mut count = 1;

    for i in 0..chars.len() {
        if i > 0 && chars[i] == chars[i - 1] {
            count += 1;
            if count <= 3 {
                result.push(chars[i]);
            }
        } else {
            count = 1;
            result.push(chars[i]);
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tweet_basic() {
        let tok = TweetTokenizer::new(true, false, false);
        let result = tok.tokenize("Hello world!");
        assert_eq!(result, vec!["Hello", "world", "!"]);
    }

    #[test]
    fn test_tweet_hashtag() {
        let tok = TweetTokenizer::new(true, false, false);
        let result = tok.tokenize("#NLP is fun");
        assert_eq!(result, vec!["#NLP", "is", "fun"]);
    }

    #[test]
    fn test_tweet_mention() {
        let tok = TweetTokenizer::new(true, false, false);
        let result = tok.tokenize("@user hello");
        assert_eq!(result, vec!["@user", "hello"]);
    }

    #[test]
    fn test_tweet_mention_stripped() {
        let tok = TweetTokenizer::new(true, false, true);
        let result = tok.tokenize("@user hello");
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_tweet_url() {
        let tok = TweetTokenizer::new(true, false, false);
        let result = tok.tokenize("Check https://example.com");
        assert_eq!(result, vec!["Check", "https://example.com"]);
    }

    #[test]
    fn test_tweet_emoticon() {
        let tok = TweetTokenizer::new(true, false, false);
        let result = tok.tokenize("Hello :)");
        assert!(result.contains(&":)".to_string()));
    }

    #[test]
    fn test_reduce_repeated() {
        assert_eq!(reduce_repeated("hellooooo"), "hellooo");
        assert_eq!(reduce_repeated("yessss"), "yesss");
        assert_eq!(reduce_repeated("no"), "no");
        assert_eq!(reduce_repeated("okkkkay"), "okkkay");
    }
}
