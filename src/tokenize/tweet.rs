//! Tweet tokenizer — handles emoji, hashtags, mentions, URLs, etc.
//! Regexes compiled once via LazyLock (not per-call).

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

#[allow(dead_code)]
static _URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"https?://[^\s<>\[\]{}|\\^`]+|www\.[^\s<>\[\]{}|\\^`]+").unwrap()
});
static _EMOTICON_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\][:;=8][<>]?").unwrap()
});
static _PHONE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:(?:\+?1[ \.-]?)?\(?\d{3}\)?[ \.-]?\d{3}[ \.-]?\d{4})").unwrap()
});
static MAIN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"https?://[^\s<>\[\]{}|\\^`]+|www\.[^\s<>\[\]{}|\\^`]+|[\#@]?\w+(?:'\w+)?|[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\][:;=8][<>]?|\S").unwrap()
});

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
    const fn new(preserve_case: bool, reduce_len: bool, strip_handles: bool) -> Self {
        Self { preserve_case, reduce_len, strip_handles }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for m in MAIN_RE.find_iter(text) {
            let token = m.as_str();

            // Handle @mentions
            if self.strip_handles && token.starts_with('@') && token.len() > 1 {
                continue;
            }

            if token.is_empty() {
                continue;
            }

            let out = if self.preserve_case {
                if self.reduce_len {
                    reduce_repeated(token)
                } else {
                    token.to_string()
                }
            } else if self.reduce_len {
                reduce_repeated(&token.to_lowercase())
            } else {
                token.to_lowercase()
            };

            tokens.push(out);
        }
        tokens
    }
}

/// Reduce repeated characters (e.g., "hellooooo" -> "hellooo")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tweet_basic() {
        let tok = TweetTokenizer::new(true, false, false);
        assert_eq!(tok.tokenize("Hello world!"), vec!["Hello", "world", "!"]);
    }

    #[test]
    fn test_tweet_hashtag() {
        let tok = TweetTokenizer::new(true, false, false);
        assert_eq!(tok.tokenize("#NLP is fun"), vec!["#NLP", "is", "fun"]);
    }

    #[test]
    fn test_tweet_mention() {
        let tok = TweetTokenizer::new(true, false, false);
        assert_eq!(tok.tokenize("@user hello"), vec!["@user", "hello"]);
    }

    #[test]
    fn test_tweet_mention_stripped() {
        let tok = TweetTokenizer::new(true, false, true);
        assert_eq!(tok.tokenize("@user hello"), vec!["hello"]);
    }

    #[test]
    fn test_tweet_url() {
        let tok = TweetTokenizer::new(true, false, false);
        assert_eq!(tok.tokenize("Check https://example.com"), vec!["Check", "https://example.com"]);
    }

    #[test]
    fn test_tweet_emoticon() {
        let tok = TweetTokenizer::new(true, false, false);
        assert!(tok.tokenize("Hello :)").contains(&":)".to_string()));
    }

    #[test]
    fn test_reduce_repeated() {
        assert_eq!(reduce_repeated("hellooooo"), "hellooo");
        assert_eq!(reduce_repeated("yessss"), "yesss");
        assert_eq!(reduce_repeated("no"), "no");
        assert_eq!(reduce_repeated("okkkkay"), "okkkay");
    }
}
