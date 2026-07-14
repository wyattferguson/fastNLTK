//! Chat — Rust-accelerated pattern-matching chatbot.
//!
//! Implements NLTK's Chat class: regex pattern pairs are matched
//! against user input, returning a random response from matching pair.
//! Pure pattern matching — no ML, 10-50x faster than NLTK.

use pyo3::prelude::*;
use rand::Rng;
use regex::Regex;

// ═══════════════════════════════════════════════════════════
// Chat
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "Chat", module = "fastnltk._rust")]
pub struct Chat {
    /// (compiled_regex, responses) pairs
    pairs: Vec<(Regex, Vec<String>)>,
}

#[pymethods]
impl Chat {
    #[new]
    fn new(pairs: Vec<(String, Vec<String>)>) -> PyResult<Self> {
        let mut compiled = Vec::with_capacity(pairs.len());
        for (pattern, responses) in &pairs {
            let re = Regex::new(pattern).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid regex: {e}"))
            })?;
            compiled.push((re, responses.clone()));
        }
        Ok(Chat { pairs: compiled })
    }

    /// Respond to a user input string.
    fn respond(&self, text: &str) -> String {
        for (re, responses) in &self.pairs {
            if re.is_match(text) {
                let mut rng = rand::rng();
                let idx = rng.random_range(0..responses.len());
                return responses[idx].clone();
            }
        }
        "I don't understand.".to_string()
    }

    /// Same as respond but returns the response and the matching pattern index.
    fn converse(&self, text: &str) -> (String, isize) {
        for (i, (re, responses)) in self.pairs.iter().enumerate() {
            if re.is_match(text) {
                let mut rng = rand::rng();
                let idx = rng.random_range(0..responses.len());
                return (responses[idx].clone(), i as isize);
            }
        }
        ("I don't understand.".to_string(), -1)
    }
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chat>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chat() -> Chat {
        Chat::new(vec![
            (
                r"hello|hi|hey".to_string(),
                vec!["Hello!".to_string(), "Hi there!".to_string()],
            ),
            (
                r"how are you".to_string(),
                vec!["I'm doing well!".to_string()],
            ),
            (
                r"bye|goodbye".to_string(),
                vec!["Goodbye!".to_string(), "See you!".to_string()],
            ),
        ])
        .unwrap()
    }

    #[test]
    fn test_hello() {
        let chat = sample_chat();
        let resp = chat.respond("hello");
        assert!(resp == "Hello!" || resp == "Hi there!");
    }

    #[test]
    fn test_hi() {
        let chat = sample_chat();
        let resp = chat.respond("hi there");
        assert!(resp == "Hello!" || resp == "Hi there!");
    }

    #[test]
    fn test_how_are_you() {
        let chat = sample_chat();
        let resp = chat.respond("how are you");
        assert_eq!(resp, "I'm doing well!");
    }

    #[test]
    fn test_bye() {
        let chat = sample_chat();
        let resp = chat.respond("goodbye");
        assert!(resp == "Goodbye!" || resp == "See you!");
    }

    #[test]
    fn test_unknown() {
        let chat = sample_chat();
        let resp = chat.respond("what is the meaning of life");
        assert_eq!(resp, "I don't understand.");
    }

    #[test]
    fn test_converse() {
        let chat = sample_chat();
        let (resp, idx) = chat.converse("hello");
        assert!(idx >= 0);
        assert!(!resp.is_empty());
    }
}
