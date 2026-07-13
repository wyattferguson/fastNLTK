//! Tokenization — Rust-accelerated tokenizers matching NLTK's API.
//!
//! Implements all tokenizers from nltk.tokenize:
//! simple (Space, Tab, Line, Char), regexp, Treebank, Tweet, Punkt, etc.

pub mod simple;
pub mod regexp;
pub mod treebank;
pub mod tweet;

use pyo3::prelude::*;

/// Register all tokenizer classes and functions with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Simple tokenizers
    m.add_class::<simple::SpaceTokenizer>()?;
    m.add_class::<simple::TabTokenizer>()?;
    m.add_class::<simple::LineTokenizer>()?;
    m.add_class::<simple::CharTokenizer>()?;

    // Regexp tokenizers
    m.add_class::<regexp::RegexpTokenizer>()?;
    m.add_class::<regexp::WhitespaceTokenizer>()?;
    m.add_class::<regexp::WordPunctTokenizer>()?;
    m.add_class::<regexp::BlanklineTokenizer>()?;

    // Treebank tokenizer
    m.add_class::<treebank::TreebankWordTokenizer>()?;
    m.add_class::<treebank::TreebankWordDetokenizer>()?;

    // Tweet tokenizer
    m.add_class::<tweet::TweetTokenizer>()?;

    // Free functions
    m.add_function(wrap_pyfunction!(sent_tokenize_py, m)?)?;
    m.add_function(wrap_pyfunction!(word_tokenize_py, m)?)?;

    Ok(())
}

/// sent_tokenize — simple sentence boundary detection.
///
/// Splits on sentence-ending punctuation (. ! ?) followed by space.
/// Full Punkt-trained implementation coming in a later phase.
#[pyfunction(name = "sent_tokenize", signature = (text, language="english"))]
#[allow(unused_variables)]
pub fn sent_tokenize_py(py: Python<'_>, text: &str, language: &str) -> PyResult<Vec<String>> {
    let result = py.allow_threads(|| {
        let mut sentences = Vec::new();
        let mut start = 0;
        let bytes = text.as_bytes();
        for (i, _) in text.char_indices() {
            if i > 0
                && (bytes[i - 1] == b'.' || bytes[i - 1] == b'!' || bytes[i - 1] == b'?')
            {
                if i + 1 < bytes.len() && bytes[i] == b' ' {
                    sentences.push(text[start..i].to_string());
                    start = i + 1;
                }
            }
        }
        if start < text.len() {
            sentences.push(text[start..].to_string());
        }
        sentences
    });
    Ok(result)
}

/// word_tokenize — word tokenization using Treebank rules.
#[pyfunction(name = "word_tokenize", signature = (text, language="english", preserve_line=false))]
#[allow(unused_variables)]
pub fn word_tokenize_py(
    py: Python<'_>,
    text: &str,
    language: &str,
    preserve_line: bool,
) -> PyResult<Vec<String>> {
    let result = py.allow_threads(|| {
        if preserve_line {
            // Tokenize each line separately
            text.lines()
                .flat_map(|line| treebank::tokenize_treebank(line))
                .collect()
        } else {
            treebank::tokenize_treebank(text)
        }
    });
    Ok(result)
}
