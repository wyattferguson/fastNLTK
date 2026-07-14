//! Tokenization — Rust-accelerated tokenizers matching NLTK's API.
//!
//! Implements all tokenizers from nltk.tokenize:
//! simple (Space, Tab, Line, Char), regexp, Treebank, Tweet, Punkt, etc.

pub mod logos_tokenizer;
pub mod mwe;
pub mod punkt;
pub mod regexp;
pub mod sexpr;
pub mod simple;
pub mod texttiling;
pub mod toktok;
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

    // Punkt sentence tokenizer
    m.add_class::<punkt::PunktSentenceTokenizer>()?;

    // S-Expr tokenizer
    m.add_class::<sexpr::SExprTokenizer>()?;

    // TokTok tokenizer
    m.add_class::<toktok::ToktokTokenizer>()?;

    // MWE tokenizer
    m.add_class::<mwe::MWETokenizer>()?;

    // TextTiling tokenizer
    m.add_class::<texttiling::TextTilingTokenizer>()?;

    // Free functions
    m.add_function(wrap_pyfunction!(sent_tokenize_py, m)?)?;
    m.add_function(wrap_pyfunction!(word_tokenize_py, m)?)?;

    // Logos fast tokenizer
    logos_tokenizer::register_module(m)?;

    Ok(())
}

/// sent_tokenize — simple sentence boundary detection.
///
/// Splits on sentence-ending punctuation (. ! ?) followed by space.
/// Full Punkt-trained implementation coming in a later phase.
#[pyfunction(name = "sent_tokenize", signature = (text, language="english"))]
pub fn sent_tokenize_py(py: Python<'_>, text: &str, language: &str) -> PyResult<Vec<String>> {
    if language != "english" {
        // Non-English language support not yet implemented; using English heuristic
        // Future: load Punkt model for requested language
    }
    let result = py.allow_threads(|| {
        let mut sentences = Vec::new();
        let mut start = 0;
        let bytes = text.as_bytes();
        for (i, _) in text.char_indices() {
            if i > 0
                && (bytes[i - 1] == b'.' || bytes[i - 1] == b'!' || bytes[i - 1] == b'?')
                && i + 1 < bytes.len()
                && bytes[i] == b' '
            {
                sentences.push(text[start..i].to_string());
                start = i + 1;
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
pub fn word_tokenize_py(
    py: Python<'_>,
    text: &str,
    language: &str,
    preserve_line: bool,
) -> PyResult<Vec<String>> {
    if language != "english" {
        // Non-English language support not yet implemented; using English Treebank rules
        // Future: load language-specific tokenizer models
    }
    let result = py.allow_threads(|| {
        if preserve_line {
            text.lines().flat_map(treebank::tokenize_treebank).collect()
        } else {
            treebank::tokenize_treebank(text)
        }
    });
    Ok(result)
}
