//! `TextCat` — language detection via `whatlang` crate.
//!
//! Bridges `whatlang::detect()` to NLTK's `TextCat` API.
//! 10-50x faster than NLTK's pure-Python ngram-based `TextCat`.

use pyo3::prelude::*;

/// Language name mapping: `whatlang::Lang` → NLTK-style name (lowercase).
fn lang_name(lang: whatlang::Lang) -> &'static str {
    match lang {
        whatlang::Lang::Eng => "english",
        whatlang::Lang::Fra => "french",
        whatlang::Lang::Deu => "german",
        whatlang::Lang::Spa => "spanish",
        whatlang::Lang::Ita => "italian",
        whatlang::Lang::Por => "portuguese",
        whatlang::Lang::Nld => "dutch",
        whatlang::Lang::Rus => "russian",
        whatlang::Lang::Jpn => "japanese",
        whatlang::Lang::Ara => "arabic",
        whatlang::Lang::Hin => "hindi",
        whatlang::Lang::Kor => "korean",
        whatlang::Lang::Tur => "turkish",
        whatlang::Lang::Pol => "polish",
        whatlang::Lang::Swe => "swedish",
        whatlang::Lang::Dan => "danish",
        whatlang::Lang::Fin => "finnish",
        whatlang::Lang::Tha => "thai",
        whatlang::Lang::Vie => "vietnamese",
        whatlang::Lang::Ind => "indonesian",
        whatlang::Lang::Ces => "czech",
        whatlang::Lang::Hun => "hungarian",
        whatlang::Lang::Ron => "romanian",
        whatlang::Lang::Ukr => "ukrainian",
        whatlang::Lang::Ell => "greek",
        whatlang::Lang::Heb => "hebrew",
        whatlang::Lang::Ben => "bengali",
        whatlang::Lang::Tam => "tamil",
        whatlang::Lang::Urd => "urdu",
        whatlang::Lang::Cat => "catalan",
        whatlang::Lang::Slv => "slovenian",
        whatlang::Lang::Lit => "lithuanian",
        whatlang::Lang::Lav => "latvian",
        whatlang::Lang::Mkd => "macedonian",
        _ => "unknown",
    }
}

// TextCat — Python-facing class

#[pyclass(name = "TextCat", module = "fastnltk._rust")]
pub struct TextCat;

#[pymethods]
impl TextCat {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Guess the language of a text string.
    /// Returns the language name (lowercase, matching NLTK's style).
    #[pyo3(signature = (text))]
    fn guess_language(&self, text: &str) -> Option<String> {
        whatlang::detect_lang(text).map(lang_name).map(String::from)
    }

    /// Guess the language with confidence score.
    /// Returns (`language_name`, confidence) tuple.
    #[pyo3(signature = (text))]
    fn guess_language_scores(&self, text: &str) -> Option<(String, f64)> {
        let info = whatlang::detect(text)?;
        Some((lang_name(info.lang()).to_string(), info.confidence()))
    }

    /// List supported languages.
    #[staticmethod]
    fn supported_languages() -> Vec<&'static str> {
        vec![
            "english",
            "french",
            "german",
            "spanish",
            "italian",
            "portuguese",
            "dutch",
            "russian",
            "japanese",
            "arabic",
            "hindi",
            "korean",
            "turkish",
            "polish",
            "swedish",
            "danish",
            "finnish",
            "thai",
            "vietnamese",
            "indonesian",
            "czech",
            "hungarian",
            "romanian",
            "ukrainian",
            "greek",
            "hebrew",
            "bengali",
            "tamil",
            "urdu",
            "catalan",
            "slovenian",
            "lithuanian",
            "latvian",
            "macedonian",
        ]
    }
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TextCat>()?;
    Ok(())
}
