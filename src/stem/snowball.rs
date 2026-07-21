//! Snowball stemmer — wraps `rust-stemmers` crate.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};

/// Map language name to rust-stemmers Algorithm.
fn lang_to_algorithm(lang: &str) -> Option<Algorithm> {
    match lang.to_lowercase().as_str() {
        "danish" | "da" => Some(Algorithm::Danish),
        "dutch" | "nl" => Some(Algorithm::Dutch),
        "english" | "en" => Some(Algorithm::English),
        "finnish" | "fi" => Some(Algorithm::Finnish),
        "french" | "fr" => Some(Algorithm::French),
        "german" | "de" => Some(Algorithm::German),
        "hungarian" | "hu" => Some(Algorithm::Hungarian),
        "italian" | "it" => Some(Algorithm::Italian),
        "norwegian" | "no" => Some(Algorithm::Norwegian),
        "portuguese" | "pt" => Some(Algorithm::Portuguese),
        "romanian" | "ro" => Some(Algorithm::Romanian),
        "russian" | "ru" => Some(Algorithm::Russian),
        "spanish" | "es" => Some(Algorithm::Spanish),
        "swedish" | "sv" => Some(Algorithm::Swedish),
        "turkish" | "tr" => Some(Algorithm::Turkish),
        "arabic" | "ar" => Some(Algorithm::Arabic),
        _ => None,
    }
}

/// `SnowballStemmer` — wraps rust-stemmers for all 16 languages.
#[pyclass(name = "SnowballStemmer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct SnowballStemmer {
    algorithm: Algorithm,
}

#[pymethods]
impl SnowballStemmer {
    #[new]
    #[pyo3(signature = (language="english"))]
    fn new(language: &str) -> PyResult<Self> {
        let algorithm = lang_to_algorithm(language).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown language for SnowballStemmer: {language}. \
                 Supported languages: danish, dutch, english, finnish, french, \
                 german, hungarian, italian, norwegian, portuguese, romanian, \
                 russian, spanish, swedish, turkish, arabic"
            ))
        })?;
        Ok(Self { algorithm })
    }

    /// Stem a single word.
    fn stem(&self, word: &str) -> String {
        let stemmer = Stemmer::create(self.algorithm);
        stemmer.stem(word).to_string()
    }

    /// Stem multiple words in batch (faster than calling `stem()` in a loop).
    fn stem_many(&self, words: Vec<String>) -> Vec<String> {
        let stemmer = Stemmer::create(self.algorithm);
        words.iter().map(|w| stemmer.stem(w).to_string()).collect()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snowball_english() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        assert_eq!(stemmer.stem("running"), "run");
        assert_eq!(stemmer.stem("runner"), "runner");
        assert_eq!(stemmer.stem("ran"), "ran");
        assert_eq!(stemmer.stem("cats"), "cat");
    }

    #[test]
    fn test_snowball_dutch() {
        let stemmer = SnowballStemmer::new("dutch").unwrap();
        let result = stemmer.stem("lopen");
        // rust-stemmers Dutch algorithm produces "lop" (valid stem)
        assert_eq!(result, "lop");
    }

    #[test]
    fn test_snowball_german() {
        let stemmer = SnowballStemmer::new("german").unwrap();
        assert_eq!(stemmer.stem("laufen"), "lauf");
    }

    #[test]
    fn test_snowball_french() {
        let stemmer = SnowballStemmer::new("french").unwrap();
        assert_eq!(stemmer.stem("courir"), "cour");
    }

    #[test]
    fn test_snowball_unknown_language() {
        let result = SnowballStemmer::new("klingon");
        assert!(result.is_err());
    }

    #[test]
    fn test_snowball_language_code() {
        let stemmer = SnowballStemmer::new("en").unwrap();
        assert_eq!(stemmer.stem("running"), "run");
    }

    #[test]
    fn test_snowball_stem_many() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        let words = vec!["running".to_string(), "cats".to_string(), "better".to_string()];
        let result = stemmer.stem_many(words);
        assert_eq!(result, vec!["run", "cat", "better"]);
    }
}
