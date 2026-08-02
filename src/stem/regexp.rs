//! Regexp stemmer — strip suffix matching a regex pattern.

use pyo3::prelude::*;
use regex::Regex;
/// `RegexpStemmer` — strip suffix matching a pattern.
use std::sync::LazyLock;

static STEM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(ing|ed|s|ly|ness|ment|tion|able)$").unwrap());

#[pyclass(name = "RegexpStemmer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RegexpStemmer {
    min_length: usize,
}

#[pymethods]
impl RegexpStemmer {
    #[new]
    #[pyo3(signature = (min_length = 0))]
    const fn new(min_length: usize) -> Self {
        Self { min_length }
    }

    fn stem(&self, word: &str) -> String {
        if word.len() <= self.min_length {
            return word.to_string();
        }
        STEM_RE.replace(word, "").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_regexp() {
        let st = RegexpStemmer::new(0);
        assert_eq!(st.stem("cats"), "cat");
    }

    #[test]
    fn test_regexp_empty() {
        let st = RegexpStemmer::new(0);
        assert_eq!(st.stem(""), "");
    }

    #[test]
    fn test_regexp_min_length() {
        let st = RegexpStemmer::new(5);
        assert_eq!(st.stem("cat"), "cat"); // too short
        assert_eq!(st.stem("running"), "runn"); // long enough
    }
}
