//! `RegexpStemmer` — strip suffix matching a pattern.
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;

static STEM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(ing|ed|s|ly|ness|ment|tion|able)$").unwrap());

#[pyclass(name = "RegexpStemmer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RegexpStemmer {
    min_length: usize,
}

#[pymethods]
impl RegexpStemmer {
    #[new]
    #[pyo3(signature = (min_length = 0))]
    fn new(min_length: usize) -> Self {
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
}
