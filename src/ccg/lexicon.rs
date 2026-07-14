//! CCG Lexicon — maps words to category lists.
//!
//! NLTK equivalent: nltk.ccg.lexicon

use std::collections::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ccg::{Category, parse_category};

/// A CCG lexicon mapping words to their possible categories.
#[pyclass(name = "CCGLexicon", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct CCGLexicon {
    entries: HashMap<String, Vec<Category>>,
}

#[pymethods]
impl CCGLexicon {
    #[new]
    #[pyo3(signature = (entries=None))]
    pub fn new(entries: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let mut map: HashMap<String, Vec<Category>> = HashMap::new();
        if let Some(entries) = entries {
            for (word, cat_str) in entries {
                let cat = parse_category(&cat_str).ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid category: {cat_str}"))
                })?;
                map.entry(word).or_default().push(cat);
            }
        }
        Ok(CCGLexicon { entries: map })
    }

    /// Look up categories for a word.
    pub fn lookup(&self, word: &str) -> Vec<Category> {
        self.entries.get(word).cloned().unwrap_or_default()
    }

    /// Add a word-category pair.
    pub fn add(&mut self, word: &str, cat_str: &str) -> PyResult<()> {
        let cat = parse_category(cat_str).ok_or_else(|| {
            PyValueError::new_err(format!("Invalid category: {cat_str}"))
        })?;
        self.entries.entry(word.to_string()).or_default().push(cat);
        Ok(())
    }

    /// Number of unique words in the lexicon.
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// All entries as (word, category strings) for debugging.
    fn entries(&self) -> Vec<(String, Vec<String>)> {
        let mut result: Vec<(String, Vec<String>)> = self.entries.iter()
            .map(|(w, cats)| (w.clone(), cats.iter().map(|c| c.to_string()).collect()))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl CCGLexicon {
    pub(crate) fn lookup_cats(&self, word: &str) -> &[Category] {
        self.entries.get(word).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub(crate) fn has_word(&self, word: &str) -> bool {
        self.entries.contains_key(word)
    }

    pub(crate) fn categories(&self) -> &HashMap<String, Vec<Category>> {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lexicon() -> CCGLexicon {
        CCGLexicon::new(Some(vec![
            ("the".into(), "NP/N".into()),
            ("cat".into(), "N".into()),
            ("dog".into(), "N".into()),
            ("chased".into(), "(S\\NP)/NP".into()),
        ])).unwrap()
    }

    #[test]
    fn test_lookup() {
        let lex = test_lexicon();
        let cats = lex.lookup("the");
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0].to_string(), "NP/N");
    }

    #[test]
    fn test_lookup_missing() {
        let lex = test_lexicon();
        let cats = lex.lookup("unknown");
        assert!(cats.is_empty());
    }

    #[test]
    fn test_len() {
        let lex = test_lexicon();
        assert_eq!(lex.__len__(), 4);
    }

    #[test]
    fn test_add() {
        let mut lex = test_lexicon();
        lex.add("saw", "NP/N").unwrap();
        assert_eq!(lex.__len__(), 5);
    }

    #[test]
    fn test_invalid_category() {
        // Empty string should fail
        let result = CCGLexicon::new(Some(vec![("x".into(), "".into())]));
        assert!(result.is_err());
    }
}
