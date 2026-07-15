//! CCG Lexicon — maps words to category lists.

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use smol_str::SmolStr;

use crate::ccg::{parse_category, Category};

/// A CCG lexicon mapping words to their possible categories.
///
/// Each word is associated with a list of syntactic categories.
/// Words can be looked up during chart parsing to determine
/// what categories are available at each position.
#[pyclass(name = "CCGLexicon", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct CCGLexicon {
    entries: HashMap<SmolStr, Vec<Category>>,
}

#[pymethods]
impl CCGLexicon {
    #[new]
    #[pyo3(signature = (entries=None))]
    pub fn new(entries: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let mut map: HashMap<SmolStr, Vec<Category>> = HashMap::new();
        if let Some(entries) = entries {
            for (word, cat_str) in entries {
                let cat = parse_category(&cat_str)
                    .ok_or_else(|| PyValueError::new_err(format!("Invalid category: {cat_str}")))?;
                map.entry(SmolStr::new(&word)).or_default().push(cat);
            }
        }
        Ok(Self { entries: map })
    }

    /// Look up categories for a word. Returns empty vec if unknown.
    ///
    /// This is the public Python API. Returns cloned Categories
    /// so callers can own them independently of the lexicon.
    pub fn lookup(&self, word: &str) -> Vec<Category> {
        self.entries.get(word).cloned().unwrap_or_default()
    }

    /// Add a word-category pair. If the word already has this category,
    /// it is appended (duplicates allowed).
    pub fn add(&mut self, word: &str, cat_str: &str) -> PyResult<()> {
        let cat = parse_category(cat_str)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid category: {cat_str}")))?;
        self.entries.entry(SmolStr::new(word)).or_default().push(cat);
        Ok(())
    }

    /// Number of unique words in the lexicon (not total category entries).
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// All entries as sorted (word, category strings) list for debugging / inspection.
    fn entries(&self) -> Vec<(String, Vec<String>)> {
        let mut result: Vec<(String, Vec<String>)> = self
            .entries
            .iter()
            .map(|(w, cats)| {
                (w.to_string(), cats.iter().map(std::string::ToString::to_string).collect())
            })
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl CCGLexicon {
    pub(crate) fn lookup_cats(&self, word: &str) -> &[Category] {
        self.entries.get(word).map_or(&[], std::vec::Vec::as_slice)
    }

    #[allow(dead_code)]
    pub(crate) fn has_word(&self, word: &str) -> bool {
        self.entries.contains_key(word)
    }

    #[allow(dead_code)]
    pub(crate) fn categories(&self) -> &HashMap<SmolStr, Vec<Category>> {
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
        ]))
        .unwrap()
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
        // Empty string should fail since no category parsed
        let result = CCGLexicon::new(Some(vec![("x".into(), String::new())]));
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_lexicon() {
        let lex = CCGLexicon::new(None).unwrap();
        assert_eq!(lex.__len__(), 0);
        assert!(lex.lookup("any").is_empty());
    }

    #[test]
    fn test_lookup_after_add() {
        let mut lex = test_lexicon();
        lex.add("ran", "S\\NP").unwrap();
        let cats = lex.lookup("ran");
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0].to_string(), "S\\NP");
    }

    #[test]
    fn test_has_word() {
        let lex = test_lexicon();
        assert!(lex.has_word("the"));
        assert!(!lex.has_word("unknown"));
    }

    #[test]
    fn test_lookup_cats_pubcrate() {
        let lex = test_lexicon();
        let cats = lex.lookup_cats("the");
        assert_eq!(cats.len(), 1);
        let cats = lex.lookup_cats("unknown");
        assert!(cats.is_empty());
    }

    #[test]
    fn test_entries_output() {
        let lex = test_lexicon();
        let entries = lex.entries();
        assert_eq!(entries.len(), 4);
        // Sorted alphabetically
        for pair in &entries {
            assert!(!pair.0.is_empty(), "word should not be empty");
            assert!(!pair.1.is_empty(), "categories should not be empty");
        }
    }

    #[test]
    fn test_duplicate_category_same_word() {
        let mut lex = test_lexicon();
        lex.add("the", "NP/N").unwrap(); // same category again
        let cats = lex.lookup("the");
        assert_eq!(cats.len(), 2, "duplicate categories should be appended");
    }
}
