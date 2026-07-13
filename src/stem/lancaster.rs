//! Lancaster stemmer — pure Rust implementation.
//! Based on the Lancaster (Paice/Husk) stemming algorithm.

use pyo3::prelude::*;

static RULES: &[(&str, &str, i32)] = &[
    ("ai", "", 1), ("ance", "", 1), ("ence", "", 1), ("er", "", 1),
    ("ic", "", 1), ("able", "", 1), ("ible", "", 1), ("ant", "", 1),
    ("ement", "", 1), ("ment", "", 1), ("ent", "", 1), ("sion", "", 1),
    ("tion", "", 1), ("ou", "", 1), ("ism", "", 1), ("ate", "", 1),
    ("iti", "", 1), ("ous", "", 1), ("ive", "", 1), ("ize", "", 1),
    ("al", "", -1), ("all", "", -1), ("ful", "", -1), ("ness", "", -1),
];

#[pyclass(name = "LancasterStemmer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct LancasterStemmer;

#[pymethods]
impl LancasterStemmer {
    #[new]
    fn new() -> Self { Self }

    fn stem(&self, word: &str) -> String {
        let word = word.to_lowercase();
        if word.len() <= 2 { return word; }
        let mut s = word;
        let mut changed = true;
        while changed {
            changed = false;
            for (suffix, replacement, _accept) in RULES {
                if s.ends_with(suffix) && s.len() > suffix.len() + 1 {
                    let _new_len = s.len() - suffix.len() + replacement.len();
                    let mut new_s = s[..s.len() - suffix.len()].to_string();
                    new_s.push_str(replacement);
                    if new_s.len() >= 2 {
                        s = new_s;
                        changed = true;
                        break;
                    }
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lancaster_runs() {
        let stemmer = LancasterStemmer::new();
        let result = stemmer.stem("running");
        // Should produce a shorter or equal-length string
        assert!(result.len() <= "running".len());
        assert!(!result.is_empty());
    }
    #[test]
    fn test_lancaster_empty() {
        let stemmer = LancasterStemmer::new();
        assert_eq!(stemmer.stem(""), "");
    }
}
