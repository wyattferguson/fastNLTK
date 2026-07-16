//! `WordNet` lemmatizer — morphy algorithm in Rust.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pyo3::prelude::*;

// Morphy data: loaded once from nltk_data

struct WordNetData {
    /// Exception lists per POS: pos → inflected → base form
    exceptions: HashMap<String, HashMap<String, String>>,
    /// All known word forms per POS
    known_words: HashMap<String, HashSet<String>>,
}

impl WordNetData {
    fn load(data_dir: &Path) -> Result<Self, String> {
        let mut exceptions: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut known_words: HashMap<String, HashSet<String>> = HashMap::new();

        // POS files mapping
        let pos_files = &[("n", "noun"), ("v", "verb"), ("a", "adj"), ("r", "adv")];

        for (pos, prefix) in pos_files {
            // Load exception file: {prefix}.exc
            let exc_path = data_dir.join(format!("{prefix}.exc"));
            if let Ok(content) = std::fs::read_to_string(&exc_path) {
                let mut exc_map = HashMap::new();
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let inflected = parts[0].to_lowercase();
                        // First base form is the primary lemma
                        let base = parts[1].to_lowercase();
                        exc_map.insert(inflected, base);
                    }
                }
                if !exc_map.is_empty() {
                    exceptions.insert((*pos).to_string(), exc_map);
                }
            }

            // Load index file: index.{prefix}
            let idx_path = data_dir.join(format!("index.{prefix}"));
            if let Ok(content) = std::fs::read_to_string(&idx_path) {
                let mut words = HashSet::new();
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with(' ') {
                        continue;
                    }
                    if let Some(word) = line.split_whitespace().next() {
                        words.insert(word.to_lowercase());
                    }
                }
                known_words.insert((*pos).to_string(), words);
            }
        }

        Ok(Self { exceptions, known_words })
    }

    fn exists(&self, word: &str, pos: &str) -> bool {
        self.known_words.get(pos).map_or(false, |words| words.contains(&word.to_lowercase()))
    }

    fn lookup_exception(&self, word: &str, pos: &str) -> Option<&str> {
        self.exceptions
            .get(pos)
            .and_then(|m| m.get(&word.to_lowercase()))
            .map(std::string::String::as_str)
    }
}

// Morphy algorithm

/// Morphological substitution rules per POS.
/// Each entry: (suffix, replacement) — try replacing suffix with replacement, check existence.
static MORPHY_RULES: &[(&str, &[(&str, &str)])] = &[
    (
        "n",
        &[
            ("s", ""),
            ("ses", "x"),
            ("xes", "x"),
            ("zes", "z"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("ies", "y"),
        ],
    ),
    (
        "v",
        &[
            ("s", ""),
            ("ies", "y"),
            ("es", "e"),
            ("es", ""),
            ("ed", "e"),
            ("ed", ""),
            ("ing", "e"),
            ("ing", ""),
        ],
    ),
    ("a", &[("er", ""), ("est", ""), ("er", "e"), ("est", "e")]),
];

/// Apply morphy rules to find the base form of a word.
fn morphy(data: &WordNetData, word: &str, pos: &str) -> Option<String> {
    let word_lower = word.to_lowercase();

    // 1. Check exceptions for the given POS
    if let Some(base) = data.lookup_exception(&word_lower, pos) {
        return Some(base.to_string());
    }

    // 2. Apply substitution rules for the given POS
    for (rule_pos, rules) in MORPHY_RULES {
        if *rule_pos != pos {
            continue;
        }
        for (suffix, replacement) in *rules {
            if word_lower.ends_with(suffix) && word_lower.len() > suffix.len() {
                let stem = &word_lower[..word_lower.len() - suffix.len()];
                let candidate = format!("{stem}{replacement}");
                if data.exists(&candidate, pos) {
                    return Some(candidate);
                }
                // Handle doubled-consonant reduction: if candidate ends with
                // a repeated character (e.g. "runn" -> "run"), try the
                // single-consonant form.
                if let Some(prev) = candidate.as_bytes().last().copied() {
                    let len = candidate.len();
                    if len >= 2
                        && candidate.as_bytes()[len - 2..] == [prev, prev]
                        && data.exists(&candidate[..len - 1], pos)
                    {
                        return Some(candidate[..len - 1].to_string());
                    }
                }
            }
        }
    }

    // 3. Check if word itself is known in WordNet for this POS.
    //    This must come AFTER rule application so inflected forms
    //    (e.g. "running" is in index.noun but should be rule-matched for pos=v).
    if data.exists(&word_lower, pos) {
        return Some(word_lower);
    }

    None
}

// WordNetLemmatizer — Python-facing class

#[pyclass(name = "WordNetLemmatizer", module = "fastnltk._rust")]
pub struct WordNetLemmatizer {
    data: Option<WordNetData>,
}

#[pymethods]
impl WordNetLemmatizer {
    #[new]
    fn new() -> Self {
        let data = Self::load_wordnet_data();
        Self { data }
    }

    /// Lemmatize a word using the morphy algorithm.
    #[pyo3(signature = (word, pos="n"))]
    fn lemmatize(&self, word: &str, pos: &str) -> String {
        match &self.data {
            Some(data) => morphy(data, word, pos).unwrap_or_else(|| word.to_string()),
            None => word.to_string(),
        }
    }
}

impl WordNetLemmatizer {
    fn load_wordnet_data() -> Option<WordNetData> {
        let search_paths = [
            std::env::var("NLTK_DATA").ok().map(|p| Path::new(&p).join("corpora").join("wordnet")),
            std::env::var("HOME")
                .ok()
                .map(|p| Path::new(&p).join("nltk_data").join("corpora").join("wordnet")),
            std::env::var("USERPROFILE")
                .ok()
                .map(|p| Path::new(&p).join("nltk_data").join("corpora").join("wordnet")),
            std::env::var("APPDATA")
                .ok()
                .map(|p| Path::new(&p).join("nltk_data").join("corpora").join("wordnet")),
        ];

        for path in search_paths.iter().flatten() {
            if path.join("index.noun").exists() {
                return WordNetData::load(path).ok();
            }
        }
        None
    }
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WordNetLemmatizer>()?;
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> WordNetData {
        let mut exceptions = HashMap::new();
        let mut noun_exc = HashMap::new();
        noun_exc.insert("geese".to_string(), "goose".to_string());
        noun_exc.insert("ran".to_string(), "run".to_string());
        exceptions.insert("n".to_string(), noun_exc);

        let mut known_words = HashMap::new();
        let mut nouns = HashSet::new();
        nouns.insert("goose".to_string());
        nouns.insert("dog".to_string());
        nouns.insert("man".to_string());
        nouns.insert("men".to_string());
        nouns.insert("running".to_string()); // running is in noun index
        known_words.insert("n".to_string(), nouns);

        let mut verbs = HashSet::new();
        verbs.insert("run".to_string());
        verbs.insert("walk".to_string());
        verbs.insert("take".to_string());
        known_words.insert("v".to_string(), verbs);

        WordNetData { exceptions, known_words }
    }

    #[test]
    fn test_morphy_noun_exists() {
        let data = test_data();
        assert_eq!(morphy(&data, "dog", "n"), Some("dog".to_string()));
    }

    #[test]
    fn test_morphy_noun_plural() {
        let data = test_data();
        assert_eq!(morphy(&data, "dogs", "n"), Some("dog".to_string()));
    }

    #[test]
    fn test_morphy_noun_exception() {
        let data = test_data();
        assert_eq!(morphy(&data, "geese", "n"), Some("goose".to_string()));
    }

    #[test]
    fn test_morphy_verb_ing() {
        let data = test_data();
        assert_eq!(morphy(&data, "running", "v"), Some("run".to_string()));
    }

    #[test]
    fn test_morphy_verb_ed() {
        let data = test_data();
        assert_eq!(morphy(&data, "walked", "v"), Some("walk".to_string()));
    }

    #[test]
    fn test_morphy_unknown() {
        let data = test_data();
        assert_eq!(morphy(&data, "xyzzy", "n"), None);
    }
}
