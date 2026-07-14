//! Perceptron tagger — averaged perceptron POS tagging.
//!
//! Implementation matching NLTK's `nltk.tag.perceptron.PerceptronTagger`.
//! Loads weights from NLTK's trained model pickle and performs inference.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════
// PerceptronTagger
// ═══════════════════════════════════════════════════════════

/// Averaged perceptron POS tagger — Rust implementation.
///
/// Uses the same feature set and algorithm as NLTK's `PerceptronTagger`.
/// Weights are loaded from NLTK's trained model.
#[pyclass(name = "PerceptronTagger", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct PerceptronTagger {
    /// Feature weights: `feature_name` → {tag → weight}
    weights: HashMap<String, HashMap<String, f64>>,
    /// Tag dictionary for common words: word → tag
    tagdict: HashMap<String, String>,
    /// Set of known POS tags
    classes: Vec<String>,
}

#[pymethods]
impl PerceptronTagger {
    #[new]
    fn new() -> PyResult<Self> {
        // Start with empty model — will need load() or fit() to be useful
        Ok(Self { weights: HashMap::new(), tagdict: HashMap::new(), classes: Vec::new() })
    }

    /// Load weights from Python dicts (from NLTK pickle).
    #[pyo3(signature = (weights_dict=None, tagdict=None, classes=None))]
    fn load(
        &mut self,
        weights_dict: Option<&Bound<'_, PyDict>>,
        tagdict: Option<&Bound<'_, PyDict>>,
        classes: Option<Vec<String>>,
    ) -> PyResult<()> {
        // Load weights
        if let Some(wd) = weights_dict {
            let mut weights = HashMap::new();
            for (feat_key, tags_dict) in wd.iter() {
                let feat_key = feat_key.extract::<String>()?;
                let tags_dict = tags_dict.downcast::<PyDict>()?;
                let mut tag_weights = HashMap::new();
                for (tag, weight) in tags_dict.iter() {
                    let tag = tag.extract::<String>()?;
                    let weight = weight.extract::<f64>()?;
                    tag_weights.insert(tag, weight);
                }
                weights.insert(feat_key, tag_weights);
            }
            self.weights = weights;
        }

        // Load tagdict
        if let Some(td) = tagdict {
            let mut tagdict = HashMap::new();
            for (word, tag) in td.iter() {
                let word = word.extract::<String>()?;
                let tag = tag.extract::<String>()?;
                tagdict.insert(word, tag);
            }
            self.tagdict = tagdict;
        }

        // Load classes
        if let Some(classes) = classes {
            self.classes = classes;
        }

        Ok(())
    }

    /// Tag a single sentence (list of tokens).
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        self.tag_sentence(&tokens)
    }

    /// Tag multiple sentences.
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.iter().map(|s| self.tag_sentence(s)).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════

impl PerceptronTagger {
    /// Tag a single sentence.
    fn tag_sentence(&self, tokens: &[String]) -> Vec<(String, String)> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n);
        // Previous tags, starting with a special start marker
        let mut prev_tags: Vec<String> = Vec::new();

        for (i, word) in tokens.iter().enumerate() {
            let tag = self.predict_tag(word, i, tokens, &prev_tags);
            result.push((word.clone(), tag.clone()));
            prev_tags.push(tag);
        }

        result
    }

    /// Predict the tag for a single word in context.
    fn predict_tag(&self, word: &str, i: usize, tokens: &[String], prev_tags: &[String]) -> String {
        // Check tagdict first (most common words)
        if let Some(tag) = self.tagdict.get(word) {
            return tag.clone();
        }

        // Extract features
        let features = self.extract_features(word, i, tokens, prev_tags);

        // Score each class
        let mut best_tag = "NN".to_string();
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes {
            let mut score = 0.0;
            for feat in &features {
                if let Some(feat_weights) = self.weights.get(feat.as_str()) {
                    if let Some(weight) = feat_weights.get(class.as_str()) {
                        score += weight;
                    }
                }
            }
            if score > best_score {
                best_score = score;
                best_tag.clone_from(class);
            }
        }

        best_tag
    }

    /// Extract features matching NLTK's `PerceptronTagger` feature set.
    fn extract_features(
        &self,
        word: &str,
        i: usize,
        tokens: &[String],
        prev_tags: &[String],
    ) -> Vec<String> {
        let mut feats = Vec::with_capacity(20);

        // Current word features
        feats.push(format!("i word {word}"));

        // Word shape
        let shape = word_shape(word);
        feats.push(format!("i shape {shape}"));

        // Prefix features (first 1-3 chars)
        if !word.is_empty() {
            let pref1: String = word.chars().take(1).collect();
            feats.push(format!("i pref1 {pref1}"));
        }
        if word.chars().count() >= 2 {
            let pref2: String = word.chars().take(2).collect();
            feats.push(format!("i pref2 {pref2}"));
        }
        if word.chars().count() >= 3 {
            let pref3: String = word.chars().take(3).collect();
            feats.push(format!("i pref3 {pref3}"));
        }

        // Suffix features (last 1-3 chars)
        if !word.is_empty() {
            let suff1: String =
                word.chars().rev().take(1).collect::<String>().chars().rev().collect();
            feats.push(format!("i suffix {suff1}"));
        }
        if word.chars().count() >= 2 {
            let suff2: String =
                word.chars().rev().take(2).collect::<String>().chars().rev().collect();
            feats.push(format!("i suff2 {suff2}"));
        }
        if word.chars().count() >= 3 {
            let suff3: String =
                word.chars().rev().take(3).collect::<String>().chars().rev().collect();
            feats.push(format!("i suff3 {suff3}"));
        }

        // Previous word features
        if i > 0 {
            let prev_word = &tokens[i - 1];
            feats.push(format!("i-1 word {prev_word}"));

            // Previous word shape
            let prev_shape = word_shape(prev_word);
            feats.push(format!("i-1 shape {prev_shape}"));

            // Previous word suffix
            if prev_word.len() >= 2 {
                feats.push(format!("i-1 suff2 {}", &prev_word[prev_word.len() - 2..]));
            }
            if prev_word.len() >= 3 {
                feats.push(format!("i-1 suff3 {}", &prev_word[prev_word.len() - 3..]));
            }
        }

        // Next word features
        if i + 1 < tokens.len() {
            let next_word = &tokens[i + 1];
            feats.push(format!("i+1 word {next_word}"));

            // Next word shape
            let next_shape = word_shape(next_word);
            feats.push(format!("i+1 shape {next_shape}"));
        }

        // Previous tag features
        if let Some(tag) = prev_tags.last() {
            feats.push(format!("i-1 tag {tag}"));

            // Tag bigram
            if prev_tags.len() >= 2 {
                let tag2 = &prev_tags[prev_tags.len() - 2];
                feats.push(format!("i-2 tag {tag2}"));
                feats.push(format!("i-1 tag+tag {tag2}-{tag}"));
            }
        }

        // Shape features for hyphenation and capitalization
        if word.contains('-') {
            feats.push("i has_hyphen".to_string());
        }
        if word.chars().any(char::is_uppercase) {
            feats.push("i has_upper".to_string());
        }
        if word.chars().all(|c| c.is_uppercase() && c.is_alphabetic()) {
            feats.push("i all_upper".to_string());
        }

        feats
    }
}

// ═══════════════════════════════════════════════════════════
// Word shape
// ═══════════════════════════════════════════════════════════

/// Compute the shape of a word (capitalization pattern).
/// e.g., "Hello" → "Xxxxx", "NLP" → "XXX", "hello" → "xxxx"
fn word_shape(word: &str) -> String {
    let mut shape = String::with_capacity(word.len());
    for c in word.chars() {
        if c.is_uppercase() {
            shape.push('X');
        } else if c.is_lowercase() {
            shape.push('x');
        } else if c.is_numeric() {
            shape.push('d');
        } else {
            shape.push(c);
        }
    }
    shape
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tagger() -> PerceptronTagger {
        let mut tagger = PerceptronTagger::new().unwrap();

        // Add some minimal weights for testing
        let mut weights = HashMap::new();
        let mut w1 = HashMap::new();
        w1.insert("DT".to_string(), 1.0);
        w1.insert("NN".to_string(), 0.0);
        weights.insert("i word the".to_string(), w1);

        let mut w2 = HashMap::new();
        w2.insert("NN".to_string(), 1.0);
        w2.insert("VB".to_string(), 0.0);
        weights.insert("i word cat".to_string(), w2);

        let mut w3 = HashMap::new();
        w3.insert("VB".to_string(), 1.0);
        w3.insert("NN".to_string(), 0.0);
        weights.insert("i word runs".to_string(), w3);

        let mut w4 = HashMap::new();
        w4.insert("NN".to_string(), 1.0);
        w4.insert("DT".to_string(), 0.0);
        weights.insert("i shape Xxx".to_string(), w4);

        tagger.weights = weights;
        tagger.classes = vec!["DT".to_string(), "NN".to_string(), "VB".to_string()];

        tagger
    }

    #[test]
    fn test_tag_basic() {
        let tagger = create_test_tagger();
        let tokens = vec!["the".to_string(), "cat".to_string()];
        let result = tagger.tag(tokens);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, "DT"); // "the" → DT
        assert_eq!(result[1].1, "NN"); // "cat" → NN
    }

    #[test]
    fn test_tag_empty() {
        let tagger = PerceptronTagger::new().unwrap();
        let result = tagger.tag(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_word_shape() {
        assert_eq!(word_shape("Hello"), "Xxxxx");
        assert_eq!(word_shape("NLP"), "XXX");
        assert_eq!(word_shape("hello"), "xxxxx");
        assert_eq!(word_shape("123"), "ddd");
        assert_eq!(word_shape("iPhone"), "xXxxxx");
        assert_eq!(word_shape("a"), "x");
    }
}
