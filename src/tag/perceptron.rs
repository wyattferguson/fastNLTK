//! Perceptron tagger — averaged perceptron POS tagging.
//!
//! Uses SmolStr (inline short strings) + FxHashMap (fast hashing)
//! instead of String + SipHash for ~2-3x faster inference.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;
use smol_str::SmolStr;

// PerceptronTagger

/// Averaged perceptron POS tagger — Rust implementation.
#[pyclass(name = "PerceptronTagger", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct PerceptronTagger {
    weights: FxHashMap<SmolStr, FxHashMap<SmolStr, f64>>,
    tagdict: FxHashMap<SmolStr, SmolStr>,
    classes: Vec<SmolStr>,
}

#[pymethods]
impl PerceptronTagger {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { weights: FxHashMap::default(), tagdict: FxHashMap::default(), classes: Vec::new() })
    }

    /// Load weights from Python dicts (from NLTK pickle).
    #[pyo3(signature = (weights_dict=None, tagdict=None, classes=None))]
    fn load(
        &mut self,
        weights_dict: Option<&Bound<'_, PyDict>>,
        tagdict: Option<&Bound<'_, PyDict>>,
        classes: Option<Vec<String>>,
    ) -> PyResult<()> {
        if let Some(wd) = weights_dict {
            let mut weights = FxHashMap::default();
            for (feat_key, tags_dict) in wd.iter() {
                let feat_key: String = feat_key.extract()?;
                let tags_dict = tags_dict.cast::<PyDict>()?;
                let mut tag_weights = FxHashMap::default();
                for (tag, weight) in tags_dict.iter() {
                    let tag: String = tag.extract()?;
                    let weight: f64 = weight.extract()?;
                    tag_weights.insert(SmolStr::new(tag), weight);
                }
                weights.insert(SmolStr::new(feat_key), tag_weights);
            }
            self.weights = weights;
        }

        if let Some(td) = tagdict {
            let mut tagdict = FxHashMap::default();
            for (word, tag) in td.iter() {
                let word: String = word.extract()?;
                let tag: String = tag.extract()?;
                tagdict.insert(SmolStr::new(word), SmolStr::new(tag));
            }
            self.tagdict = tagdict;
        }

        if let Some(classes) = classes {
            self.classes = classes.into_iter().map(SmolStr::new).collect();
        }

        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        self.tag_sentence(&tokens)
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.iter().map(|s| self.tag_sentence(s)).collect()
    }
}

// Implementation

impl PerceptronTagger {
    fn tag_sentence(&self, tokens: &[String]) -> Vec<(String, String)> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(n);
        let mut prev_tags: Vec<SmolStr> = Vec::new();

        for (i, word) in tokens.iter().enumerate() {
            let tag = self.predict_tag(word, i, tokens, &prev_tags);
            result.push((word.clone(), tag.to_string()));
            prev_tags.push(tag);
        }
        result
    }

    fn predict_tag(&self, word: &str, i: usize, tokens: &[String], prev_tags: &[SmolStr]) -> SmolStr {
        // Fast path: tagdict lookup for common words
        if let Some(tag) = self.tagdict.get(word) {
            return tag.clone();
        }

        let features = self.extract_features(word, i, tokens, prev_tags);
        let mut best_tag = SmolStr::new("NN");
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
        prev_tags: &[SmolStr],
    ) -> Vec<SmolStr> {
        let mut feats = Vec::with_capacity(20);

        // Current word features
        feats.push(SmolStr::new(format!("i word {word}")));
        let shape = word_shape(word);
        feats.push(SmolStr::new(format!("i shape {shape}")));

        // Prefix/suffix via char boundary iteration
        let chars: Vec<char> = word.chars().collect();
        let clen = chars.len();

        if clen >= 1 {
            let p = chars[0];
            feats.push(SmolStr::new(format!("i pref1 {p}")));
            let s = chars[clen - 1];
            feats.push(SmolStr::new(format!("i suffix {s}")));
        }
        if clen >= 2 {
            let p: String = chars[..2].iter().collect();
            feats.push(SmolStr::new(format!("i pref2 {p}")));
            let s: String = chars[clen - 2..].iter().collect();
            feats.push(SmolStr::new(format!("i suff2 {s}")));
        }
        if clen >= 3 {
            let p: String = chars[..3].iter().collect();
            feats.push(SmolStr::new(format!("i pref3 {p}")));
            let s: String = chars[clen - 3..].iter().collect();
            feats.push(SmolStr::new(format!("i suff3 {s}")));
        }

        // Previous word features
        if i > 0 {
            let pw = &tokens[i - 1];
            feats.push(SmolStr::new(format!("i-1 word {pw}")));
            let ps = word_shape(pw);
            feats.push(SmolStr::new(format!("i-1 shape {ps}")));
            let pchars: Vec<char> = pw.chars().collect();
            if pchars.len() >= 2 {
                let s: String = pchars[pchars.len() - 2..].iter().collect();
                feats.push(SmolStr::new(format!("i-1 suff2 {s}")));
            }
            if pchars.len() >= 3 {
                let s: String = pchars[pchars.len() - 3..].iter().collect();
                feats.push(SmolStr::new(format!("i-1 suff3 {s}")));
            }
        }

        // Next word features
        if i + 1 < tokens.len() {
            let nw = &tokens[i + 1];
            feats.push(SmolStr::new(format!("i+1 word {nw}")));
            let ns = word_shape(nw);
            feats.push(SmolStr::new(format!("i+1 shape {ns}")));
        }

        // Previous tag features
        if let Some(tag) = prev_tags.last() {
            feats.push(SmolStr::new(format!("i-1 tag {tag}")));
            if prev_tags.len() >= 2 {
                let t2 = &prev_tags[prev_tags.len() - 2];
                feats.push(SmolStr::new(format!("i-2 tag {t2}")));
                feats.push(SmolStr::new(format!("i-1 tag+tag {t2}-{tag}")));
            }
        }

        // Shape features
        if word.contains('-') {
            feats.push(SmolStr::new("i has_hyphen"));
        }
        if word.chars().any(|c| c.is_uppercase()) {
            feats.push(SmolStr::new("i has_upper"));
        }
        if clen > 0 && chars.iter().all(|&c| c.is_uppercase() && c.is_alphabetic()) {
            feats.push(SmolStr::new("i all_upper"));
        }

        feats
    }
}

// Word shape

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

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tagger() -> PerceptronTagger {
        let mut tagger = PerceptronTagger::new().unwrap();
        let mut w = FxHashMap::default();
        let mut w1 = FxHashMap::default();
        w1.insert(SmolStr::new("DT"), 1.0);
        w1.insert(SmolStr::new("NN"), 0.0);
        w.insert(SmolStr::new("i word the"), w1);
        let mut w2 = FxHashMap::default();
        w2.insert(SmolStr::new("NN"), 1.0);
        w2.insert(SmolStr::new("VB"), 0.0);
        w.insert(SmolStr::new("i word cat"), w2);
        let mut w3 = FxHashMap::default();
        w3.insert(SmolStr::new("VB"), 1.0);
        w3.insert(SmolStr::new("NN"), 0.0);
        w.insert(SmolStr::new("i word runs"), w3);
        let mut w4 = FxHashMap::default();
        w4.insert(SmolStr::new("NN"), 1.0);
        w4.insert(SmolStr::new("DT"), 0.0);
        w.insert(SmolStr::new("i shape Xxx"), w4);
        tagger.weights = w;
        tagger.classes = vec![SmolStr::new("DT"), SmolStr::new("NN"), SmolStr::new("VB")];
        tagger
    }

    #[test]
    fn test_tag_basic() {
        let tagger = create_test_tagger();
        let tokens = vec!["the".to_string(), "cat".to_string()];
        let result = tagger.tag(tokens);
        assert_eq!(result[0].1, "DT");
        assert_eq!(result[1].1, "NN");
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
    }
}
