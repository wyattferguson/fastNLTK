//! Perceptron tagger — averaged perceptron POS tagging.
//!
//! Uses u64 feature IDs (`FxHash` of component strings) to eliminate
//! all per-feature String allocation during inference.
//! Fast path: tagdict lookup for common words avoids feature extraction.

use std::hash::Hasher;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;
use smol_str::SmolStr;

/// Hash a 2-component feature (prefix + value).
/// Writes bytes directly via `Hasher::write`, avoiding any String allocation.
/// Consistent with hashing the concatenated string during model load.
#[inline]
fn hash2(a: &str, b: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(a.as_bytes());
    h.write(b.as_bytes());
    h.finish()
}

/// Hash a 3-component feature.
#[inline]
fn hash3(a: &str, b: &str, c: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(a.as_bytes());
    h.write(b.as_bytes());
    h.write(c.as_bytes());
    h.finish()
}

#[pyclass(name = "PerceptronTagger", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct PerceptronTagger {
    /// Feature weights keyed by u64 hash of feature name string.
    weights: FxHashMap<u64, FxHashMap<SmolStr, f64>>,
    /// Tag dictionary for common words: word → tag.
    tagdict: FxHashMap<SmolStr, SmolStr>,
    /// Known POS tag classes.
    classes: Vec<SmolStr>,
}

#[pymethods]
impl PerceptronTagger {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            weights: FxHashMap::default(),
            tagdict: FxHashMap::default(),
            classes: Vec::new(),
        })
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
                // Hash the full key string via write (consistent with hash2)
                let mut h = FxHasher::default();
                h.write(feat_key.as_bytes());
                let feat_id = h.finish();

                let mut tag_weights = FxHashMap::default();
                for (tag, weight) in tags_dict.iter() {
                    let tag: String = tag.extract()?;
                    let weight: f64 = weight.extract()?;
                    tag_weights.insert(SmolStr::new(tag), weight);
                }
                weights.insert(feat_id, tag_weights);
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

impl PerceptronTagger {
    fn tag_sentence(&self, tokens: &[String]) -> Vec<(String, String)> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(tokens.len());
        let mut prev_tags: Vec<SmolStr> = Vec::new();
        let mut feat_ids = Vec::with_capacity(20);

        for (i, word) in tokens.iter().enumerate() {
            let tag = self.predict_tag(word, i, tokens, &prev_tags, &mut feat_ids);
            result.push((word.clone(), tag.to_string()));
            prev_tags.push(tag);
        }
        result
    }

    fn predict_tag(
        &self,
        word: &str,
        i: usize,
        tokens: &[String],
        prev_tags: &[SmolStr],
        feat_ids: &mut Vec<u64>,
    ) -> SmolStr {
        // Fast path: tagdict lookup for common words (e.g. "the" → "DT")
        if let Some(tag) = self.tagdict.get(word) {
            return tag.clone();
        }

        // Collect feature IDs without any String allocation
        feat_ids.clear();
        collect_feature_ids(word, i, tokens, prev_tags, feat_ids);

        // Score each class against all features
        let mut best_tag = SmolStr::new("NN");
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes {
            let mut score = 0.0;
            for &fid in feat_ids.iter() {
                if let Some(feat_weights) = self.weights.get(&fid) {
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
}

/// Collect feature integer IDs for a word in context.
/// No String allocation — hashes feature components directly.
fn collect_feature_ids(
    word: &str,
    i: usize,
    tokens: &[String],
    prev_tags: &[SmolStr],
    out: &mut Vec<u64>,
) {
    out.push(hash2("i word ", word));

    let shape = word_shape(word);
    out.push(hash2("i shape ", &shape));

    let chars: Vec<char> = word.chars().collect();
    let clen = chars.len();

    if clen >= 1 {
        out.push(hash2("i pref1 ", &chars[0].to_string()));
        out.push(hash2("i suffix ", &chars[clen - 1].to_string()));
    }
    if clen >= 2 {
        let p2: String = chars[..2].iter().collect();
        out.push(hash2("i pref2 ", &p2));
        let s2: String = chars[clen - 2..].iter().collect();
        out.push(hash2("i suff2 ", &s2));
    }
    if clen >= 3 {
        let p3: String = chars[..3].iter().collect();
        out.push(hash2("i pref3 ", &p3));
        let s3: String = chars[clen - 3..].iter().collect();
        out.push(hash2("i suff3 ", &s3));
    }

    // Previous word
    if i > 0 {
        let pw = &tokens[i - 1];
        out.push(hash2("i-1 word ", pw));
        let ps = word_shape(pw);
        out.push(hash2("i-1 shape ", &ps));
        let pchars: Vec<char> = pw.chars().collect();
        if pchars.len() >= 2 {
            let s: String = pchars[pchars.len() - 2..].iter().collect();
            out.push(hash2("i-1 suff2 ", &s));
        }
        if pchars.len() >= 3 {
            let s: String = pchars[pchars.len() - 3..].iter().collect();
            out.push(hash2("i-1 suff3 ", &s));
        }
    }

    // Next word
    if i + 1 < tokens.len() {
        let nw = &tokens[i + 1];
        out.push(hash2("i+1 word ", nw));
        let ns = word_shape(nw);
        out.push(hash2("i+1 shape ", &ns));
    }

    // Previous tags
    if let Some(tag) = prev_tags.last() {
        out.push(hash2("i-1 tag ", tag.as_str()));
        if prev_tags.len() >= 2 {
            let t2 = &prev_tags[prev_tags.len() - 2];
            out.push(hash2("i-2 tag ", t2.as_str()));
            out.push(hash3("i-1 tag+tag ", t2.as_str(), tag.as_str()));
        }
    }

    // Shape flags
    if word.contains('-') {
        out.push(hash2("i has_hyphen", ""));
    }
    if word.chars().any(char::is_uppercase) {
        out.push(hash2("i has_upper", ""));
    }
    if clen > 0 && chars.iter().all(|&c| c.is_uppercase() && c.is_alphabetic()) {
        out.push(hash2("i all_upper", ""));
    }
}

/// Compute the shape of a word (capitalization pattern).
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
        w.insert(hash2("i word ", "the"), w1);
        let mut w2 = FxHashMap::default();
        w2.insert(SmolStr::new("NN"), 1.0);
        w2.insert(SmolStr::new("VB"), 0.0);
        w.insert(hash2("i word ", "cat"), w2);
        let mut w3 = FxHashMap::default();
        w3.insert(SmolStr::new("VB"), 1.0);
        w3.insert(SmolStr::new("NN"), 0.0);
        w.insert(hash2("i word ", "runs"), w3);
        w.insert(hash2("i shape ", "Xxx"), {
            let mut m = FxHashMap::default();
            m.insert(SmolStr::new("NN"), 1.0);
            m.insert(SmolStr::new("DT"), 0.0);
            m
        });
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

    #[test]
    fn test_hash_consistency() {
        let h1 = hash2("i word ", "the");
        let h2 = hash2("i word ", "the");
        assert_eq!(h1, h2);
        let h3 = hash2("i word ", "cat");
        assert_ne!(h1, h3);
    }
}
