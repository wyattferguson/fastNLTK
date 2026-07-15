//! Perceptron tagger — averaged perceptron POS tagging.
//!
//! Uses u64 feature IDs (`FxHash` of component strings) to eliminate
//! all per-feature String allocation during inference.
//! Fast path: tagdict lookup for common words avoids feature extraction.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// FxHash algorithm constant (same as rustc-hash).
const FXHASH_K: u64 = 6_364_136_223_846_793_005;

/// Deterministic FxHash of a single byte slice.
/// rustc-hash v2 randomizes `FxHasher::default()`, breaking model
/// weights — we use our own deterministic implementation instead.
#[inline]
fn fxhash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |hash, &b| hash.wrapping_mul(FXHASH_K).wrapping_add(b as u64))
}

/// Hash a 2-component feature (prefix + value).
/// Writes bytes directly, avoiding any String allocation.
/// Consistent with hashing the concatenated string during model load.
#[inline]
fn hash2(a: &str, b: &str) -> u64 {
    let mut hash = 0u64;
    for &byte in a.as_bytes() { hash = hash.wrapping_mul(FXHASH_K).wrapping_add(byte as u64); }
    for &byte in b.as_bytes() { hash = hash.wrapping_mul(FXHASH_K).wrapping_add(byte as u64); }
    hash
}

/// Hash a 3-component feature. NLTK format: "a b c" (space-separated).
/// Both a and b should include trailing spaces; c appended with space prefix.
#[inline]
fn hash3(a: &str, b: &str, c: &str) -> u64 {
    let mut hash = 0u64;
    for &byte in a.as_bytes() { hash = hash.wrapping_mul(FXHASH_K).wrapping_add(byte as u64); }
    for &byte in b.as_bytes() { hash = hash.wrapping_mul(FXHASH_K).wrapping_add(byte as u64); }
    // Space separator before third component (NLTK space-joins all parts)
    hash = hash.wrapping_mul(FXHASH_K).wrapping_add(b' ' as u64);
    for &byte in c.as_bytes() { hash = hash.wrapping_mul(FXHASH_K).wrapping_add(byte as u64); }
    hash
}

#[pyclass(name = "PerceptronTagger", module = "fastnltk._rust")]
#[derive(Clone, Serialize, Deserialize)]
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
                // Hash the full key string (consistent with hash2)
                let feat_id = fxhash_bytes(feat_key.as_bytes());

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
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            sentences.par_iter().map(|s| self.tag_sentence(s)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        sentences.iter().map(|s| self.tag_sentence(s)).collect()
    }

    /// Save tagger state to a bincode cache file.
    fn save_cache(&self, path: &str) -> PyResult<()> {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        std::fs::write(path, bytes)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load tagger state from a bincode cache file into self.
    fn load_from_cache(&mut self, path: &str) -> PyResult<()> {
        let bytes =
            std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let (tagger, _): (Self, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.weights = tagger.weights;
        self.tagdict = tagger.tagdict;
        self.classes = tagger.classes;
        Ok(())
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

/// Collect feature integer IDs matching NLTK's `_get_features` exactly.
/// NLTK format: `' '.join((name,) + args)` — space-separated components.
fn collect_feature_ids(
    word: &str,
    i: usize,
    tokens: &[String],
    prev_tags: &[SmolStr],
    out: &mut Vec<u64>,
) {
    // NLTK: add("i word", context[i])
    out.push(hash2("i word ", word));

    // NLTK: add("i suffix", word[-3:]) — last 3 chars
    let suffix = if word.len() >= 3 { &word[word.len() - 3..] } else { word };
    out.push(hash2("i suffix ", suffix));

    // NLTK: add("i pref1", word[0])
    if let Some(c) = word.chars().next() {
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        out.push(hash2("i pref1 ", s));
    }

    // NLTK: add("i-1 tag", prev)
    // NLTK: add("i-2 tag", prev2)
    // NLTK: add("i tag+i-2 tag", prev, prev2)
    // NLTK: add("i-1 tag+i word", prev, context[i])
    if let Some(tag) = prev_tags.last() {
        out.push(hash2("i-1 tag ", tag.as_str()));
        if prev_tags.len() >= 2 {
            let t2 = &prev_tags[prev_tags.len() - 2];
            out.push(hash2("i-2 tag ", t2.as_str()));
            // NLTK: add("i tag+i-2 tag", prev, prev2) → "i tag+i-2 tag prev prev2"
            out.push(hash3("i tag+i-2 tag ", tag.as_str(), t2.as_str()));
        }
        // NLTK: add("i-1 tag+i word", prev, context[i]) → "i-1 tag+i word prev word"
        out.push(hash3("i-1 tag+i word ", tag.as_str(), word));
    }

    // NLTK: add("i-1 word", context[i - 1])
    if i > 0 {
        let pw = &tokens[i - 1];
        out.push(hash2("i-1 word ", pw));
        // NLTK: add("i-1 suffix", context[i-1][-3:])
        let prev_suffix = if pw.len() >= 3 { &pw[pw.len() - 3..] } else { pw.as_str() };
        out.push(hash2("i-1 suffix ", prev_suffix));
    }

    // NLTK: add("i-2 word", context[i - 2])
    if i > 1 {
        out.push(hash2("i-2 word ", &tokens[i - 2]));
    }

    // NLTK: add("i+1 word", context[i + 1])
    if i + 1 < tokens.len() {
        let nw = &tokens[i + 1];
        out.push(hash2("i+1 word ", nw));
        // NLTK: add("i+1 suffix", context[i+1][-3:])
        let next_suffix = if nw.len() >= 3 { &nw[nw.len() - 3..] } else { nw.as_str() };
        out.push(hash2("i+1 suffix ", next_suffix));
    }

    // NLTK: add("i+2 word", context[i + 2])
    if i + 2 < tokens.len() {
        out.push(hash2("i+2 word ", &tokens[i + 2]));
    }

    // NLTK: add("bias")
    out.push(fxhash_bytes(b"bias"));
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

    #[test]
    fn test_hash_equivalent_to_full_key() {
        // Verify hash2("i word ", "believe") == fxhash_bytes("i word believe")
        assert_eq!(hash2("i word ", "believe"), fxhash_bytes(b"i word believe"),
            "hash2 vs fxhash_bytes mismatch");
        // Verify hash2("i+1 word ", "how") == fxhash_bytes("i+1 word how")
        assert_eq!(hash2("i+1 word ", "how"), fxhash_bytes(b"i+1 word how"));
        // Verify hash2("i suffix ", "e") == fxhash_bytes("i suffix e")
        assert_eq!(hash2("i suffix ", "e"), fxhash_bytes(b"i suffix e"));
        // Verify hash2 == model-load style
        assert_eq!(hash2("i-1 word ", "the"), fxhash_bytes(b"i-1 word the"));
        // Verify hash3 == combined with space separator
        // NLTK format: "i-1 tag+tag DT NN" (space before each component)
        assert_eq!(hash3("i-1 tag+tag ", "DT", "NN"),
            fxhash_bytes(b"i-1 tag+tag DT NN"));
    }
}
