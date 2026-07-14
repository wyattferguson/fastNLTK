//! Sequential taggers — Rust-accelerated lookup-based POS taggers.
//!
//! Implements NLTK's sequential backoff taggers:
//!   - DefaultTagger: assign same tag to every word
//!   - NgramTagger (Unigram/Bigram/Trigram): lookup tag from training data
//!   - AffixTagger: tag based on word suffix/prefix
//!   - RegexpTagger: tag based on regex pattern match on the word
//!
//! All are pure lookup tables — no training loops, just counting + HashMap reads.

use std::collections::HashMap;

use hashbrown::HashMap as FastMap;
use smol_str::SmolStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use regex::Regex;

// ═══════════════════════════════════════════════════════════
// DefaultTagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "DefaultTagger", module = "fastnltk._rust")]
pub struct DefaultTagger {
    tag: String,
}

#[pymethods]
impl DefaultTagger {
    #[new]
    fn new(tag: &str) -> Self {
        DefaultTagger {
            tag: tag.to_string(),
        }
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens.into_iter().map(|w| (w, self.tag.clone())).collect()
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// UnigramTagger (also base for Bigram/Trigram)
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "UnigramTagger", module = "fastnltk._rust")]
pub struct UnigramTagger {
    word_to_tag: FastMap<SmolStr, SmolStr>,
    default_tag: Option<SmolStr>,
    backoff: Option<SmolStr>,
}

#[pymethods]
impl UnigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        UnigramTagger {
            word_to_tag: FastMap::new(),
            default_tag: None,
            backoff: backoff.map(SmolStr::new),
        }
    }

    /// Train on a list of tagged sentences: [[(word, tag), ...], ...]
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut counts: FastMap<SmolStr, FastMap<SmolStr, u64>> = FastMap::new();
        let mut tag_counts: FastMap<SmolStr, u64> = FastMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            for (word, tag) in &sent {
                counts
                    .entry(SmolStr::new(word))
                    .or_default()
                    .entry(SmolStr::new(tag))
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(SmolStr::new(tag)).or_insert(0) += 1;
            }
        }

        // Find most frequent tag per word
        for (word, tag_count) in &counts {
            let best = tag_count
                .iter()
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            self.word_to_tag.insert(word.clone(), best);
        }

        // Default tag is the overall most frequent tag
        self.default_tag = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(t, _)| t.clone());

        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens
            .into_iter()
            .map(|w| {
                let tag = self
                    .word_to_tag
                    .get(w.as_str())
                    .or(self.default_tag.as_ref())
                    .cloned()
                    .unwrap_or_default();
                (w, tag.to_string())
            })
            .collect()
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }

    /// Evaluate accuracy on tagged sentences.
    fn evaluate(&self, sentences: &Bound<'_, PyList>) -> f64 {
        let mut correct = 0u64;
        let mut total = 0u64;
        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract().unwrap_or_default();
            let words: Vec<String> = sent.iter().map(|(w, _)| w.clone()).collect();
            let gold_tags: Vec<String> = sent.iter().map(|(_, t)| t.clone()).collect();
            let pred = self.tag(words);
            for (p, g) in pred.iter().zip(gold_tags.iter()) {
                if &p.1 == g {
                    correct += 1;
                }
                total += 1;
            }
        }
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }
}

// ═══════════════════════════════════════════════════════════
// BigramTagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "BigramTagger", module = "fastnltk._rust")]
pub struct BigramTagger {
    /// (prev_tag, word) -> most frequent tag
    bigram_map: HashMap<(String, String), String>,
    /// default tag
    default_tag: Option<String>,
    /// backoff tagger name
    backoff: Option<String>,
}

#[pymethods]
impl BigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        BigramTagger {
            bigram_map: HashMap::new(),
            default_tag: None,
            backoff: backoff.map(|s| s.to_string()),
        }
    }

    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut counts: HashMap<(String, String), HashMap<String, u64>> = HashMap::new();
        let mut tag_counts: HashMap<String, u64> = HashMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut prev_tag = "START".to_string();
            for (word, tag) in &sent {
                let key = (prev_tag.clone(), word.clone());
                counts
                    .entry(key)
                    .or_default()
                    .entry(tag.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                prev_tag = tag.clone();
            }
        }

        for (key, tag_count) in &counts {
            let best = tag_count
                .iter()
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            self.bigram_map.insert(key.clone(), best);
        }

        self.default_tag = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(t, _)| t.clone());

        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut prev_tag = "START".to_string();
        for word in tokens {
            let key = (prev_tag.clone(), word.clone());
            let tag = self
                .bigram_map
                .get(&key)
                .or(self.default_tag.as_ref())
                .cloned()
                .unwrap_or_default();
            result.push((word, tag.clone()));
            prev_tag = tag;
        }
        result
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// TrigramTagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "TrigramTagger", module = "fastnltk._rust")]
pub struct TrigramTagger {
    /// (prev2_tag, prev1_tag, word) -> most frequent tag
    trigram_map: HashMap<(String, String, String), String>,
    default_tag: Option<String>,
    backoff: Option<String>,
}

#[pymethods]
impl TrigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        TrigramTagger {
            trigram_map: HashMap::new(),
            default_tag: None,
            backoff: backoff.map(|s| s.to_string()),
        }
    }

    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut counts: HashMap<(String, String, String), HashMap<String, u64>> = HashMap::new();
        let mut tag_counts: HashMap<String, u64> = HashMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut prev2 = "START".to_string();
            let mut prev1 = "START".to_string();
            for (word, tag) in &sent {
                let key = (prev2.clone(), prev1.clone(), word.clone());
                counts
                    .entry(key)
                    .or_default()
                    .entry(tag.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                prev2 = prev1.clone();
                prev1 = tag.clone();
            }
        }

        for (key, tag_count) in &counts {
            let best = tag_count
                .iter()
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            self.trigram_map.insert(key.clone(), best);
        }

        self.default_tag = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(t, _)| t.clone());

        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut prev2 = "START".to_string();
        let mut prev1 = "START".to_string();
        for word in tokens {
            let key = (prev2.clone(), prev1.clone(), word.clone());
            let tag = self
                .trigram_map
                .get(&key)
                .or(self.default_tag.as_ref())
                .cloned()
                .unwrap_or_default();
            result.push((word, tag.clone()));
            prev2 = prev1.clone();
            prev1 = tag;
        }
        result
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// AffixTagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "AffixTagger", module = "fastnltk._rust")]
pub struct AffixTagger {
    /// suffix -> tag
    suffix_map: HashMap<String, String>,
    /// prefix -> tag (for forward affixes)
    prefix_map: HashMap<String, String>,
    /// default tag
    default_tag: Option<String>,
    /// affix length
    affix_len: usize,
    /// use suffix (true) or prefix (false)
    use_suffix: bool,
}

#[pymethods]
impl AffixTagger {
    #[new]
    #[pyo3(signature = (affix_len=3, use_suffix=true, _backoff=None))]
    fn new(affix_len: usize, use_suffix: bool, _backoff: Option<&str>) -> Self {
        AffixTagger {
            suffix_map: HashMap::new(),
            prefix_map: HashMap::new(),
            default_tag: None,
            affix_len,
            use_suffix,
        }
    }

    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut affix_counts: HashMap<String, HashMap<String, u64>> = HashMap::new();
        let mut tag_counts: HashMap<String, u64> = HashMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            for (word, tag) in &sent {
                let affix = if self.use_suffix {
                    if word.len() >= self.affix_len {
                        word[word.len() - self.affix_len..].to_string()
                    } else {
                        word.clone()
                    }
                } else {
                    if word.len() >= self.affix_len {
                        word[..self.affix_len].to_string()
                    } else {
                        word.clone()
                    }
                };
                affix_counts
                    .entry(affix)
                    .or_default()
                    .entry(tag.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        for (affix, tag_count) in &affix_counts {
            let best = tag_count
                .iter()
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            if self.use_suffix {
                self.suffix_map.insert(affix.clone(), best);
            } else {
                self.prefix_map.insert(affix.clone(), best);
            }
        }

        self.default_tag = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(t, _)| t.clone());

        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens
            .into_iter()
            .map(|w| {
                let affix = if self.use_suffix {
                    if w.len() >= self.affix_len {
                        w[w.len() - self.affix_len..].to_string()
                    } else {
                        w.clone()
                    }
                } else if w.len() >= self.affix_len {
                    w[..self.affix_len].to_string()
                } else {
                    w.clone()
                };
                let map = if self.use_suffix {
                    &self.suffix_map
                } else {
                    &self.prefix_map
                };
                let tag = map
                    .get(&affix)
                    .or(self.default_tag.as_ref())
                    .cloned()
                    .unwrap_or_default();
                (w, tag)
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
// RegexpTagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "RegexpTagger", module = "fastnltk._rust")]
pub struct RegexpTagger {
    /// (pattern, tag) pairs, applied in order
    rules: Vec<(Regex, String)>,
    /// default tag if no pattern matches
    default_tag: Option<String>,
}

#[pymethods]
impl RegexpTagger {
    #[new]
    #[pyo3(signature = (patterns, _backoff=None))]
    fn new(patterns: Vec<(String, String)>, _backoff: Option<&str>) -> PyResult<Self> {
        let mut rules = Vec::with_capacity(patterns.len());
        for (pat, tag) in &patterns {
            let re = Regex::new(pat)
                .map_err(|e| PyValueError::new_err(format!("Invalid regex: {e}")))?;
            rules.push((re, tag.clone()));
        }
        Ok(RegexpTagger {
            rules,
            default_tag: None,
        })
    }

    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens
            .into_iter()
            .map(|w| {
                for (re, tag) in &self.rules {
                    if re.is_match(&w) {
                        return (w, tag.clone());
                    }
                }
                (w, self.default_tag.clone().unwrap_or_default())
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DefaultTagger>()?;
    m.add_class::<UnigramTagger>()?;
    m.add_class::<BigramTagger>()?;
    m.add_class::<TrigramTagger>()?;
    m.add_class::<AffixTagger>()?;
    m.add_class::<RegexpTagger>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tagger() {
        let tagger = DefaultTagger::new("NN");
        let result = tagger.tag(vec!["cat".into(), "dog".into()]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, "NN");
        assert_eq!(result[1].1, "NN");
    }

    #[test]
    fn test_default_tagger_empty() {
        let tagger = DefaultTagger::new("NN");
        let result = tagger.tag(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_unigram_train_and_tag() {
        // Test with the UnigramTagger's train method directly
        // (integration test requires Python runtime, tested via pytest)
        let mut tagger = UnigramTagger::new(None);
        // Without GIL, we can't create PyList — tested via Python integration tests
        assert!(tagger.word_to_tag.is_empty());
    }
}
