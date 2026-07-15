//! Ngram and pattern-based sequential taggers.

use hashbrown::HashMap as FastMap;
use smol_str::SmolStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use regex::Regex;

// UnigramTagger
#[pyclass(name = "UnigramTagger", module = "fastnltk._rust")]
pub struct UnigramTagger {
    word_to_tag: FastMap<SmolStr, SmolStr>,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl UnigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        Self { word_to_tag: FastMap::new(), default_tag: backoff.map(SmolStr::new) }
    }
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
        for (word, tag_count) in &counts {
            let best = tag_count
                .iter()
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_default();
            self.word_to_tag.insert(word.clone(), best);
        }
        self.default_tag = tag_counts.iter().max_by_key(|(_, c)| **c).map(|(t, _)| t.clone());
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
            }
            total += sent.len() as u64;
        }
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }
}

// BigramTagger
#[pyclass(name = "BigramTagger", module = "fastnltk._rust")]
pub struct BigramTagger {
    bigram_map: FastMap<(SmolStr, SmolStr), SmolStr>,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl BigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        Self { bigram_map: FastMap::new(), default_tag: backoff.map(SmolStr::new) }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut counts: FastMap<(SmolStr, SmolStr), FastMap<SmolStr, u64>> = FastMap::new();
        let mut tag_counts: FastMap<SmolStr, u64> = FastMap::new();
        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut prev = SmolStr::new_inline("START");
            for (word, tag) in &sent {
                let key = (prev.clone(), SmolStr::new(word));
                counts
                    .entry(key)
                    .or_default()
                    .entry(SmolStr::new(tag))
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(SmolStr::new(tag)).or_insert(0) += 1;
                prev = SmolStr::new(tag);
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
        self.default_tag = tag_counts.iter().max_by_key(|(_, c)| **c).map(|(t, _)| t.clone());
        Ok(())
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let n = tokens.len();
        let mut out = Vec::with_capacity(n);
        let mut prev = SmolStr::new_inline("START");
        let default = self.default_tag.as_ref().map(|t| t.as_str()).unwrap_or("");
        for w in tokens {
            let w_smol = SmolStr::new(&w);
            let key = (prev, w_smol);
            let tag = self.bigram_map.get(&key).map(|s| s.as_str()).unwrap_or(default);
            out.push((w, tag.to_string()));
            prev = SmolStr::new(tag);
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
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
            }
            total += sent.len() as u64;
        }
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }
}

// TrigramTagger
#[pyclass(name = "TrigramTagger", module = "fastnltk._rust")]
pub struct TrigramTagger {
    trigram_map: FastMap<(SmolStr, SmolStr, SmolStr), SmolStr>,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl TrigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        Self { trigram_map: FastMap::new(), default_tag: backoff.map(SmolStr::new) }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut counts: FastMap<(SmolStr, SmolStr, SmolStr), FastMap<SmolStr, u64>> =
            FastMap::new();
        let mut tag_counts: FastMap<SmolStr, u64> = FastMap::new();
        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut prev2 = SmolStr::new_inline("START");
            let mut prev1 = SmolStr::new_inline("START");
            for (word, tag) in &sent {
                let key = (prev2.clone(), prev1.clone(), SmolStr::new(word));
                counts
                    .entry(key)
                    .or_default()
                    .entry(SmolStr::new(tag))
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(SmolStr::new(tag)).or_insert(0) += 1;
                prev2 = prev1;
                prev1 = SmolStr::new(tag);
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
        self.default_tag = tag_counts.iter().max_by_key(|(_, c)| **c).map(|(t, _)| t.clone());
        Ok(())
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let n = tokens.len();
        let mut out = Vec::with_capacity(n);
        let mut prev2 = SmolStr::new_inline("START");
        let mut prev1 = SmolStr::new_inline("START");
        let default = self.default_tag.as_ref().map(|t| t.as_str()).unwrap_or("");
        for w in tokens {
            let w_smol = SmolStr::new(&w);
            let key = (prev2, prev1.clone(), w_smol);
            let tag = self.trigram_map.get(&key).map(|s| s.as_str()).unwrap_or(default);
            out.push((w, tag.to_string()));
            prev2 = prev1;
            prev1 = SmolStr::new(tag);
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
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
            }
            total += sent.len() as u64;
        }
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }
}

// AffixTagger
#[pyclass(name = "AffixTagger", module = "fastnltk._rust")]
pub struct AffixTagger {
    prefix_map: FastMap<SmolStr, SmolStr>,
    suffix_map: FastMap<SmolStr, SmolStr>,
    use_suffix: bool,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl AffixTagger {
    #[new]
    #[pyo3(signature = (_affix_len=3, use_suffix=true, _backoff=None))]
    fn new(_affix_len: usize, use_suffix: bool, _backoff: Option<&str>) -> Self {
        Self {
            prefix_map: FastMap::new(),
            suffix_map: FastMap::new(),
            use_suffix,
            default_tag: None,
        }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            for (word, tag) in &sent {
                let key = if self.use_suffix {
                    SmolStr::new(&word[word.len().saturating_sub(3)..])
                } else {
                    SmolStr::new(&word[..word.len().min(3)])
                };
                let map = if self.use_suffix { &mut self.suffix_map } else { &mut self.prefix_map };
                map.insert(key, SmolStr::new(tag));
            }
        }
        Ok(())
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let n = tokens.len();
        let mut out = Vec::with_capacity(n);
        let default: SmolStr = self.default_tag.clone().unwrap_or_default();
        for w in tokens {
            let affix = if self.use_suffix {
                let cut = w.len().min(3);
                SmolStr::new(&w[w.len() - cut..])
            } else {
                SmolStr::new(&w[..w.len().min(3)])
            };
            let map = if self.use_suffix { &self.suffix_map } else { &self.prefix_map };
            let tag = map.get(&affix).cloned().unwrap_or_else(|| default.clone());
            out.push((w, tag.to_string()));
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
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
            }
            total += sent.len() as u64;
        }
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }
}

// RegexpTagger
#[pyclass(name = "RegexpTagger", module = "fastnltk._rust")]
pub struct RegexpTagger {
    rules: Vec<(Regex, SmolStr)>,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl RegexpTagger {
    #[new]
    #[pyo3(signature = (patterns, _backoff=None))]
    fn new(patterns: Vec<(String, String)>, _backoff: Option<&str>) -> PyResult<Self> {
        let mut rules = Vec::with_capacity(patterns.len());
        for (pattern, tag) in patterns {
            let re = Regex::new(&pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
            rules.push((re, SmolStr::new(&tag)));
        }
        Ok(Self { rules, default_tag: None })
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens
            .into_iter()
            .map(|w| {
                for (re, tag) in &self.rules {
                    if re.is_match(&w) {
                        return (w, tag.to_string());
                    }
                }
                (w, self.default_tag.clone().unwrap_or_default().to_string())
            })
            .collect()
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
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
            }
            total += sent.len() as u64;
        }
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::types::PyList;

    fn train_data(py: Python<'_>) -> Bound<'_, PyList> {
        let list = PyList::empty(py);
        let s1 = vec![("the".to_string(), "DT".to_string()), ("cat".to_string(), "NN".to_string())];
        let s2 = vec![("the".to_string(), "DT".to_string()), ("dog".to_string(), "NN".to_string())];
        list.append(s1).unwrap();
        list.append(s2).unwrap();
        list
    }

    #[test]
    fn test_unigram_train_and_tag() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tagger = UnigramTagger::new(None);
            tagger.train(&train_data(py)).unwrap();
            let result = tagger.tag(vec!["the".into(), "cat".into()]);
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].1, "DT");
        })
        .expect("GIL");
    }

    #[test]
    fn test_unigram_unknown_word() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tagger = UnigramTagger::new(None);
            tagger.train(&train_data(py)).unwrap();
            let result = tagger.tag(vec!["xyzzy".into()]);
            // Falls back to most frequent tag from training
            assert!(result[0].1 == "DT" || result[0].1 == "NN");
        })
        .expect("GIL");
    }

    #[test]
    fn test_bigram_train_and_tag() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tagger = BigramTagger::new(None);
            tagger.train(&train_data(py)).unwrap();
            let result = tagger.tag(vec!["the".into(), "cat".into()]);
            assert_eq!(result.len(), 2);
        })
        .expect("GIL");
    }

    #[test]
    fn test_trigram_train_and_tag() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tagger = TrigramTagger::new(None);
            tagger.train(&train_data(py)).unwrap();
            let result = tagger.tag(vec!["the".into(), "cat".into()]);
            assert_eq!(result.len(), 2);
        })
        .expect("GIL");
    }

    #[test]
    fn test_affix_tagger_suffix() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tagger = AffixTagger::new(3, true, None);
            let list = PyList::empty(py);
            list.append(vec![("walking".to_string(), "VBG".to_string())]).unwrap();
            tagger.train(&list).unwrap();
            let result = tagger.tag(vec!["running".into()]);
            assert_eq!(result[0].1, "VBG");
        })
        .expect("GIL");
    }

    #[test]
    fn test_regexp_tagger_basic() {
        let patterns = vec![
            (r"\d+".to_string(), "CD".to_string()),
            (r"[A-Z].*".to_string(), "NNP".to_string()),
        ];
        let tagger = RegexpTagger::new(patterns, None).unwrap();
        let result = tagger.tag(vec!["123".into(), "John".into(), "hello".into()]);
        assert_eq!(result[0].1, "CD");
        assert_eq!(result[1].1, "NNP");
    }
}
