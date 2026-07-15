//! Ngram and pattern-based sequential taggers.

use std::hash::Hasher;

use hashbrown::HashMap as FastMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use regex::Regex;
use rustc_hash::FxHasher;
use smol_str::SmolStr;

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
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return sentences.into_par_iter().map(|s| self.tag(s)).collect();
        }
        #[cfg(not(feature = "parallel"))]
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

// BigramTagger — integer IDs: (prev_tag_id, word_hash) → tag_id.
// No SmolStr clone per lookup. FxHash of word is ~4ns per word.
#[pyclass(name = "BigramTagger", module = "fastnltk._rust")]
pub struct BigramTagger {
    /// (`prev_tag_id`, `word_hash`) → `tag_id`
    map: FastMap<(u16, u64), u16>,
    /// `tag_id` → tag string (for output)
    tag_names: Vec<SmolStr>,
    /// Maximum tag ID seen (for bound checks)
    start_id: u16,
    default_id: u16,
}

fn hash_word(w: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(w.as_bytes());
    h.finish()
}

fn ensure_tag(
    tag: &str,
    tag_to_id: &mut FastMap<SmolStr, u16>,
    tag_names: &mut Vec<SmolStr>,
) -> u16 {
    let next = tag_to_id.len() as u16;
    let id = *tag_to_id.entry(SmolStr::new(tag)).or_insert(next);
    if id == next {
        tag_names.push(SmolStr::new(tag));
    }
    id
}

#[pymethods]
impl BigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        let _ = backoff;
        let mut tag_to_id = FastMap::new();
        let mut tag_names = Vec::new();
        let start_id = ensure_tag("<S>", &mut tag_to_id, &mut tag_names);
        let def_id = ensure_tag("NN", &mut tag_to_id, &mut tag_names);
        Self { map: FastMap::new(), tag_names, start_id, default_id: def_id }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut tag_counts: FastMap<SmolStr, u64> = FastMap::new();
        let mut tag_to_id: FastMap<SmolStr, u16> =
            self.tag_names.iter().enumerate().map(|(i, t)| (t.clone(), i as u16)).collect();
        let mut raw: FastMap<(u16, u64), FastMap<u16, u64>> = FastMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut prev_id = self.start_id;
            for (word, tag) in &sent {
                let tag_id = ensure_tag(tag, &mut tag_to_id, &mut self.tag_names);
                let wh = hash_word(word);
                raw.entry((prev_id, wh))
                    .or_default()
                    .entry(tag_id)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(SmolStr::new(tag)).or_insert(0) += 1;
                prev_id = tag_id;
            }
        }

        // Flatten: keep only the most frequent tag per (prev, word)
        for ((prev, wh), tag_count) in raw {
            if let Some((best, _)) = tag_count.iter().max_by_key(|(_, c)| **c) {
                self.map.insert((prev, wh), *best);
            }
        }

        self.default_id = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map_or(self.default_id, |(t, _)| tag_to_id[t]);
        Ok(())
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let mut out = Vec::with_capacity(tokens.len());
        let mut prev = self.start_id;
        for w in tokens {
            let wh = hash_word(&w);
            let tag_id = self.map.get(&(prev, wh)).copied().unwrap_or(self.default_id);
            out.push((w, self.tag_names[tag_id as usize].to_string()));
            prev = tag_id;
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return sentences.into_par_iter().map(|s| self.tag(s)).collect();
        }
        #[cfg(not(feature = "parallel"))]
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

// TrigramTagger — integer IDs: (prev2_id, prev1_id, word_hash) → tag_id.
#[pyclass(name = "TrigramTagger", module = "fastnltk._rust")]
pub struct TrigramTagger {
    map: FastMap<(u16, u16, u64), u16>,
    tag_names: Vec<SmolStr>,
    start_id: u16,
    default_id: u16,
}

#[pymethods]
impl TrigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        let _ = backoff;
        let mut tag_to_id = FastMap::new();
        let mut tag_names = Vec::new();
        let start_id = ensure_tag("<S>", &mut tag_to_id, &mut tag_names);
        let def_id = ensure_tag("NN", &mut tag_to_id, &mut tag_names);
        Self { map: FastMap::new(), tag_names, start_id, default_id: def_id }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        let mut tag_counts: FastMap<SmolStr, u64> = FastMap::new();
        let mut tag_to_id: FastMap<SmolStr, u16> =
            self.tag_names.iter().enumerate().map(|(i, t)| (t.clone(), i as u16)).collect();
        let mut raw: FastMap<(u16, u16, u64), FastMap<u16, u64>> = FastMap::new();

        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            let mut p2 = self.start_id;
            let mut p1 = self.start_id;
            for (word, tag) in &sent {
                let tag_id = ensure_tag(tag, &mut tag_to_id, &mut self.tag_names);
                let wh = hash_word(word);
                raw.entry((p2, p1, wh))
                    .or_default()
                    .entry(tag_id)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                *tag_counts.entry(SmolStr::new(tag)).or_insert(0) += 1;
                p2 = p1;
                p1 = tag_id;
            }
        }

        for ((p2, p1, wh), tag_count) in raw {
            if let Some((best, _)) = tag_count.iter().max_by_key(|(_, c)| **c) {
                self.map.insert((p2, p1, wh), *best);
            }
        }

        self.default_id = tag_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map_or(self.default_id, |(t, _)| tag_to_id[t]);
        Ok(())
    }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let mut out = Vec::with_capacity(tokens.len());
        let mut p2 = self.start_id;
        let mut p1 = self.start_id;
        for w in tokens {
            let wh = hash_word(&w);
            let tag_id = self.map.get(&(p2, p1, wh)).copied().unwrap_or(self.default_id);
            out.push((w, self.tag_names[tag_id as usize].to_string()));
            p2 = p1;
            p1 = tag_id;
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return sentences.into_par_iter().map(|s| self.tag(s)).collect();
        }
        #[cfg(not(feature = "parallel"))]
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
    affix_len: usize,
    default_tag: Option<SmolStr>,
}

#[pymethods]
impl AffixTagger {
    #[new]
    #[pyo3(signature = (affix_len=3, use_suffix=true))]
    fn new(affix_len: usize, use_suffix: bool) -> Self {
        Self {
            prefix_map: FastMap::new(),
            suffix_map: FastMap::new(),
            use_suffix,
            affix_len,
            default_tag: None,
        }
    }
    fn train(&mut self, sentences: &Bound<'_, PyList>) -> PyResult<()> {
        for item in sentences.iter() {
            let sent: Vec<(String, String)> = item.extract()?;
            for (word, tag) in &sent {
                let al = self.affix_len;
                let key = if self.use_suffix {
                    SmolStr::new(&word[word.len().saturating_sub(al)..])
                } else {
                    SmolStr::new(&word[..word.len().min(al)])
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
        let al = self.affix_len;
        for w in tokens {
            let affix = if self.use_suffix {
                let cut = w.len().min(al);
                SmolStr::new(&w[w.len() - cut..])
            } else {
                SmolStr::new(&w[..w.len().min(al)])
            };
            let map = if self.use_suffix { &self.suffix_map } else { &self.prefix_map };
            let tag = map.get(&affix).cloned().unwrap_or_else(|| default.clone());
            out.push((w, tag.to_string()));
        }
        out
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return sentences.into_par_iter().map(|s| self.tag(s)).collect();
        }
        #[cfg(not(feature = "parallel"))]
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
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            return sentences.into_par_iter().map(|s| self.tag(s)).collect();
        }
        #[cfg(not(feature = "parallel"))]
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
