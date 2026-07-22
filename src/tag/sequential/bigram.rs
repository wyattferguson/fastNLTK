//! BigramTagger — integer-ID bigram tagger (prev_tag, word_hash) → tag.

use hashbrown::HashMap as FastMap;
use pyo3::prelude::*;
use pyo3::types::PyList;
use smol_str::SmolStr;

use super::tagger_utils::{self, ensure_tag, hash_word};

#[pyclass(name = "BigramTagger", module = "fastnltk._rust")]
pub struct BigramTagger {
    /// (prev_tag_id, word_hash) → tag_id
    map: FastMap<(u16, u64), u16>,
    /// tag_id → tag string (for output)
    tag_names: Vec<SmolStr>,
    start_id: u16,
    default_id: u16,
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
        tagger_utils::maybe_parallel(sentences, |s| self.tag(s))
    }

    fn evaluate(&self, sentences: &Bound<'_, PyList>) -> f64 {
        tagger_utils::evaluate(sentences, |tokens| self.tag(tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;

    fn train_data(py: Python<'_>) -> Bound<'_, PyList> {
        let list = PyList::empty(py);
        list.append(vec![
            ("the".to_string(), "DT".to_string()),
            ("cat".to_string(), "NN".to_string()),
        ])
        .unwrap();
        list.append(vec![
            ("the".to_string(), "DT".to_string()),
            ("dog".to_string(), "NN".to_string()),
        ])
        .unwrap();
        list
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
}
