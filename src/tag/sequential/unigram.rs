//! UnigramTagger — word-to-tag lookup (most frequent tag wins).

use hashbrown::HashMap as FastMap;
use pyo3::prelude::*;
use pyo3::types::PyList;
use smol_str::SmolStr;

use super::tagger_utils;

#[pyclass(name = "UnigramTagger", module = "fastnltk._rust")]
pub struct UnigramTagger {
    word_to_tag: FastMap<SmolStr, SmolStr>,
    default_tag: Option<SmolStr>,
    has_backoff: bool,
}

#[pymethods]
impl UnigramTagger {
    #[new]
    #[pyo3(signature = (backoff=None))]
    fn new(backoff: Option<&str>) -> Self {
        let default_tag = backoff.map(SmolStr::new);
        Self { word_to_tag: FastMap::new(), default_tag, has_backoff: backoff.is_some() }
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
        if !self.has_backoff {
            self.default_tag = tag_counts.iter().max_by_key(|(_, c)| **c).map(|(t, _)| t.clone());
        }
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
            assert!(result[0].1 == "DT" || result[0].1 == "NN");
        })
        .expect("GIL");
    }
}
