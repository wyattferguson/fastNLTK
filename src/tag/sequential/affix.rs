//! AffixTagger — tag words by fixed-length prefix or suffix.

use hashbrown::HashMap as FastMap;
use pyo3::prelude::*;
use pyo3::types::PyList;
use smol_str::SmolStr;

use super::tagger_utils;

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
    #[pyo3(signature = (affix_len=3, use_suffix=true, _backoff=None))]
    fn new(affix_len: usize, use_suffix: bool, _backoff: Option<&str>) -> Self {
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
}
