//! RegexpTagger — tag words by regex pattern matching.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use regex::Regex;
use smol_str::SmolStr;

use super::tagger_utils;

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
        tagger_utils::maybe_parallel(sentences, |s| self.tag(s))
    }

    fn evaluate(&self, sentences: &Bound<'_, PyList>) -> f64 {
        tagger_utils::evaluate(sentences, |tokens| self.tag(tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
