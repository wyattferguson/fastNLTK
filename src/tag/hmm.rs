//! HMM tagger — wraps `rustling::hmm::HiddenMarkovModel`.
//!
//! NLTK equivalent: nltk.tag.hmm.HiddenMarkovModelTagger

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustling::hmm::{BaseHiddenMarkovModel, HiddenMarkovModel};
use rustling::persistence::ModelError;
use rustling::seq_feature::default_tagger_hmm_features;

#[pyclass(name = "HiddenMarkovModelTagger", module = "fastnltk._rust")]
pub struct HiddenMarkovModelTagger {
    inner: Option<HiddenMarkovModel>,
    n_states: usize,
    n_iter: usize,
    tolerance: f64,
    gamma: f64,
}

fn map_err(e: ModelError) -> PyErr {
    PyValueError::new_err(format!("HMM error: {e}"))
}

#[pymethods]
impl HiddenMarkovModelTagger {
    #[new]
    #[pyo3(signature = (n_states=10, n_iter=10, tolerance=1e-4, gamma=0.1))]
    fn new(n_states: usize, n_iter: usize, tolerance: f64, gamma: f64) -> Self {
        Self { inner: None, n_states, n_iter, tolerance, gamma }
    }

    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        if sentences.is_empty() {
            return Err(PyValueError::new_err("No training data"));
        }
        // Build HMM with default tagger features
        let mut model = HiddenMarkovModel::new(
            self.n_states,
            self.n_iter,
            self.tolerance,
            self.gamma,
            None,
            Some(default_tagger_hmm_features()),
        )
        .map_err(map_err)?;

        // Extract observation sequences (words) and labels (tags) in one pass
        let (observations, labels): (Vec<Vec<String>>, Vec<Vec<String>>) = sentences
            .iter()
            .map(|s| {
                let (obs, lbls): (Vec<_>, Vec<_>) =
                    s.iter().map(|(w, t)| (w.clone(), t.clone())).unzip();
                (obs, lbls)
            })
            .unzip();

        // Build vocabulary then train supervised
        model.build_vocab(&observations);
        model.fit_labeled(observations, labels).map_err(map_err)?;
        self.inner = Some(model);
        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> PyResult<Vec<(String, String)>> {
        let model =
            self.inner.as_ref().ok_or_else(|| PyValueError::new_err("Model not trained"))?;
        let result = model.predict(vec![tokens.clone()]).map_err(map_err)?;
        if result.is_empty() || result[0].is_empty() {
            return Err(PyValueError::new_err("No output from HMM predict"));
        }
        let labels = model.state_labels().clone().unwrap_or_default();
        let tagged: Vec<(String, String)> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
                let state = if i < result[0].len() { result[0][i] } else { 0 };
                let tag = labels.get(state).cloned().unwrap_or_else(|| format!("TAG_{state}"));
                (w, tag)
            })
            .collect();
        Ok(tagged)
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> PyResult<Vec<Vec<(String, String)>>> {
        sentences.into_iter().map(|t| self.tag(t)).collect()
    }
}
