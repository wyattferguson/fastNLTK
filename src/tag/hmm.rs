//! HMM tagger — Rust-accelerated Hidden Markov Model tagger.
//!
//! Wraps rustling::hmm::HiddenMarkovModel to provide NLTK-compatible:
//! - Train from labeled data
//! - Tag (Viterbi decoding)
//! - Tag multiple sentences
//!
//! NLTK equivalent: nltk.tag.hmm.HiddenMarkovModelTagger

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustling::hmm::{BaseHiddenMarkovModel, HiddenMarkovModel};
use rustling::seq_feature::default_tagger_hmm_features;

#[pyclass(name = "HiddenMarkovModelTagger", module = "fastnltk._rust")]
pub struct HiddenMarkovModelTagger {
    inner: Option<HiddenMarkovModel>,
    n_states: usize,
    fitted: bool,
}

#[pymethods]
impl HiddenMarkovModelTagger {
    #[new]
    #[pyo3(signature = (n_states=10, n_iter=10, tolerance=1e-4, gamma=0.1))]
    fn new(n_states: usize, n_iter: usize, tolerance: f64, gamma: f64) -> PyResult<Self> {
        Ok(HiddenMarkovModelTagger {
            inner: None,
            n_states,
            fitted: false,
        })
    }

    /// Train the HMM tagger from labeled sentences.
    /// sentences: list of list of (word, tag) tuples
    #[pyo3(signature = (sentences))]
    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        if sentences.is_empty() {
            return Err(PyValueError::new_err("No training data provided"));
        }

        let features = default_tagger_hmm_features();
        let mut model = HiddenMarkovModel::new(
            self.n_states,
            features,
            1, // n_iter for vocab building
            self.n_states.max(10) as f64, // tolerance
            self.n_states.max(10) as f64, // gamma
            0.1,                           // random_seed
            None,
        );

        // Convert training data: each word is an observation sequence
        // Tags are used as labels for supervised training
        let observations: Vec<Vec<String>> = sentences
            .iter()
            .map(|sent| sent.iter().map(|(word, _)| word.clone()).collect())
            .collect();

        // Build vocabulary from observations
        model.build_vocab(&observations);

        // Convert labels to sequences of tag strings
        let labels: Vec<Vec<String>> = sentences
            .iter()
            .map(|sent| sent.iter().map(|(_, tag)| tag.clone()).collect())
            .collect();

        // Supervised training with labels
        model.fit_labeled(observations, labels).map_err(|e| {
            PyValueError::new_err(format!("HMM training failed: {e}"))
        })?;

        self.inner = Some(model);
        self.fitted = true;
        Ok(())
    }

    /// Tag a sequence of tokens using Viterbi decoding.
    fn tag(&self, tokens: Vec<String>) -> PyResult<Vec<(String, String)>> {
        let model = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("HMM tagger has not been trained yet")
        })?;

        let result = model
            .predict(vec![tokens.clone()])
            .map_err(|e| PyValueError::new_err(format!("HMM tagging failed: {e}")))?;

        if result.is_empty() || result[0].is_empty() {
            return Err(PyValueError::new_err("HMM tagging produced no output"));
        }

        // Map state indices to tag labels
        let tags = if let Some(labels) = &model.state_labels() {
            result[0]
                .iter()
                .map(|&state| {
                    let tag = labels
                        .get(state)
                        .cloned()
                        .unwrap_or_else(|| format!("TAG_{state}"));
                    (tokens[result[0].iter().position(|&s| s == state).unwrap_or(0)].clone(), tag)
                })
                .collect()
        } else {
            tokens
                .iter()
                .zip(result[0].iter())
                .map(|(word, &state)| (word.clone(), format!("TAG_{state}")))
                .collect()
        };

        Ok(tags)
    }

    /// Tag multiple sentences.
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> PyResult<Vec<Vec<(String, String)>>> {
        sentences
            .into_iter()
            .map(|tokens| self.tag(tokens))
            .collect()
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
