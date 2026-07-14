//! Language models — Rust-accelerated ngram language models.
//!
//! Implements NLTK-compatible:
//! - `MLE` (Maximum Likelihood Estimation)
//! - `Lidstone` (additive smoothing)
//! - `Laplace` (add-one smoothing)
//! - `KneserNeyInterpolated`
//! - `WittenBellInterpolated`
//! - `StupidBackoff`

pub mod kneser_ney;
pub mod stupid_backoff;
pub mod witten_bell;

use rustling::lm::{
    BaseLanguageModel, Laplace as RustLaplace, Lidstone as RustLidstone, MLE as RustMLE,
};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// MLE
#[pyclass(name = "MLE", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct MLE {
    inner: RustMLE,
}

#[pymethods]
impl MLE {
    #[new]
    #[pyo3(signature = (order))]
    fn new(order: usize) -> PyResult<Self> {
        RustMLE::new(order)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.inner.fit(sentences);
    }
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.score(word, context).unwrap_or(0.0)
    }
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.logscore(word, context).unwrap_or(f64::NEG_INFINITY)
    }
    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        self.inner.generate(num_words, text_seed, random_seed).unwrap_or_default()
    }
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
}

// Lidstone
#[pyclass(name = "Lidstone", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct Lidstone {
    inner: RustLidstone,
}

#[pymethods]
impl Lidstone {
    #[new]
    #[pyo3(signature = (order, gamma))]
    fn new(order: usize, gamma: f64) -> PyResult<Self> {
        RustLidstone::new(order, gamma)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.inner.fit(sentences);
    }
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.score(word, context).unwrap_or(0.0)
    }
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.logscore(word, context).unwrap_or(f64::NEG_INFINITY)
    }
    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        self.inner.generate(num_words, text_seed, random_seed).unwrap_or_default()
    }
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
}

// Laplace
#[pyclass(name = "Laplace", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct Laplace {
    inner: RustLaplace,
}

#[pymethods]
impl Laplace {
    #[new]
    #[pyo3(signature = (order))]
    fn new(order: usize) -> PyResult<Self> {
        RustLaplace::new(order)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.inner.fit(sentences);
    }
    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.score(word, context).unwrap_or(0.0)
    }
    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: String, context: Option<Vec<String>>) -> f64 {
        self.inner.logscore(word, context).unwrap_or(f64::NEG_INFINITY)
    }
    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        self.inner.generate(num_words, text_seed, random_seed).unwrap_or_default()
    }
    fn order(&self) -> usize {
        self.inner.order()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn fitted(&self) -> bool {
        self.inner.fitted()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MLE>()?;
    m.add_class::<Lidstone>()?;
    m.add_class::<Laplace>()?;
    m.add_class::<kneser_ney::KneserNeyInterpolated>()?;
    m.add_class::<witten_bell::WittenBellInterpolated>()?;
    m.add_class::<stupid_backoff::StupidBackoff>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_create() {
        let model = MLE::new(2).unwrap();
        assert_eq!(model.order(), 2);
        assert!(!model.fitted());
    }
    #[test]
    fn test_mle_fit_and_score() {
        let mut model = MLE::new(2).unwrap();
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let score = model.score("cat".into(), Some(vec!["the".into()]));
        assert!((score - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_mle_logscore() {
        let mut model = MLE::new(2).unwrap();
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let logscore = model.logscore("cat".into(), Some(vec!["the".into()]));
        assert!(logscore.is_finite() && logscore < 0.0);
    }
    #[test]
    fn test_mle_generate() {
        let mut model = MLE::new(2).unwrap();
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let words = model.generate(3, Some(vec!["the".into()]), Some(42));
        assert!(!words.is_empty() && words.len() <= 3);
    }
    #[test]
    fn test_mle_vocab_size() {
        let mut model = MLE::new(2).unwrap();
        model.fit(vec![vec!["the".into(), "cat".into()]]);
        assert!(model.vocab_size() >= 4);
    }
    #[test]
    fn test_lidstone_create() {
        let model = Lidstone::new(2, 0.5).unwrap();
        assert_eq!(model.order(), 2);
    }
    #[test]
    fn test_lidstone_fit_and_score() {
        let mut model = Lidstone::new(2, 0.5).unwrap();
        model.fit(vec![vec!["the".into(), "cat".into(), "sat".into()]]);
        let score = model.score("cat".into(), Some(vec!["the".into()]));
        assert!(score > 0.0 && score <= 1.0);
    }
    #[test]
    fn test_laplace_create() {
        let model = Laplace::new(2).unwrap();
        assert_eq!(model.order(), 2);
    }
    #[test]
    fn test_laplace_fit_and_score() {
        let mut model = Laplace::new(2).unwrap();
        model.fit(vec![vec!["the".into(), "cat".into(), "sat".into()]]);
        let score = model.score("dog".into(), Some(vec!["the".into()]));
        assert!(score > 0.0);
    }
    #[test]
    fn test_oov_score() {
        let mut model = MLE::new(2).unwrap();
        model.fit(vec![vec!["the".into(), "cat".into()]]);
        let score = model.score("xyzzy".into(), Some(vec!["the".into()]));
        assert!(score >= 0.0);
    }
}
