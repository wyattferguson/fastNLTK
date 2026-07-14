//! Stupid backoff language model.
//!
//! Implements: P(w) = c(w)/total for seen, alpha/(N+V) for unseen.

use pyo3::prelude::*;

/// Stupid backoff language model.
#[pyclass(name = "StupidBackoff", module = "fastnltk._rust")]
pub struct StupidBackoff {
    order: usize,
    counts: rustc_hash::FxHashMap<String, f64>,
    total: f64,
    alpha: f64,
    fitted: bool,
}

#[pymethods]
impl StupidBackoff {
    #[new]
    #[pyo3(signature = (order, alpha=0.4))]
    fn new(order: usize, alpha: f64) -> Self {
        Self {
            order,
            counts: rustc_hash::FxHashMap::default(),
            total: 0.0,
            alpha,
            fitted: false,
        }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        for sentence in &sentences {
            let mut tokens = vec!["<s>".to_string(); self.order - 1];
            for word in sentence { tokens.push(word.clone()); }
            tokens.push("</s>".to_string());
            for token in &tokens {
                *self.counts.entry(token.clone()).or_insert(0.0) += 1.0;
                self.total += 1.0;
            }
        }
        self.fitted = true;
    }

    #[pyo3(signature = (word, _context=None))]
    fn score(&self, word: &str, _context: Option<Vec<String>>) -> f64 {
        if !self.fitted || self.total == 0.0 { return 0.0; }
        let count = self.counts.get(word).copied().unwrap_or(0.0);
        if count > 0.0 { count / self.total }
        else { self.alpha / (self.total + self.counts.len() as f64).max(1.0) }
    }

    fn order(&self) -> usize { self.order }
    fn fitted(&self) -> bool { self.fitted }
}
