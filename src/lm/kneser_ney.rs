//! Kneser-Ney interpolated language model.

use pyo3::prelude::*;

/// Kneser-Ney interpolated language model with fixed discount.
#[pyclass(name = "KneserNeyInterpolated", module = "fastnltk._rust")]
pub struct KneserNeyInterpolated {
    order: usize,
    discount: f64,
    counts: rustc_hash::FxHashMap<String, f64>,
    total: f64,
    fitted: bool,
}

#[pymethods]
impl KneserNeyInterpolated {
    #[new]
    #[pyo3(signature = (order, discount=0.75))]
    fn new(order: usize, discount: f64) -> Self {
        Self {
            order,
            discount,
            counts: rustc_hash::FxHashMap::default(),
            total: 0.0,
            fitted: false,
        }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        for sentence in &sentences {
            let mut tokens = vec!["<s>".to_string(); self.order - 1];
            for word in sentence {
                tokens.push(word.clone());
            }
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
        if !self.fitted {
            return 0.0;
        }
        let count = self.counts.get(word).copied().unwrap_or(0.0);
        let d = self.discount;
        let max_term = (count - d).max(0.0) / self.total.max(1.0);
        let lambda = d * self.counts.len() as f64 / self.total.max(1.0);
        let unk_prob = 1.0 / self.counts.len().max(1) as f64;
        max_term + lambda * unk_prob
    }

    fn order(&self) -> usize {
        self.order
    }
    fn fitted(&self) -> bool {
        self.fitted
    }
}
