//! Witten-Bell interpolated language model.

use pyo3::prelude::*;

/// Witten-Bell interpolated language model.
#[pyclass(name = "WittenBellInterpolated", module = "fastnltk._rust")]
pub struct WittenBellInterpolated {
    order: usize,
    counts: rustc_hash::FxHashMap<String, f64>,
    types: f64,
    total: f64,
    fitted: bool,
}

#[pymethods]
impl WittenBellInterpolated {
    #[new]
    #[pyo3(signature = (order))]
    fn new(order: usize) -> Self {
        Self {
            order,
            counts: rustc_hash::FxHashMap::default(),
            types: 0.0,
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
                let entry = self.counts.entry(token.clone()).or_insert(0.0);
                if *entry == 0.0 {
                    self.types += 1.0;
                }
                *entry += 1.0;
                self.total += 1.0;
            }
        }
        self.fitted = true;
    }

    #[pyo3(signature = (word, _context=None))]
    fn score(&self, word: &str, _context: Option<Vec<String>>) -> f64 {
        if !self.fitted || self.total == 0.0 {
            return 0.0;
        }
        let count = self.counts.get(word).copied().unwrap_or(0.0);
        if count > 0.0 {
            count / self.total
        } else {
            1.0 / (self.total + self.types).max(1.0)
        }
    }

    const fn order(&self) -> usize {
        self.order
    }
    fn logscore(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        let s = self.score(word, context);
        if s > 0.0 { s.ln() } else { f64::NEG_INFINITY }
    }

    fn vocab_size(&self) -> usize {
        self.counts.len()
    }

    const fn fitted(&self) -> bool {
        self.fitted
    }
}
