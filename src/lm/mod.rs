//! Language models — Rust-accelerated ngram language models.

pub mod kneser_ney;
pub mod stupid_backoff;
pub mod witten_bell;

use pyo3::prelude::*;
use rand::seq::IndexedRandom;
use rustc_hash::FxHashMap;

/// Shared ngram counts tracker.
#[derive(Clone)]
struct NgramCounts {
    order: usize,
    /// Unigram counts: word → count
    unigrams: FxHashMap<String, f64>,
    /// Bigram+ counts: `context_key` → (word → count)
    context_counts: FxHashMap<String, FxHashMap<String, f64>>,
    total_tokens: f64,
    vocab: Vec<String>,
}

impl NgramCounts {
    fn new(order: usize) -> Self {
        Self {
            order,
            unigrams: FxHashMap::default(),
            context_counts: FxHashMap::default(),
            total_tokens: 0.0,
            vocab: Vec::new(),
        }
    }

    fn fit(&mut self, sentences: &[Vec<String>]) {
        let mut seen = rustc_hash::FxHashSet::default();
        seen.insert("<s>".to_string());
        seen.insert("</s>".to_string());
        self.vocab.push("<s>".to_string());
        self.vocab.push("</s>".to_string());
        for sentence in sentences {
            let pad = self.order.saturating_sub(1);
            let padded: Vec<&str> = std::iter::repeat_n("<s>", pad)
                .chain(sentence.iter().map(String::as_str))
                .chain(std::iter::once("</s>"))
                .collect();
            for i in pad..padded.len() {
                let word = padded[i];
                *self.unigrams.entry(word.to_string()).or_insert(0.0) += 1.0;
                self.total_tokens += 1.0;
                if seen.insert(word.to_string()) {
                    self.vocab.push(word.to_string());
                }
                let ctx_start = i.saturating_sub(self.order.saturating_sub(1));
                let ctx = padded[ctx_start..i].join(" ");
                *self
                    .context_counts
                    .entry(ctx)
                    .or_default()
                    .entry(word.to_string())
                    .or_insert(0.0) += 1.0;
            }
        }
    }

    fn context_key(&self, context: &[String]) -> String {
        let pad = self.order.saturating_sub(1 + context.len());
        let mut parts: Vec<&str> = vec!["<s>"; pad];
        parts.extend(context.iter().map(String::as_str));
        parts.join(" ")
    }
}

// ── MLE ────────────────────────────────────────────────────────────────────

/// Maximum Likelihood Estimation language model.
#[pyclass(name = "MLE", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct MLE {
    counts: NgramCounts,
    fitted: bool,
}

#[pymethods]
impl MLE {
    #[new]
    fn new(order: usize) -> Self {
        Self { counts: NgramCounts::new(order), fitted: false }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.counts.fit(&sentences);
        self.fitted = true;
    }

    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        if !self.fitted {
            return 0.0;
        }
        let ctx_key = self.counts.context_key(&context.unwrap_or_default());
        if let Some(ctx) = self.counts.context_counts.get(&ctx_key) {
            if let Some(&count) = ctx.get(word) {
                let total: f64 = ctx.values().sum();
                return if total > 0.0 { count / total } else { 0.0 };
            }
        }
        0.0
    }

    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        let s = self.score(word, context);
        if s > 0.0 {
            s.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        let _ = random_seed;
        if !self.fitted || num_words == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(num_words);
        let mut context: Vec<String> = text_seed.unwrap_or_default();
        for _ in 0..num_words {
            let ctx_key = self.counts.context_key(&context);
            let word = self
                .counts
                .context_counts
                .get(&ctx_key)
                .and_then(|ctx| {
                    let keys: Vec<&String> = ctx.keys().collect();
                    keys.choose(&mut rng).map(|s| (*s).clone())
                })
                .or_else(|| self.counts.vocab.choose(&mut rng).cloned())
                .unwrap_or_else(|| "</s>".to_string());
            result.push(word.clone());
            context.push(word);
            if context.len() >= self.counts.order {
                context.remove(0);
            }
        }
        result
    }

    const fn order(&self) -> usize {
        self.counts.order
    }

    const fn vocab_size(&self) -> usize {
        self.counts.vocab.len()
    }

    const fn fitted(&self) -> bool {
        self.fitted
    }
}

// ── Laplace ────────────────────────────────────────────────────────────────

/// Laplace (add-one) smoothed language model.
#[pyclass(name = "Laplace", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct Laplace {
    counts: NgramCounts,
    fitted: bool,
}

#[pymethods]
impl Laplace {
    #[new]
    fn new(order: usize) -> Self {
        Self { counts: NgramCounts::new(order), fitted: false }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.counts.fit(&sentences);
        self.fitted = true;
    }

    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        if !self.fitted {
            return 0.0;
        }
        let v = self.counts.vocab.len().max(1) as f64;
        let ctx_key = self.counts.context_key(&context.unwrap_or_default());
        let (count, total) = self
            .counts
            .context_counts
            .get(&ctx_key)
            .map_or((0.0, 0.0), |ctx| {
                (ctx.get(word).copied().unwrap_or(0.0), ctx.values().sum::<f64>())
            });
        (count + 1.0) / (total + v)
    }

    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        let s = self.score(word, context);
        if s > 0.0 {
            s.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        let _ = random_seed;
        if !self.fitted || num_words == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(num_words);
        let mut context: Vec<String> = text_seed.unwrap_or_default();
        for _ in 0..num_words {
            let ctx_key = self.counts.context_key(&context);
            let word = self
                .counts
                .context_counts
                .get(&ctx_key)
                .and_then(|ctx| {
                    let keys: Vec<&String> = ctx.keys().collect();
                    keys.choose(&mut rng).map(|s| (*s).clone())
                })
                .or_else(|| self.counts.vocab.choose(&mut rng).cloned())
                .unwrap_or_else(|| "</s>".to_string());
            result.push(word.clone());
            context.push(word);
            if context.len() >= self.counts.order {
                context.remove(0);
            }
        }
        result
    }

    const fn order(&self) -> usize {
        self.counts.order
    }

    const fn vocab_size(&self) -> usize {
        self.counts.vocab.len()
    }

    const fn fitted(&self) -> bool {
        self.fitted
    }
}

// ── Lidstone ───────────────────────────────────────────────────────────────

/// Lidstone-smoothed language model (add-gamma smoothing).
#[pyclass(name = "Lidstone", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct Lidstone {
    counts: NgramCounts,
    gamma: f64,
    fitted: bool,
}

#[pymethods]
impl Lidstone {
    #[new]
    #[pyo3(signature = (order, gamma))]
    fn new(order: usize, gamma: f64) -> Self {
        Self { counts: NgramCounts::new(order), gamma, fitted: false }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) {
        self.counts.fit(&sentences);
        self.fitted = true;
    }

    #[pyo3(signature = (word, context=None))]
    fn score(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        if !self.fitted {
            return 0.0;
        }
        let v = self.counts.vocab.len().max(1) as f64;
        let ctx_key = self.counts.context_key(&context.unwrap_or_default());
        let (count, total) = self
            .counts
            .context_counts
            .get(&ctx_key)
            .map_or((0.0, 0.0), |ctx| {
                (ctx.get(word).copied().unwrap_or(0.0), ctx.values().sum::<f64>())
            });
        (count + self.gamma) / self.gamma.mul_add(v, total)
    }

    #[pyo3(signature = (word, context=None))]
    fn logscore(&self, word: &str, context: Option<Vec<String>>) -> f64 {
        let s = self.score(word, context);
        if s > 0.0 {
            s.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    #[pyo3(signature = (num_words, text_seed=None, random_seed=None))]
    fn generate(
        &self,
        num_words: usize,
        text_seed: Option<Vec<String>>,
        random_seed: Option<u64>,
    ) -> Vec<String> {
        let _ = random_seed;
        if !self.fitted || num_words == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(num_words);
        let mut context: Vec<String> = text_seed.unwrap_or_default();
        for _ in 0..num_words {
            let ctx_key = self.counts.context_key(&context);
            let word = self
                .counts
                .context_counts
                .get(&ctx_key)
                .and_then(|ctx| {
                    let keys: Vec<&String> = ctx.keys().collect();
                    keys.choose(&mut rng).map(|s| (*s).clone())
                })
                .or_else(|| self.counts.vocab.choose(&mut rng).cloned())
                .unwrap_or_else(|| "</s>".to_string());
            result.push(word.clone());
            context.push(word);
            if context.len() >= self.counts.order {
                context.remove(0);
            }
        }
        result
    }

    const fn order(&self) -> usize {
        self.counts.order
    }

    const fn vocab_size(&self) -> usize {
        self.counts.vocab.len()
    }

    const fn fitted(&self) -> bool {
        self.fitted
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
        let model = MLE::new(2);
        assert_eq!(model.order(), 2);
        assert!(!model.fitted());
    }
    #[test]
    fn test_mle_fit_and_score() {
        let mut model = MLE::new(2);
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let score = model.score("cat", Some(vec!["the".into()]));
        assert!((score - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_mle_logscore() {
        let mut model = MLE::new(2);
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let logscore = model.logscore("cat", Some(vec!["the".into()]));
        assert!(logscore.is_finite() && logscore < 0.0);
    }
    #[test]
    fn test_mle_generate() {
        let mut model = MLE::new(2);
        model.fit(vec![
            vec!["the".into(), "cat".into(), "sat".into()],
            vec!["the".into(), "dog".into(), "ran".into()],
        ]);
        let words = model.generate(3, Some(vec!["the".into()]), Some(42));
        assert!(!words.is_empty() && words.len() <= 3);
    }
    #[test]
    fn test_mle_vocab_size() {
        let mut model = MLE::new(2);
        model.fit(vec![vec!["the".into(), "cat".into()]]);
        assert!(model.vocab_size() >= 4);
    }
    #[test]
    fn test_lidstone_create() {
        let model = Lidstone::new(2, 0.5);
        assert_eq!(model.order(), 2);
    }
    #[test]
    fn test_lidstone_fit_and_score() {
        let mut model = Lidstone::new(2, 0.5);
        model.fit(vec![vec!["the".into(), "cat".into(), "sat".into()]]);
        let score = model.score("cat", Some(vec!["the".into()]));
        assert!(score > 0.0 && score <= 1.0);
    }
    #[test]
    fn test_laplace_create() {
        let model = Laplace::new(2);
        assert_eq!(model.order(), 2);
    }
    #[test]
    fn test_laplace_fit_and_score() {
        let mut model = Laplace::new(2);
        model.fit(vec![vec!["the".into(), "cat".into(), "sat".into()]]);
        let score = model.score("dog", Some(vec!["the".into()]));
        assert!(score > 0.0);
    }
    #[test]
    fn test_oov_score() {
        let mut model = MLE::new(2);
        model.fit(vec![vec!["the".into(), "cat".into()]]);
        let score = model.score("xyzzy", Some(vec!["the".into()]));
        assert!(score >= 0.0);
    }
}
