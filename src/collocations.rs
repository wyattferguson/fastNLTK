//! Collocations — ngram collocation finders matching NLTK's API.
//!
//! Implements BigramCollocationFinder, TrigramCollocationFinder,
//! QuadgramCollocationFinder with frequency counting and scoring.

use hashbrown::HashMap as FastHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════
// CollocationFinderBase
// ═══════════════════════════════════════════════════════════

/// Internal collocation finder state shared by Bigram/Trigram/Quadgram.
struct CollocationData {
    word_fd: FastHashMap<String, u64>,
    ngram_fd: FastHashMap<Vec<String>, u64>,
    n: usize,
    min_freq: u64,
}

impl CollocationData {
    fn from_words(words: &[String], n: usize) -> Self {
        let mut word_fd: FastHashMap<String, u64> = FastHashMap::new();
        let mut ngram_fd: FastHashMap<Vec<String>, u64> = FastHashMap::new();

        for word in words {
            *word_fd.entry(word.clone()).or_insert(0) += 1;
        }

        for ngram in words.windows(n) {
            let key = ngram.to_vec();
            *ngram_fd.entry(key).or_insert(0) += 1;
        }

        CollocationData { word_fd, ngram_fd, n, min_freq: 1 }
    }

    fn apply_freq_filter(&mut self, min_freq: u64) {
        self.min_freq = min_freq;
        self.ngram_fd.retain(|_, &mut count| count >= min_freq);
    }

    #[allow(dead_code)]
    fn apply_word_filter<F>(&mut self, filter_fn: F)
    where
        F: Fn(&str) -> bool,
    {
        self.ngram_fd.retain(|ngram, _| !ngram.iter().any(|w| filter_fn(w)));
    }

    fn score_ngrams(&self, score_fn: &str) -> Vec<(Vec<String>, f64)> {
        let total_words: u64 = self.word_fd.values().sum();
        let _total_bigrams: u64 = self.ngram_fd.values().sum();
        let num_words = self.word_fd.len() as f64;

        let mut scored: Vec<(Vec<String>, f64)> = Vec::new();
        for (ngram, count) in &self.ngram_fd {
            let score = match score_fn {
                "pmi" => self.pmi(ngram, *count, total_words as f64),
                "chi_sq" => self.chi_sq(ngram, *count, total_words as f64, num_words),
                "likelihood_ratio" => self.likelihood_ratio(ngram, *count, total_words as f64),
                "raw_freq" => *count as f64,
                _ => *count as f64, // default: raw frequency
            };
            scored.push((ngram.clone(), score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn pmi(&self, ngram: &[String], count: u64, total: f64) -> f64 {
        if self.n != 2 || ngram.len() != 2 {
            return 0.0;
        }
        let w1_count = self.word_fd.get(&ngram[0]).copied().unwrap_or(1) as f64;
        let w2_count = self.word_fd.get(&ngram[1]).copied().unwrap_or(1) as f64;
        let expected = (w1_count * w2_count) / total;
        if expected <= 0.0 || count == 0 {
            return 0.0;
        }
        (count as f64 / expected).log2()
    }

    fn chi_sq(&self, ngram: &[String], count: u64, total: f64, _num_words: f64) -> f64 {
        if self.n != 2 || ngram.len() != 2 {
            return 0.0;
        }
        let w1_count = self.word_fd.get(&ngram[0]).copied().unwrap_or(1) as f64;
        let w2_count = self.word_fd.get(&ngram[1]).copied().unwrap_or(1) as f64;

        let o11 = count as f64;
        let o12 = w1_count - o11;
        let o21 = w2_count - o11;
        let o22 = total - w1_count - w2_count + o11;

        let e11 = (w1_count * w2_count) / total;
        let e12 = (w1_count * (total - w2_count)) / total;
        let e21 = ((total - w1_count) * w2_count) / total;
        let e22 = ((total - w1_count) * (total - w2_count)) / total;

        let chi = |o: f64, e: f64| -> f64 {
            if e <= 0.0 {
                0.0
            } else {
                (o - e) * (o - e) / e
            }
        };

        chi(o11, e11) + chi(o12, e12) + chi(o21, e21) + chi(o22, e22)
    }

    fn likelihood_ratio(&self, ngram: &[String], count: u64, total: f64) -> f64 {
        if self.n != 2 || ngram.len() != 2 {
            return 0.0;
        }
        let w1_count = self.word_fd.get(&ngram[0]).copied().unwrap_or(1) as f64;
        let w2_count = self.word_fd.get(&ngram[1]).copied().unwrap_or(1) as f64;

        let k = count as f64;
        let n = total;
        let p = k / n;
        let p1 = k / w1_count;
        let p2 = (w2_count - k) / (n - w1_count);

        let ll = |k: f64, n: f64, p: f64| -> f64 {
            if k <= 0.0 || n <= 0.0 || p <= 0.0 {
                return 0.0;
            }
            k * p.ln() + (n - k) * (1.0 - p).ln()
        };

        2.0 * (ll(k, w1_count, p1) + ll(w2_count - k, n - w1_count, p2)
            - ll(k, w1_count, p)
            - ll(w2_count - k, n - w1_count, p))
    }
}

// ═══════════════════════════════════════════════════════════
// BigramCollocationFinder
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "BigramCollocationFinder", module = "fastnltk._rust")]
pub struct BigramCollocationFinder {
    data: CollocationData,
}

#[pymethods]
impl BigramCollocationFinder {
    #[new]
    fn new(_word_fd: &Bound<'_, PyDict>, _ngram_fd: &Bound<'_, PyDict>) -> PyResult<Self> {
        Err(PyValueError::new_err("Use BigramCollocationFinder.from_words(words)"))
    }

    #[staticmethod]
    #[pyo3(signature = (words, window_size=2))]
    fn from_words(words: Vec<String>, window_size: usize) -> Self {
        BigramCollocationFinder { data: CollocationData::from_words(&words, window_size) }
    }

    fn score_ngrams(&self, score_fn: &str) -> Vec<(Vec<String>, f64)> {
        self.data.score_ngrams(score_fn)
    }

    #[pyo3(signature = (score_fn, n=10))]
    fn nbest(&self, score_fn: &str, n: usize) -> Vec<Vec<String>> {
        let scored = self.data.score_ngrams(score_fn);
        scored.into_iter().take(n).map(|(ngram, _)| ngram).collect()
    }

    fn apply_freq_filter(&mut self, min_freq: u64) {
        self.data.apply_freq_filter(min_freq);
    }
}

// ═══════════════════════════════════════════════════════════
// TrigramCollocationFinder
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "TrigramCollocationFinder", module = "fastnltk._rust")]
pub struct TrigramCollocationFinder {
    data: CollocationData,
}

#[pymethods]
impl TrigramCollocationFinder {
    #[new]
    fn new(_word_fd: &Bound<'_, PyDict>, _ngram_fd: &Bound<'_, PyDict>) -> PyResult<Self> {
        Err(PyValueError::new_err("Use TrigramCollocationFinder.from_words(words)"))
    }

    #[staticmethod]
    #[pyo3(signature = (words, window_size=3))]
    fn from_words(words: Vec<String>, window_size: usize) -> Self {
        TrigramCollocationFinder { data: CollocationData::from_words(&words, window_size) }
    }

    fn score_ngrams(&self, score_fn: &str) -> Vec<(Vec<String>, f64)> {
        self.data.score_ngrams(score_fn)
    }

    #[pyo3(signature = (score_fn, n=10))]
    fn nbest(&self, score_fn: &str, n: usize) -> Vec<Vec<String>> {
        let scored = self.data.score_ngrams(score_fn);
        scored.into_iter().take(n).map(|(ngram, _)| ngram).collect()
    }

    fn apply_freq_filter(&mut self, min_freq: u64) {
        self.data.apply_freq_filter(min_freq);
    }
}

// ═══════════════════════════════════════════════════════════
// QuadgramCollocationFinder
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "QuadgramCollocationFinder", module = "fastnltk._rust")]
pub struct QuadgramCollocationFinder {
    data: CollocationData,
}

#[pymethods]
impl QuadgramCollocationFinder {
    #[new]
    fn new(_word_fd: &Bound<'_, PyDict>, _ngram_fd: &Bound<'_, PyDict>) -> PyResult<Self> {
        Err(PyValueError::new_err("Use QuadgramCollocationFinder.from_words(words)"))
    }

    #[staticmethod]
    #[pyo3(signature = (words, window_size=4))]
    fn from_words(words: Vec<String>, window_size: usize) -> Self {
        QuadgramCollocationFinder { data: CollocationData::from_words(&words, window_size) }
    }

    fn score_ngrams(&self, score_fn: &str) -> Vec<(Vec<String>, f64)> {
        self.data.score_ngrams(score_fn)
    }

    #[pyo3(signature = (score_fn, n=10))]
    fn nbest(&self, score_fn: &str, n: usize) -> Vec<Vec<String>> {
        let scored = self.data.score_ngrams(score_fn);
        scored.into_iter().take(n).map(|(ngram, _)| ngram).collect()
    }

    fn apply_freq_filter(&mut self, min_freq: u64) {
        self.data.apply_freq_filter(min_freq);
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BigramCollocationFinder>()?;
    m.add_class::<TrigramCollocationFinder>()?;
    m.add_class::<QuadgramCollocationFinder>()?;
    Ok(())
}
