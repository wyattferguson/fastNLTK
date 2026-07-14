//! Frequency & probability distributions matching NLTK's API.
//!
//! Implements FreqDist, ConditionalFreqDist, and ProbDist types
//! with Rust-accelerated operations.

use hashbrown::HashMap;

use pyo3::prelude::*;

// ═══════════════════════════════════════════════════════════
// FreqDist
// ═══════════════════════════════════════════════════════════

/// A frequency distribution for a list of samples.
///
/// Matches NLTK's `nltk.probability.FreqDist` API.
/// Internally uses `hashbrown::HashMap` for O(1) operations.
#[pyclass(name = "FreqDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct FreqDist {
    counts: HashMap<String, u64>,
    total: u64,
}

// Non-pymethods interface for Rust-to-Rust usage
impl FreqDist {
    pub fn get_count(&self, sample: &str) -> u64 {
        self.counts.get(sample).copied().unwrap_or(0)
    }
    pub fn get_total(&self) -> u64 {
        self.total
    }
    pub fn num_samples(&self) -> usize {
        self.counts.len()
    }
}

#[pymethods]
impl FreqDist {
    #[new]
    #[pyo3(signature = (samples=None))]
    fn new(samples: Option<Vec<String>>) -> Self {
        let mut fd = FreqDist {
            counts: HashMap::new(),
            total: 0,
        };
        if let Some(s) = samples {
            fd.update(s);
        }
        fd
    }

    /// Total number of sample occurrences (sum of all counts).
    #[allow(non_snake_case)]
    fn N(&self) -> u64 {
        self.total
    }

    /// Number of unique samples (B = "bins").
    #[allow(non_snake_case)]
    fn B(&self) -> usize {
        self.counts.len()
    }

    /// Return the frequency of a given sample (count / total).
    fn freq(&self, sample: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let count = self.counts.get(sample).copied().unwrap_or(0);
        count as f64 / self.total as f64
    }

    /// Return the sample with the greatest frequency.
    fn max(&self) -> Option<String> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(sample, _)| sample.clone())
    }

    /// Return samples that occur only once (hapax legomena).
    fn hapaxes(&self) -> Vec<String> {
        self.counts
            .iter()
            .filter(|(_, &count)| count == 1)
            .map(|(sample, _)| sample.clone())
            .collect()
    }

    /// Return all unique samples.
    fn samples(&self) -> Vec<String> {
        let mut samples: Vec<String> = self.counts.keys().cloned().collect();
        samples.sort();
        samples
    }

    /// Update counts with additional samples.
    fn update(&mut self, samples: Vec<String>) {
        for sample in samples {
            *self.counts.entry(sample).or_insert(0) += 1;
            self.total += 1;
        }
    }

    /// Return the n most common (sample, count) pairs.
    #[pyo3(signature = (n=None))]
    fn most_common(&self, n: Option<usize>) -> Vec<(String, u64)> {
        let mut items: Vec<(String, u64)> = self
            .counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        if let Some(n) = n {
            items.truncate(n);
        }
        items
    }

    /// Return the count for a sample.
    fn __getitem__(&self, sample: &str) -> u64 {
        self.counts.get(sample).copied().unwrap_or(0)
    }

    /// Return the number of unique samples.
    fn __len__(&self) -> usize {
        self.counts.len()
    }

    /// Check if a sample is in the distribution.
    fn __contains__(&self, sample: &str) -> bool {
        self.counts.contains_key(sample)
    }

    /// Iterate over samples.
    fn __iter__(&self) -> Vec<String> {
        self.samples()
    }

    /// Return all sample keys.
    fn keys(&self) -> Vec<String> {
        self.samples()
    }

    /// Return all counts.
    fn values(&self) -> Vec<u64> {
        self.counts.values().copied().collect()
    }

    /// Return all (sample, count) pairs.
    fn items(&self) -> Vec<(String, u64)> {
        self.counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Combine two FreqDists by adding counts.
    fn __add__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (sample, count) in &other.counts {
            *result.counts.entry(sample.clone()).or_insert(0) += count;
        }
        result.total = self.total + other.total;
        result
    }

    /// Create a copy.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Python string representation.
    fn __repr__(&self) -> String {
        let items = self.most_common(Some(10));
        let _item_strs: Vec<String> = items
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect();
        format!("<FreqDist with {} samples and {} outcomes>", self.N(), self.B())
    }

    /// Increment count for a single sample.
    #[pyo3(signature = (sample, count=1))]
    fn inc(&mut self, sample: &str, count: u64) {
        *self.counts.entry(sample.to_string()).or_insert(0) += count;
        self.total += count;
    }

    /// Return count for sample (Python dict-style get).
    #[pyo3(signature = (sample, default=None))]
    fn get(&self, sample: &str, default: Option<u64>) -> u64 {
        self.counts.get(sample).copied().unwrap_or(default.unwrap_or(0))
    }

    /// Subtract another FreqDist (only keep positive counts).
    fn __sub__(&self, other: &Self) -> Self {
        let mut result = FreqDist::new(None);
        for (sample, count) in &self.counts {
            let other_count = other.counts.get(sample).copied().unwrap_or(0);
            let new_count = count.saturating_sub(other_count);
            if new_count > 0 {
                result.counts.insert(sample.clone(), new_count);
                result.total += new_count;
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════
// ConditionalFreqDist
// ═══════════════════════════════════════════════════════════

/// A frequency distribution conditioned on another variable.
///
/// Matches NLTK's `nltk.probability.ConditionalFreqDist`.
/// Maps conditions to their own FreqDist instances.
#[pyclass(name = "ConditionalFreqDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct ConditionalFreqDist {
    conditions: HashMap<String, FreqDist>,
}

#[pymethods]
impl ConditionalFreqDist {
    #[new]
    fn new() -> Self {
        ConditionalFreqDist {
            conditions: HashMap::new(),
        }
    }

    /// Get the FreqDist for a condition.
    fn __getitem__(&self, condition: &str) -> Option<FreqDist> {
        self.conditions.get(condition).cloned()
    }

    /// Return all known conditions.
    fn conditions(&self) -> Vec<String> {
        let mut conds: Vec<String> = self.conditions.keys().cloned().collect();
        conds.sort();
        conds
    }

    /// Return the total number of samples across all conditions.
    #[allow(non_snake_case)]
    fn N(&self) -> u64 {
        self.conditions.values().map(|fd| fd.N()).sum()
    }

    /// Add a (condition, sample) pair.
    fn inc(&mut self, condition: &str, sample: &str) {
        self.conditions
            .entry(condition.to_string())
            .or_insert_with(|| FreqDist::new(None))
            .inc(sample, 1);
    }

    /// Return the number of conditions.
    fn __len__(&self) -> usize {
        self.conditions.len()
    }

    /// Tabulate the conditional frequency distribution.
    /// (Python-side uses nltk for display, but we provide data)
    fn conditions_and_samples(&self) -> Vec<(String, Vec<(String, u64)>)> {
        self.conditions
            .iter()
            .map(|(cond, fd)| (cond.clone(), fd.most_common(None)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ConditionalFreqDist with {} conditions>",
            self.conditions.len()
        )
    }
}

// ═══════════════════════════════════════════════════════════
// ProbDist types
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "MLEProbDist", module = "fastnltk._rust")]
pub struct MLEProbDist {
    freqdist: FreqDist,
    bins: usize,
}

#[pymethods]
impl MLEProbDist {
    #[new]
    #[pyo3(signature = (freqdist, bins=0))]
    fn new(freqdist: FreqDist, bins: usize) -> Self {
        MLEProbDist { freqdist, bins }
    }

    fn prob(&self, sample: &str) -> f64 {
        let n = self.freqdist.get_total();
        if n == 0 { return 0.0; }
        self.freqdist.get_count(sample) as f64 / n as f64
    }

    fn max(&self) -> Option<String> {
        self.freqdist.max()
    }

    fn freqdist(&self) -> FreqDist {
        self.freqdist.clone()
    }

    fn samples(&self) -> Vec<String> {
        self.freqdist.samples()
    }
}

#[pyclass(name = "LaplaceProbDist", module = "fastnltk._rust")]
pub struct LaplaceProbDist {
    freqdist: FreqDist,
    bins: usize,
}

#[pymethods]
impl LaplaceProbDist {
    #[new]
    #[pyo3(signature = (freqdist, bins=0))]
    fn new(freqdist: FreqDist, bins: usize) -> Self {
        LaplaceProbDist { freqdist, bins }
    }

    fn prob(&self, sample: &str) -> f64 {
        let n = self.freqdist.get_total();
        let b = if self.bins > 0 { self.bins } else { self.freqdist.num_samples() };
        if n == 0 { return 1.0 / b.max(1) as f64; }
        (self.freqdist.get_count(sample) + 1) as f64 / (n as f64 + b.max(1) as f64)
    }

    fn max(&self) -> Option<String> {
        self.freqdist.max()
    }

    fn freqdist(&self) -> FreqDist {
        self.freqdist.clone()
    }

    fn samples(&self) -> Vec<String> {
        self.freqdist.samples()
    }
}

/// Register the module with Python.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FreqDist>()?;
    m.add_class::<ConditionalFreqDist>()?;
    m.add_class::<MLEProbDist>()?;
    m.add_class::<LaplaceProbDist>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dist() -> FreqDist {
        let mut fd = FreqDist::new(None);
        fd.update(vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
        ]);
        fd
    }

    #[test]
    fn test_empty() {
        let fd = FreqDist::new(None);
        assert_eq!(fd.N(), 0);
        assert_eq!(fd.B(), 0);
        assert!(fd.hapaxes().is_empty());
        assert!(fd.max().is_none());
    }

    #[test]
    fn test_counts() {
        let fd = sample_dist();
        assert_eq!(fd.N(), 6);
        assert_eq!(fd.B(), 3);
        assert_eq!(fd.__getitem__("a"), 3);
        assert_eq!(fd.__getitem__("b"), 2);
        assert_eq!(fd.__getitem__("c"), 1);
        assert_eq!(fd.__getitem__("z"), 0);
    }

    #[test]
    fn test_freq() {
        let fd = sample_dist();
        assert!((fd.freq("a") - 0.5).abs() < 1e-9);
        assert!((fd.freq("b") - 1.0 / 3.0).abs() < 1e-9);
        assert!((fd.freq("c") - 1.0 / 6.0).abs() < 1e-9);
        assert!((fd.freq("z") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_max() {
        let fd = sample_dist();
        assert_eq!(fd.max(), Some("a".to_string()));
    }

    #[test]
    fn test_hapaxes() {
        let fd = sample_dist();
        assert_eq!(fd.hapaxes(), vec!["c"]);
    }

    #[test]
    fn test_most_common() {
        let fd = sample_dist();
        let mc = fd.most_common(None);
        assert_eq!(mc[0], ("a".to_string(), 3));
        assert_eq!(mc[1], ("b".to_string(), 2));
        assert_eq!(mc[2], ("c".to_string(), 1));

        let top2 = fd.most_common(Some(2));
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn test_update() {
        let mut fd = FreqDist::new(None);
        fd.update(vec!["x".to_string(), "x".to_string(), "y".to_string()]);
        assert_eq!(fd.N(), 3);
        assert_eq!(fd.__getitem__("x"), 2);
    }

    #[test]
    fn test_contains() {
        let fd = sample_dist();
        assert!(fd.__contains__("a"));
        assert!(!fd.__contains__("z"));
    }

    #[test]
    fn test_add() {
        let fd1 = sample_dist();
        let mut fd2 = FreqDist::new(None);
        fd2.update(vec!["a".to_string(), "d".to_string()]);
        let fd3 = fd1.__add__(&fd2);
        assert_eq!(fd3.N(), 8);
        assert_eq!(fd3.__getitem__("a"), 4);
        assert_eq!(fd3.__getitem__("d"), 1);
    }

    #[test]
    fn test_sub() {
        let fd1 = sample_dist();
        let mut fd2 = FreqDist::new(None);
        fd2.update(vec!["a".to_string()]);
        let fd3 = fd1.__sub__(&fd2);
        assert_eq!(fd3.__getitem__("a"), 2);
        assert_eq!(fd3.__getitem__("b"), 2);
        assert_eq!(fd3.__getitem__("c"), 1);
    }

    #[test]
    fn test_inc() {
        let mut fd = FreqDist::new(None);
        fd.inc("x", 5);
        assert_eq!(fd.__getitem__("x"), 5);
        assert_eq!(fd.N(), 5);
    }
}
