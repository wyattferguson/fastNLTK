//! Frequency & probability distributions matching NLTK's API.
//!
//! Uses `SmolStr` internally for compact inlined string keys (avoids heap
//! allocation for common short words up to 22 bytes) and `hashbrown::HashMap`
//! for performance.

pub mod dist;

use hashbrown::HashMap;
use pyo3::prelude::*;
use smol_str::SmolStr;

/// A frequency distribution for a list of samples.
#[pyclass(name = "FreqDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct FreqDist {
    counts: HashMap<SmolStr, u64>,
    total: u64,
}

impl FreqDist {
    #[must_use]
    pub fn get_count(&self, sample: &str) -> u64 {
        self.counts.get(sample).copied().unwrap_or(0)
    }
    #[must_use]
    pub const fn get_total(&self) -> u64 {
        self.total
    }
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.counts.len()
    }
    #[must_use]
    pub fn counts(&self) -> &HashMap<SmolStr, u64> {
        &self.counts
    }
}

#[pymethods]
impl FreqDist {
    #[new]
    #[pyo3(signature = (samples=None))]
    fn new(samples: Option<Vec<String>>) -> Self {
        let mut fd = Self { counts: HashMap::new(), total: 0 };
        if let Some(s) = samples {
            fd.update(s);
        }
        fd
    }
    #[allow(non_snake_case)]
    const fn N(&self) -> u64 {
        self.total
    }
    #[allow(non_snake_case)]
    fn B(&self) -> usize {
        self.counts.len()
    }
    fn freq(&self, sample: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.counts.get(sample).copied().unwrap_or(0) as f64 / self.total as f64
    }
    fn max(&self) -> Option<String> {
        self.counts.iter().max_by_key(|(_, &count)| count).map(|(sample, _)| sample.to_string())
    }
    fn hapaxes(&self) -> Vec<String> {
        self.counts
            .iter()
            .filter(|(_, &count)| count == 1)
            .map(|(sample, _)| sample.to_string())
            .collect()
    }
    fn samples(&self) -> Vec<String> {
        let mut s: Vec<String> = self.counts.keys().map(|k| k.to_string()).collect();
        s.sort();
        s
    }
    fn update(&mut self, samples: Vec<String>) {
        for sample in samples {
            *self.counts.entry(SmolStr::new(&sample)).or_insert(0) += 1;
            self.total += 1;
        }
    }
    fn inc(&mut self, sample: &str, count: u64) {
        *self.counts.entry(SmolStr::new(sample)).or_insert(0) += count;
        self.total += count;
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn __len__(&self) -> usize {
        self.counts.len()
    }
    fn __repr__(&self) -> String {
        format!("<FreqDist with {} samples and {} outcomes>", self.counts.len(), self.total)
    }
    fn __getitem__(&self, sample: &str) -> u64 {
        self.counts.get(sample).copied().unwrap_or(0)
    }
    fn __contains__(&self, sample: &str) -> bool {
        self.counts.contains_key(sample)
    }
    fn __add__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (sample, count) in &other.counts {
            *result.counts.entry(sample.clone()).or_insert(0) += count;
            result.total += count;
        }
        result
    }
    fn __sub__(&self, other: &Self) -> Self {
        let mut result = Self { counts: HashMap::new(), total: 0 };
        for (sample, count) in &self.counts {
            let new_count =
                count.saturating_sub(other.counts.get(sample.as_str()).copied().unwrap_or(0));
            if new_count > 0 {
                result.counts.insert(sample.clone(), new_count);
                result.total += new_count;
            }
        }
        result
    }
    #[pyo3(signature = (n=None))]
    fn most_common(&self, n: Option<usize>) -> Vec<(String, u64)> {
        let mut items: Vec<_> = self.counts.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        match n {
            Some(n) => items.into_iter().take(n).collect(),
            None => items,
        }
    }
    #[pyo3(signature = (n=None))]
    fn tabulate(&self, n: Option<usize>) {
        let items = self.most_common(n);
        for (sample, count) in &items {
            println!("{sample:<20} {count}");
        }
    }
    #[pyo3(signature = (n=None))]
    fn plot(&self, n: Option<usize>) -> PyResult<()> {
        let items = self.most_common(n);
        let max_count = items.first().map_or(1, |(_, c)| *c).max(1);
        for (sample, count) in &items {
            let bar_width = (*count as f64 / max_count as f64 * 40.0) as usize;
            println!("{sample:<15} |{} {}", "#".repeat(bar_width), count);
        }
        Ok(())
    }
}

/// A conditional frequency distribution: {condition: `FreqDist`}.
#[pyclass(name = "ConditionalFreqDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct ConditionalFreqDist {
    conditions: HashMap<SmolStr, FreqDist>,
}

#[pymethods]
impl ConditionalFreqDist {
    #[new]
    fn new() -> Self {
        Self { conditions: HashMap::new() }
    }
    fn conditions(&self) -> Vec<String> {
        let mut conds: Vec<String> = self.conditions.keys().map(|k| k.to_string()).collect();
        conds.sort();
        conds
    }
    #[allow(non_snake_case)]
    fn N(&self) -> u64 {
        self.conditions.values().map(FreqDist::N).sum()
    }
    fn inc(&mut self, condition: &str, sample: &str) {
        self.conditions
            .entry(SmolStr::new(condition))
            .or_insert_with(|| FreqDist::new(None))
            .inc(sample, 1);
    }
    fn freqdist(&self, condition: &str) -> Option<FreqDist> {
        self.conditions.get(condition).cloned()
    }
    fn __getitem__(&self, condition: &str) -> Option<FreqDist> {
        self.freqdist(condition)
    }
    fn __contains__(&self, condition: &str) -> bool {
        self.conditions.contains_key(condition)
    }
    #[pyo3(signature = (n=None))]
    fn most_common(&self, n: Option<usize>) -> Vec<(String, Vec<(String, u64)>)> {
        self.conditions.iter().map(|(cond, fd)| (cond.to_string(), fd.most_common(n))).collect()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FreqDist>()?;
    m.add_class::<ConditionalFreqDist>()?;
    m.add_class::<dist::MLEProbDist>()?;
    m.add_class::<dist::LaplaceProbDist>()?;
    Ok(())
}
