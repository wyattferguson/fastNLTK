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
    pub const fn counts(&self) -> &HashMap<SmolStr, u64> {
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
        let mut s: Vec<String> = self.counts.keys().map(ToString::to_string).collect();
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
    fn __setitem__(&mut self, sample: &str, count: u64) {
        // Set exact count, adjusting total.
        let old = self.counts.get(sample).copied().unwrap_or(0);
        if count == 0 {
            self.counts.remove(sample);
            self.total = self.total.saturating_sub(old);
        } else {
            self.counts.insert(SmolStr::new(sample), count);
            self.total = self.total.saturating_sub(old) + count;
        }
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
///
/// Stores `FreqDist` as shared Python objects so mutations via
/// `cfd[cond][sample] = value` propagate back to the distribution.
#[pyclass(name = "ConditionalFreqDist", module = "fastnltk._rust")]
pub struct ConditionalFreqDist {
    conditions: HashMap<SmolStr, Py<FreqDist>>,
}

impl Clone for ConditionalFreqDist {
    fn clone(&self) -> Self {
        // Deep-clone each FreqDist via Python so we get independent copies.
        let cloned = pyo3::Python::try_attach(|py| {
            self.conditions
                .iter()
                .map(|(k, v)| {
                    let fd: FreqDist = v.borrow(py).clone();
                    (k.clone(), pyo3::Py::new(py, fd).unwrap())
                })
                .collect()
        })
        .expect("GIL");
        Self { conditions: cloned }
    }
}

#[pymethods]
impl ConditionalFreqDist {
    #[new]
    fn new() -> Self {
        Self { conditions: HashMap::new() }
    }
    fn conditions(&self) -> Vec<String> {
        let mut conds: Vec<String> = self.conditions.keys().map(ToString::to_string).collect();
        conds.sort();
        conds
    }
    #[allow(non_snake_case)]
    fn N(&self) -> u64 {
        pyo3::Python::try_attach(|py| self.conditions.values().map(|fd| fd.borrow(py).N()).sum())
            .expect("GIL")
    }
    fn inc(&mut self, condition: &str, sample: &str) {
        pyo3::Python::try_attach(|py| {
            self.conditions
                .entry(SmolStr::new(condition))
                .or_insert_with(|| pyo3::Py::new(py, FreqDist::new(None)).unwrap())
                .borrow_mut(py)
                .inc(sample, 1);
        })
        .expect("GIL");
    }
    fn freqdist(&self, condition: &str) -> Option<FreqDist> {
        pyo3::Python::try_attach(|py| {
            self.conditions.get(condition).map(|fd| fd.borrow(py).clone())
        })
        .expect("GIL")
    }
    fn __getitem__(&self, condition: &str) -> Option<Py<FreqDist>> {
        self.conditions
            .get(condition)
            .map(|py_fd| pyo3::Python::try_attach(|py| py_fd.clone_ref(py)).expect("GIL"))
    }
    fn __contains__(&self, condition: &str) -> bool {
        self.conditions.contains_key(condition)
    }
    #[pyo3(signature = (n=None))]
    fn most_common(&self, n: Option<usize>) -> Vec<(String, Vec<(String, u64)>)> {
        pyo3::Python::try_attach(|py| {
            self.conditions
                .iter()
                .map(|(cond, fd)| (cond.to_string(), fd.borrow(py).most_common(n)))
                .collect()
        })
        .expect("GIL")
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FreqDist>()?;
    m.add_class::<ConditionalFreqDist>()?;
    m.add_class::<dist::MLEProbDist>()?;
    m.add_class::<dist::LaplaceProbDist>()?;
    Ok(())
}
