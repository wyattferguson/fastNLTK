//! Probability distribution types — MLE and Laplace smoothing.
//!
//! Implements MLEProbDist (maximum likelihood) and LaplaceProbDist
//! (add-one smoothing) matching NLTK's API.

use pyo3::prelude::*;
use crate::probability::FreqDist;

/// Maximum Likelihood Estimation probability distribution.
#[pyclass(name = "MLEProbDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct MLEProbDist {
    freqdist: FreqDist,
}

#[pymethods]
impl MLEProbDist {
    #[new]
    #[pyo3(signature = (freqdist, bins=None))]
    fn new(freqdist: FreqDist, bins: Option<usize>) -> Self { MLEProbDist { freqdist } }
    fn prob(&self, sample: &str) -> f64 {
        let n = self.freqdist.get_total();
        if n == 0 { return 0.0; }
        self.freqdist.get_count(sample) as f64 / n as f64
    }
    fn max(&self) -> Option<String> { self.freqdist.max() }
    fn freqdist(&self) -> FreqDist { self.freqdist.clone() }
    fn samples(&self) -> Vec<String> { self.freqdist.samples() }
}

/// Laplace (add-one) smoothed probability distribution.
#[pyclass(name = "LaplaceProbDist", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct LaplaceProbDist {
    freqdist: FreqDist,
    bins: usize,
}

#[pymethods]
impl LaplaceProbDist {
    #[new]
    #[pyo3(signature = (freqdist, bins=None))]
    fn new(freqdist: FreqDist, bins: Option<usize>) -> Self {
        let b = bins.unwrap_or(freqdist.num_samples());
        LaplaceProbDist { freqdist, bins: b }
    }
    fn prob(&self, sample: &str) -> f64 {
        let n = self.freqdist.get_total();
        let b = self.bins;
        let count = self.freqdist.get_count(sample);
        (count + 1) as f64 / (n + b as u64) as f64
    }
    fn max(&self) -> Option<String> { self.freqdist.max() }
    fn freqdist(&self) -> FreqDist { self.freqdist.clone() }
    fn samples(&self) -> Vec<String> { self.freqdist.samples() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::FreqDist;
    use crate::probability::ConditionalFreqDist;

    fn sample_fd() -> FreqDist {
        let mut fd = FreqDist::new(None);
        fd.update(vec!["a".into(), "b".into(), "a".into(), "c".into()]);
        fd
    }

    #[test]
    fn test_freqdist_new() {
        let fd = FreqDist::new(None);
        assert_eq!(fd.N(), 0);
        assert_eq!(fd.B(), 0);
    }

    #[test]
    fn test_freqdist_update() {
        let mut fd = FreqDist::new(None);
        fd.update(vec!["a".into(), "b".into(), "a".into()]);
        assert_eq!(fd.N(), 3);
        assert_eq!(fd.B(), 2);
        assert_eq!(fd.__getitem__("a"), 2);
    }

    #[test]
    fn test_freqdist_freq() {
        let fd = sample_fd();
        let f = fd.freq("a");
        assert!((f - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_freqdist_max() {
        let fd = sample_fd();
        assert_eq!(fd.max(), Some("a".into()));
    }

    #[test]
    fn test_freqdist_most_common() {
        let fd = sample_fd();
        let mc = fd.most_common(None);
        assert_eq!(mc[0].0, "a");
        assert_eq!(mc[0].1, 2);
    }

    #[test]
    fn test_freqdist_contains() {
        let fd = sample_fd();
        assert!(fd.__contains__("a"));
        assert!(!fd.__contains__("x"));
    }

    #[test]
    fn test_freqdist_copy_add_sub() {
        let fd = sample_fd();
        let c = fd.copy();
        assert_eq!(c.N(), fd.N());
        let fd2 = fd.__add__(&c);
        assert_eq!(fd2.__getitem__("a"), 4);
        let fd3 = fd.__sub__(&c);
        assert_eq!(fd3.N(), 0);
    }

    #[test]
    fn test_freqdist_hapaxes() {
        let fd = sample_fd();
        let hap = fd.hapaxes();
        assert!(hap.contains(&"b".into()) || hap.contains(&"c".into()));
    }

    #[test]
    fn test_cond_freqdist() {
        let mut cfd = ConditionalFreqDist::new();
        cfd.inc("cond1", "a");
        cfd.inc("cond1", "a");
        cfd.inc("cond2", "b");
        assert_eq!(cfd.N(), 3);
        assert!(cfd.__contains__("cond1"));
        assert!(!cfd.__contains__("cond3"));
    }

    #[test]
    fn test_mle_probdist() {
        let fd = sample_fd();
        let pd = MLEProbDist::new(fd, None);
        let p = pd.prob("a");
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_laplace_probdist() {
        let fd = sample_fd();
        let pd = LaplaceProbDist::new(fd, Some(4));
        let p = pd.prob("a");
        assert!((p - 0.375).abs() < 1e-9);
    }

    #[test]
    fn test_cond_freqdist_most_common() {
        let mut cfd = ConditionalFreqDist::new();
        cfd.inc("cond1", "a");
        cfd.inc("cond1", "a");
        cfd.inc("cond2", "b");
        let mc = cfd.most_common(None);
        assert_eq!(mc.len(), 2);
    }
}
