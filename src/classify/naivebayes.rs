//! NaiveBayesClassifier — Rust implementation matching NLTK's API.
//!
//! Naive Bayes classifier with:
//!   - Maximum Likelihood Estimation for P(feature|label)
//!   - Add-k smoothing (Laplace smoothing)
//!   - Log-space computation to avoid underflow
//!   - Training + prediction in compiled Rust

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

// ═══════════════════════════════════════════════════════════
// NaiveBayesClassifier
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "NaiveBayesClassifier", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct NaiveBayesClassifier {
    /// Log probability of each label: P(label)
    label_logprobs: HashMap<String, f64>,
    /// Log probability of each feature value given label: P(feature=value|label)
    feature_logprobs: HashMap<String, HashMap<String, f64>>,
    /// List of known labels
    labels: Vec<String>,
    /// List of known features
    features: Vec<String>,
    /// Smoothing parameter (add-k)
    alpha: f64,
}

#[pymethods]
impl NaiveBayesClassifier {
    #[new]
    fn new() -> Self {
        NaiveBayesClassifier {
            label_logprobs: HashMap::new(),
            feature_logprobs: HashMap::new(),
            labels: Vec::new(),
            features: Vec::new(),
            alpha: 1.0, // Laplace smoothing
        }
    }

    /// Train the classifier on labeled feature sets.
    ///
    /// `labeled_featuresets` is a list of (dict, label) pairs where
    /// dict maps feature_name → feature_value.
    #[pyo3(signature = (labeled_featuresets, alpha=1.0))]
    fn train(
        &mut self,
        py: Python<'_>,
        labeled_featuresets: &Bound<'_, PyList>,
        alpha: f64,
    ) -> PyResult<()> {
        // Extract data from Python objects first (can't pass PyList into allow_threads)
        let mut raw_data: Vec<(String, Vec<(String, String)>)> = Vec::new();
        for item in labeled_featuresets.iter() {
            let tuple = item
                .downcast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("Expected (features_dict, label) tuples"))?;
            let item0 = tuple.get_item(0)?;
            let features_dict = item0
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Expected dict as first element"))?;
            let item1 = tuple.get_item(1)?;
            let label: String = item1
                .extract()
                .map_err(|_| PyValueError::new_err("Expected string label as second element"))?;

            let mut feats = Vec::new();
            for (feat_key, feat_value) in features_dict.iter() {
                let k: String = feat_key.extract().unwrap_or_default();
                let v: String = feat_value.extract().unwrap_or_default();
                feats.push((k, v));
            }
            raw_data.push((label, feats));
        }

        py.allow_threads(|| {
            let mut label_counts: HashMap<String, u64> = HashMap::new();
            let mut feature_value_counts: HashMap<String, HashMap<String, u64>> = HashMap::new();
            let mut total = 0u64;

            for (label, feats) in &raw_data {
                *label_counts.entry(label.clone()).or_insert(0) += 1;
                total += 1;

                let inner = feature_value_counts.entry(label.clone()).or_default();
                for (feat_key, feat_value) in feats {
                    let key = format!("{feat_key}={feat_value}");
                    *inner.entry(key).or_insert(0) += 1;
                }
            }

            if total == 0 {
                return Err(PyValueError::new_err("Empty training set"));
            }

            let mut labels: Vec<String> = label_counts.keys().cloned().collect();
            labels.sort();

            let features: Vec<String> = {
                let set: std::collections::HashSet<String> =
                    feature_value_counts.values().flat_map(|m| m.keys()).cloned().collect();
                let mut v: Vec<String> = set.into_iter().collect();
                v.sort();
                v
            };

            let mut label_logprobs = HashMap::new();
            for label in &labels {
                let count = *label_counts.get(label).unwrap_or(&0);
                label_logprobs.insert(label.clone(), (count as f64 / total as f64).ln());
            }

            let mut feature_logprobs: HashMap<String, HashMap<String, f64>> = HashMap::new();
            for label in &labels {
                let mut inner = HashMap::new();
                let fc = feature_value_counts.entry(label.clone()).or_default();
                let total_feature_occurrences: u64 = fc.values().sum();
                let denom = total_feature_occurrences as f64 + alpha * features.len() as f64;

                for feat in &features {
                    let count = fc.get(feat).copied().unwrap_or(0) as f64;
                    let prob = (count + alpha) / denom;
                    inner.insert(feat.clone(), prob.ln());
                }
                feature_logprobs.insert(label.clone(), inner);
            }

            self.label_logprobs = label_logprobs;
            self.feature_logprobs = feature_logprobs;
            self.labels = labels;
            self.features = features;
            self.alpha = alpha;

            Ok(())
        })
    }

    /// Classify a feature dictionary.
    fn classify(&self, features_dict: &Bound<'_, PyDict>) -> PyResult<String> {
        let features = self.extract_feature_vector(features_dict);
        let best = self.classify_internal(&features);
        Ok(best)
    }

    /// Return the list of known labels.
    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    /// Return probabilities for each label.
    fn prob_classify(&self, features_dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, f64>> {
        let features = self.extract_feature_vector(features_dict);
        let mut scores = HashMap::new();

        for label in &self.labels {
            let prior = *self.label_logprobs.get(label).unwrap_or(&0.0f64.ln());
            let mut logprob = prior;
            if let Some(feat_probs) = self.feature_logprobs.get(label) {
                for feat in &features {
                    if let Some(p) = feat_probs.get(feat.as_str()) {
                        logprob += p;
                    }
                }
            }
            scores.insert(label.clone(), logprob);
        }

        // Normalize via softmax
        let max_logprob = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_logprob == f64::NEG_INFINITY {
            let uniform = 1.0 / self.labels.len() as f64;
            return Ok(self.labels.iter().map(|l| (l.clone(), uniform)).collect());
        }

        let exp_scores: Vec<f64> = scores.values().map(|v| (v - max_logprob).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let result: HashMap<String, f64> =
            self.labels.iter().zip(exp_scores.iter()).map(|(l, s)| (l.clone(), s / sum)).collect();
        Ok(result)
    }

    /// Show the most informative features.
    #[pyo3(signature = (n=10))]
    fn show_most_informative_features(&self, n: usize) -> Vec<String> {
        let mut scores: Vec<(String, f64)> = Vec::new();

        for feat in &self.features {
            for label_i in &self.labels {
                for label_j in &self.labels {
                    if label_i >= label_j {
                        continue;
                    }
                    let p_i = self
                        .feature_logprobs
                        .get(label_i)
                        .and_then(|m| m.get(feat))
                        .copied()
                        .unwrap_or(0.0);
                    let p_j = self
                        .feature_logprobs
                        .get(label_j)
                        .and_then(|m| m.get(feat))
                        .copied()
                        .unwrap_or(0.0);
                    let score = (p_i - p_j).abs();
                    scores.push((format!("{feat}"), score));
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);

        scores.iter().map(|(feat, _)| feat.clone()).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// Internal methods
// ═══════════════════════════════════════════════════════════

impl NaiveBayesClassifier {
    fn extract_feature_vector(&self, features_dict: &Bound<'_, PyDict>) -> Vec<String> {
        let mut features = Vec::new();
        for (key, value) in features_dict.iter() {
            let k: String = key.extract().unwrap_or_default();
            let v: String = value.extract().unwrap_or_default();
            features.push(format!("{k}={v}"));
        }
        features
    }

    fn classify_internal(&self, features: &[String]) -> String {
        let mut best_label = if self.labels.is_empty() {
            return String::new();
        } else {
            self.labels[0].clone()
        };
        let mut best_score = f64::NEG_INFINITY;

        for label in &self.labels {
            let prior = *self.label_logprobs.get(label).unwrap_or(&0.0f64.ln());
            let mut logprob = prior;
            if let Some(feat_probs) = self.feature_logprobs.get(label) {
                for feat in features {
                    // Unknown features: skip (contribute 0 in log space = multiply by 1)
                    if let Some(p) = feat_probs.get(feat.as_str()) {
                        logprob += p;
                    }
                }
            }
            if logprob > best_score || (logprob == best_score && logprob != f64::NEG_INFINITY) {
                best_score = logprob;
                best_label = label.clone();
            }
        }

        best_label
    }
}

/// Register module with Python.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NaiveBayesClassifier>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn train_test_classifier() -> NaiveBayesClassifier {
        let mut nb = NaiveBayesClassifier::new();
        // We can't easily test `train` without Python objects,
        // but we can test the classify_internal method directly
        nb.labels = vec!["pos".to_string(), "neg".to_string()];
        nb.features =
            vec!["word=great".to_string(), "word=terrible".to_string(), "word=ok".to_string()];

        // P(pos) = 0.6, P(neg) = 0.4
        nb.label_logprobs.insert("pos".to_string(), 0.6f64.ln());
        nb.label_logprobs.insert("neg".to_string(), 0.4f64.ln());

        // P(word=great|pos) = 0.7 > P(word=great|neg) = 0.2
        let mut pos_feats = HashMap::new();
        pos_feats.insert("word=great".to_string(), 0.7f64.ln());
        pos_feats.insert("word=terrible".to_string(), 0.05f64.ln());
        pos_feats.insert("word=ok".to_string(), 0.25f64.ln());
        nb.feature_logprobs.insert("pos".to_string(), pos_feats);

        let mut neg_feats = HashMap::new();
        neg_feats.insert("word=great".to_string(), 0.2f64.ln());
        neg_feats.insert("word=terrible".to_string(), 0.7f64.ln());
        neg_feats.insert("word=ok".to_string(), 0.1f64.ln());
        nb.feature_logprobs.insert("neg".to_string(), neg_feats);

        nb
    }

    #[test]
    fn test_classify_pos() {
        let nb = train_test_classifier();
        let result = nb.classify_internal(&["word=great".to_string()]);
        assert_eq!(result, "pos");
    }

    #[test]
    fn test_classify_neg() {
        let nb = train_test_classifier();
        let result = nb.classify_internal(&["word=terrible".to_string()]);
        assert_eq!(result, "neg");
    }

    #[test]
    fn test_classify_unknown_feature() {
        let nb = train_test_classifier();
        let result = nb.classify_internal(&["word=unknown".to_string()]);
        // Should fall back to prior (pos since P(pos) > P(neg))
        assert_eq!(result, "pos");
    }

    #[test]
    fn test_labels() {
        let nb = train_test_classifier();
        let labels = nb.labels();
        assert!(labels.contains(&"pos".to_string()));
        assert!(labels.contains(&"neg".to_string()));
        assert_eq!(labels.len(), 2);
    }
}
