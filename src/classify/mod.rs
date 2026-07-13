//! Classification — Rust-accelerated classifiers matching NLTK's API.
//!
//! Implements NaiveBayesClassifier and MaxentClassifier
//! with training and prediction in compiled Rust.

pub mod naivebayes;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    naivebayes::register_module(m)?;
    Ok(())
}
