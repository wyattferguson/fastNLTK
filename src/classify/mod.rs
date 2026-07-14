//! Classification — Rust-accelerated classifiers matching NLTK's API.
//!
//! Implements:
//!   - `NaiveBayesClassifier` (training + prediction)
//!   - `MaxentClassifier` (GIS training + inference)
//!   - `TextCat` (language detection via whatlang)

pub mod maxent;
pub mod naivebayes;
pub mod textcat;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    naivebayes::register_module(m)?;
    textcat::register_module(m)?;
    maxent::register_module(m)?;
    Ok(())
}
