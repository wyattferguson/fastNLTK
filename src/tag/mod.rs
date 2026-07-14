//! POS tagging — Rust-accelerated taggers matching NLTK's API.
//!
//! Implements the averaged perceptron tagger from NLTK,
//! loading weights from NLTK's trained model pickle.

pub mod perceptron;
pub mod tnt;

use pyo3::prelude::*;

/// Register all tagger classes with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<perceptron::PerceptronTagger>()?;
    m.add_class::<tnt::TnT>()?;
    Ok(())
}
