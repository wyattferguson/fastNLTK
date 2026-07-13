//! Stemming — Rust-accelerated stemmers matching NLTK's API.
//!
//! Uses the `rust-stemmers` crate for Snowball algorithms,
//! with custom implementations for Porter, Lancaster, and others.

pub mod snowball;
pub mod porter;
pub mod lancaster;

use pyo3::prelude::*;

/// Register all stemmer classes with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<snowball::SnowballStemmer>()?;
    m.add_class::<porter::PorterStemmer>()?;
    m.add_class::<lancaster::LancasterStemmer>()?;
    Ok(())
}
