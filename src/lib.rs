//! fastNLTK — Drop-in Rust-accelerated replacement for NLTK.
//!
//! This crate compiles to a PyO3 extension module (`fastnltk._rust`)
//! that provides native-speed implementations of NLTK's NLP algorithms.

use pyo3::prelude::*;

mod prelude;
pub mod data;
pub mod tokenize;
pub mod stem;
pub mod tag;
pub mod util;

/// The Python extension module: `fastnltk._rust`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Tokenization ─────────────────────────────────────
    tokenize::register_module(m)?;

    // ── Stemming ─────────────────────────────────────────
    stem::register_module(m)?;

    // ── Tagging ──────────────────────────────────────────
    tag::register_module(m)?;

    Ok(())
}
