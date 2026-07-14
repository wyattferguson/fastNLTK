//! fastNLTK — Drop-in Rust-accelerated replacement for NLTK.
//!
//! This crate compiles to a PyO3 extension module (`fastnltk._rust`)
//! that provides native-speed implementations of NLTK's NLP algorithms.

use pyo3::prelude::*;

mod prelude;
pub mod chunk;
pub mod classify;
pub mod collocations;
pub mod corpus;
pub mod data;
pub mod lm;
pub mod metrics;
pub mod probability;
pub mod sentiment;
pub mod stem;
pub mod tag;
pub mod tokenize;
pub mod translate;
pub mod tree;
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

    // ── Probability ──────────────────────────────────────
    probability::register_module(m)?;

    // ── Metrics ──────────────────────────────────────────
    metrics::register_module(m)?;

    // ── Language Models ────────────────────────────────
    lm::register_module(m)?;

    // ── Classification ───────────────────────────────────
    classify::register_module(m)?;

    // ── Collocations ─────────────────────────────────────
    collocations::register_module(m)?;

    // ── Sentiment ────────────────────────────────────────
    sentiment::register_module(m)?;

    // ── Translation ──────────────────────────────────────
    translate::register_module(m)?;

    // ── Chunking ───────────────────────────────────────
    chunk::register_module(m)?;

    // ── Corpus ─────────────────────────────────────────
    corpus::register_module(m)?;

    // ── Tree ─────────────────────────────────────────────
    tree::register_module(m)?;

    Ok(())
}
