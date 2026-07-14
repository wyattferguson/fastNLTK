//! fastNLTK — Drop-in Rust-accelerated replacement for NLTK.
//!
//! This crate compiles to a PyO3 extension module (`fastnltk._rust`)
//! that provides native-speed implementations of NLTK's NLP algorithms.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;

pub mod ccg;
pub mod chat;
pub mod chunk;
pub mod classify;
pub mod cluster;
pub mod collocations;
pub mod corpus;
pub mod data;
pub mod drt;
pub mod inference;
pub mod lm;
pub mod metrics;
pub mod parse;
mod prelude;
pub mod probability;
pub mod sem;
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

    // ── CCG ──────────────────────────────────────────
    ccg::register_module(m)?;

    // ── Inference ────────────────────────────────────────
    inference::register_module(m)?;

    // ── DRT ───────────────────────────────────────────
    drt::register_module(m)?;

    // ── Chat ──────────────────────────────────────────
    chat::register_module(m)?;

    // ── Clustering ────────────────────────────────────
    cluster::register_module(m)?;

    // ── Corpus ─────────────────────────────────────────
    corpus::register_module(m)?;

    // ── Tree ─────────────────────────────────────────────
    tree::register_module(m)?;

    // ── Parsing ────────────────────────────────────────
    parse::register_module(m)?;

    // ── Semantics ──────────────────────────────────────
    sem::register_module(m)?;

    Ok(())
}
