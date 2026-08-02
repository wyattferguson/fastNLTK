//! Drop-in Rust-accelerated replacement for NLTK.

#![allow(deprecated)] // PyO3 0.29: FromPyObject for Clone #[pyclass]
#![allow(clippy::cast_precision_loss)] // NLP: counts/lengths to f64, won't exceed 2^52
#![allow(clippy::needless_pass_by_value)] // PyO3 sigs: #[pymethods] needs owned types
#![allow(clippy::unused_self)] // PyO3 #[pymethods] often don't use self

// System allocator on Linux aarch64 (manylinux GCC too old for
// -Wdate-time injected by cc crate into libmimalloc-sys).
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use mimalloc::MiMalloc;

#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
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
pub mod error;
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
    tokenize::register_module(m)?;
    stem::register_module(m)?;
    data::register_module(m)?;
    tag::register_module(m)?;
    probability::register_module(m)?;
    metrics::register_module(m)?;
    lm::register_module(m)?;
    classify::register_module(m)?;
    collocations::register_module(m)?;
    sentiment::register_module(m)?;
    translate::register_module(m)?;
    chunk::register_module(m)?;
    ccg::register_module(m)?;
    inference::register_module(m)?;
    drt::register_module(m)?;
    chat::register_module(m)?;
    cluster::register_module(m)?;
    corpus::register_module(m)?;
    tree::register_module(m)?;
    parse::register_module(m)?;
    sem::register_module(m)?;
    Ok(())
}
