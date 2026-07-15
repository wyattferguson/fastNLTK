//! Semantics — Rust-accelerated logical expression parsing and evaluation.

pub mod evaluate;
pub mod expression;
pub mod parse;

// Re-exports for external callers (drt.rs, inference)
pub use evaluate::model_evaluate;
pub use expression::{Assignment, Expression, Individual, Type, Valuation};
pub use parse::parse_expression;

use pyo3::prelude::*;

/// Register the `sem` module with Python.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    evaluate::register_module(m)
}
