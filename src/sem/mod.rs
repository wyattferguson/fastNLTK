//! Semantics — Rust-accelerated logical expression parsing and evaluation.
//!
//! Implements NLTK's `nltk.sem.logic` module:
//!   - `Expression` types (Constant, Variable, Application, Lambda, Quantifier)
//!   - Recursive descent parser for logical formulas
//!   - Substitution, simplification (beta-reduction)
//!   - Free variable extraction
//!   - Model evaluation with domain + valuation
//!
//! Sub-modules:
//!   - `expression`: Core types (Expression, Type), Display, substitution, free vars
//!   - `parse`: Tokenizer, recursive descent parser, parse_expression()
//!   - `evaluate`: Model evaluation, PyO3 bindings (fromstring, simplify, evaluate_formula)

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
