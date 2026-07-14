//! Structured error types for fastNLTK.
//!
//! Replaces 64 raw `PyValueError::new_err(format!(...))` sites with
//! a single typed error enum. Uses `thiserror` for derive macros.

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

/// All errors that fastNLTK can return to Python callers.
#[derive(Debug, Error)]
pub enum FastNltkError {
    #[error("invalid grammar: expected '->' in line '{0}'")]
    GrammarParse(String),

    #[error("empty input")]
    EmptyInput,

    #[error("input too long ({0} words, max {1})")]
    InputTooLong(usize, usize),

    #[error("model not trained")]
    NotTrained,

    #[error("invalid category: {0}")]
    InvalidCategory(String),

    #[error("no parse found")]
    NoParse,

    #[error("invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("segmentations have unequal length ({0} vs {1})")]
    UnequalLength(usize, usize),

    #[error("window width {0} exceeds segmentation length {1}")]
    WindowTooLarge(usize, usize),

    #[error("num_clusters ({0}) exceeds vector count ({1})")]
    TooManyClusters(usize, usize),

    #[error("clusterer not fitted")]
    NotFitted,

    #[error("HMM error: {0}")]
    HmmError(String),

    #[error("{0}")]
    Other(String),
}

impl From<FastNltkError> for PyErr {
    fn from(e: FastNltkError) -> Self {
        PyValueError::new_err(e.to_string())
    }
}
