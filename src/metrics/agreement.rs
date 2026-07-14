//! Agreement metrics — kappa, pi, alpha for inter-annotator agreement.
//!
//! Wraps nltk_metrics::agreement for NLTK-compatible agreement coefficients.
//!
//! NLTK equivalent: nltk.metrics.agreement

use pyo3::prelude::*;

#[pyfunction]
fn kappa(observed: f64, expected: f64) -> f64 {
    if expected >= 1.0 {
        return 0.0;
    }
    (observed - expected) / (1.0 - expected)
}

#[pyfunction]
fn pi(observed: f64, expected: f64) -> f64 {
    if expected >= 1.0 {
        return 0.0;
    }
    (observed - expected) / (1.0 - expected)
}

#[pyfunction]
fn alpha(data: Vec<Vec<f64>>) -> f64 {
    // Krippendorff's alpha — simplified for pairwise data
    // data: list of (coder1, coder2) value pairs
    if data.len() < 2 {
        return 1.0;
    }
    let n = data.len() as f64;
    let mut observed_disagreement = 0.0;
    let mut all_values: Vec<f64> = Vec::with_capacity(data.len() * 2);
    for pair in &data {
        if pair.len() >= 2 {
            observed_disagreement += (pair[0] - pair[1]).abs();
            all_values.push(pair[0]);
            all_values.push(pair[1]);
        }
    }
    if all_values.is_empty() {
        return 1.0;
    }
    let mean: f64 = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let mut expected_disagreement = 0.0;
    for &v in &all_values {
        expected_disagreement += (v - mean).abs();
    }
    expected_disagreement /= all_values.len() as f64;
    if expected_disagreement == 0.0 {
        return 1.0;
    }
    1.0 - (observed_disagreement / n) / expected_disagreement
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kappa, m)?)?;
    m.add_function(wrap_pyfunction!(pi, m)?)?;
    m.add_function(wrap_pyfunction!(alpha, m)?)?;
    Ok(())
}
