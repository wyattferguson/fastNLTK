//! Spearman rank correlation coefficient.
//!
//! NLTK equivalent: nltk.metrics.spearman

use pyo3::prelude::*;

#[pyfunction]
fn spearman(x: Vec<f64>, y: Vec<f64>) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    // Convert to ranks
    let n = x.len();
    let mut x_ranks: Vec<(usize, f64)> = x.into_iter().enumerate().collect();
    let mut y_ranks: Vec<(usize, f64)> = y.into_iter().enumerate().collect();
    x_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    y_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut rank_x = vec![0.0_f64; n];
    let mut rank_y = vec![0.0_f64; n];
    for (i, (idx, _)) in x_ranks.iter().enumerate() {
        rank_x[*idx] = i as f64;
    }
    for (i, (idx, _)) in y_ranks.iter().enumerate() {
        rank_y[*idx] = i as f64;
    }

    // Spearman's rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    let d_sq: f64 = rank_x
        .iter()
        .zip(&rank_y)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    1.0 - (6.0 * d_sq) / (n as f64 * ((n * n - 1) as f64))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spearman, m)?)?;
    Ok(())
}
