//! Segmentation metrics: windowdiff, pk.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(signature = (reference, hypothesis, k=3, boundary="1"))]
fn windowdiff(reference: &str, hypothesis: &str, k: usize, boundary: &str) -> PyResult<f64> {
    if reference.len() != hypothesis.len() {
        return Err(PyValueError::new_err("Segmentations have unequal length"));
    }
    if k > reference.len() {
        return Err(PyValueError::new_err(
            "Window width k should be smaller or equal than segmentation lengths",
        ));
    }
    if k >= reference.len() {
        // k >= n is degenerate; return 0.0 (caller should validate input)
        return Ok(0.0);
    }
    let bound = boundary.as_bytes().first().copied().unwrap_or(b'1');
    let b_ref: Vec<bool> = reference.bytes().map(|b| b == bound).collect();
    let b_hyp: Vec<bool> = hypothesis.bytes().map(|b| b == bound).collect();
    let n = reference.len();
    let mut count1: usize = b_ref[..k].iter().filter(|&&x| x).count();
    let mut count2: usize = b_hyp[..k].iter().filter(|&&x| x).count();
    let mut wd: f64 = 0.0;
    for i in 0..=(n - k) {
        if i > 0 {
            if b_ref[i - 1] {
                count1 = count1.saturating_sub(1);
            }
            if b_ref[i + k - 1] {
                count1 += 1;
            }
            if b_hyp[i - 1] {
                count2 = count2.saturating_sub(1);
            }
            if b_hyp[i + k - 1] {
                count2 += 1;
            }
        }
        wd += if count1 != count2 { 1.0 } else { 0.0 };
    }
    Ok(wd / (n - k + 1) as f64)
}

#[pyfunction(signature = (reference, hypothesis, k=None, boundary="1"))]
fn pk(reference: &str, hypothesis: &str, k: Option<usize>, boundary: &str) -> PyResult<f64> {
    if reference.len() != hypothesis.len() {
        return Err(PyValueError::new_err("Segmentations have unequal length"));
    }
    let n = reference.len();
    let k = match k {
        Some(k) => k,
        None => ((n as f64) / (n as f64).ln().ceil()).ceil() as usize,
    };
    if k >= n {
        // Degenerate input — return 0.0 (caller should validate)
        return Ok(0.0);
    }
    let bound = boundary.as_bytes().first().copied().unwrap_or(b'1');
    let b_ref: Vec<bool> = reference.bytes().map(|b| b == bound).collect();
    let b_hyp: Vec<bool> = hypothesis.bytes().map(|b| b == bound).collect();
    let half = k / 2;
    let mut errors: f64 = 0.0;
    let mut total: f64 = 0.0;
    for i in half..(n - half) {
        let start = i - half;
        let end = i + half + 1;
        let ref_bound = b_ref[start..end].iter().any(|&x| x);
        let hyp_bound = b_hyp[start..end].iter().any(|&x| x);
        if ref_bound != hyp_bound {
            errors += 1.0;
        }
        total += 1.0;
    }
    if total == 0.0 {
        return Ok(0.0);
    }
    Ok(errors / total)
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(windowdiff, m)?)?;
    m.add_function(wrap_pyfunction!(pk, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowdiff_identical() {
        let s = "000100000010";
        let wd = windowdiff(s, s, 3, "1").unwrap();
        assert!((wd - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_windowdiff_example() {
        let s1 = "000100000010";
        let s2 = "000010000100";
        let wd = windowdiff(s1, s2, 3, "1").unwrap();
        assert!((wd - 0.30).abs() < 0.01);
    }

    #[test]
    fn test_pk_example() {
        let r = "0100";
        let result = pk(r, r, Some(2), "1").unwrap();
        assert!((result - 0.0).abs() < 0.001);
    }
}
