//! Scoring metrics: precision, recall, `f_measure`, `edit_distance`, `jaro/jaro_winkler` similarity,

use pyo3::prelude::*;

#[pyfunction(signature = (reference, test))]
fn precision(reference: Vec<String>, test: Vec<String>) -> f64 {
    if test.is_empty() {
        return 0.0;
    }
    let ref_set: std::collections::HashSet<String> = reference.into_iter().collect();
    let correct = test.iter().filter(|t| ref_set.contains(*t)).count();
    correct as f64 / test.len() as f64
}

#[pyfunction(signature = (reference, test))]
fn recall(reference: Vec<String>, test: Vec<String>) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let ref_set: std::collections::HashSet<String> = reference.into_iter().collect();
    let test_set: std::collections::HashSet<String> = test.into_iter().collect();
    let correct = ref_set.intersection(&test_set).count();
    correct as f64 / ref_set.len() as f64
}

fn precision_ref(reference: &std::collections::HashSet<String>, test: &[String]) -> f64 {
    if test.is_empty() {
        return 0.0;
    }
    let correct = test.iter().filter(|t| reference.contains(*t)).count();
    correct as f64 / test.len() as f64
}

fn recall_ref(
    reference: &std::collections::HashSet<String>,
    test_set: &std::collections::HashSet<String>,
) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }
    let correct = reference.intersection(test_set).count();
    correct as f64 / reference.len() as f64
}

#[pyfunction(signature = (reference, test, alpha=0.5))]
fn f_measure(reference: Vec<String>, test: Vec<String>, alpha: f64) -> f64 {
    if reference.is_empty() || test.is_empty() {
        return 0.0;
    }
    let ref_set: std::collections::HashSet<String> = reference.into_iter().collect();
    let test_set: std::collections::HashSet<String> = test.into_iter().collect();
    let p = precision_ref(&ref_set, &test_set.iter().cloned().collect::<Vec<_>>());
    let r = recall_ref(&ref_set, &test_set);
    if p + r == 0.0 {
        return 0.0;
    }
    1.0 / (alpha / p + (1.0 - alpha) / r)
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(precision, m)?)?;
    m.add_function(wrap_pyfunction!(recall, m)?)?;
    m.add_function(wrap_pyfunction!(f_measure, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_precision() {
        let p = precision(vec!["a".into(), "b".into()], vec!["a".into(), "c".into()]);
        assert!((p - 0.5).abs() < 0.001);
    }
    #[test]
    fn test_f_measure() {
        let f = f_measure(vec!["a".into(), "b".into()], vec!["a".into()], 0.5);
        assert!(f > 0.0);
    }
}
