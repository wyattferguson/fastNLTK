//! Jaccard, binary, and related distance metrics.

use pyo3::prelude::*;
use hashbrown::HashSet;

#[pyfunction(signature = (s1, s2))]
fn jaccard_distance(s1: Vec<String>, s2: Vec<String>) -> f64 {
    let set1: HashSet<&str> = s1.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = s2.iter().map(|s| s.as_str()).collect();
    let union_len = set1.union(&set2).count();
    if union_len == 0 {
        return 0.0;
    }
    let intersection_len = set1.intersection(&set2).count();
    1.0 - (intersection_len as f64 / union_len as f64)
}

#[pyfunction(signature = (s1, s2))]
fn binary_distance(s1: Vec<String>, s2: Vec<String>) -> f64 {
    let set1: HashSet<&str> = s1.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = s2.iter().map(|s| s.as_str()).collect();
    if set1 == set2 {
        0.0
    } else {
        1.0
    }
}

#[pyfunction(signature = (x, y))]
fn masi_distance(x: Vec<String>, y: Vec<String>) -> f64 {
    let set1: HashSet<&str> = x.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = y.iter().map(|s| s.as_str()).collect();
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    if union == 0 {
        return 0.0;
    }
    let jaccard = intersection as f64 / union as f64;
    let set1_only = set1.difference(&set2).count();
    let set2_only = set2.difference(&set1).count();
    let diff_ratio = if set1_only > set2_only {
        set2_only as f64 / set1_only as f64
    } else if set2_only > 0 {
        set1_only as f64 / set2_only as f64
    } else {
        1.0
    };
    1.0 - jaccard * diff_ratio
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jaccard_distance, m)?)?;
    m.add_function(wrap_pyfunction!(binary_distance, m)?)?;
    m.add_function(wrap_pyfunction!(masi_distance, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_jaccard_identical() {
        let d = jaccard_distance(vec!["a".into(), "b".into()], vec!["a".into(), "b".into()]);
        assert!((d - 0.0).abs() < 0.001);
    }
    #[test]
    fn test_jaccard_disjoint() {
        let d = jaccard_distance(vec!["a".into()], vec!["b".into()]);
        assert!((d - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_binary_identical() {
        assert!((binary_distance(vec!["a".into()], vec!["a".into()]) - 0.0).abs() < 0.001);
    }
    #[test]
    fn test_binary_different() {
        assert!((binary_distance(vec!["a".into()], vec!["b".into()]) - 1.0).abs() < 0.001);
    }
}
