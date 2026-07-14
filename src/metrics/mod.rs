//! String & scoring metrics — Rust implementations.

mod agreement;
mod association;
mod jaccard;
mod jaro;
mod scores;
mod segmentation;
mod spearman;

use pyo3::prelude::*;

#[pyfunction(signature = (s1, s2, substitution_cost=1, transpositions=false))]
fn edit_distance(s1: &str, s2: &str, substitution_cost: u32, transpositions: bool) -> f64 {
    compute_edit_distance(s1, s2, substitution_cost as usize, transpositions) as f64
}

fn compute_edit_distance(s1: &str, s2: &str, sub_cost: usize, trans: bool) -> usize {
    let (xl, yl) = (s1.chars().count(), s2.chars().count());
    if xl == 0 {
        return yl * sub_cost.min(1);
    }
    if yl == 0 {
        return xl * sub_cost.min(1);
    }
    let mut prev: Vec<usize> = (0..=yl).collect();
    let mut curr = vec![0; yl + 1];
    let mut prev_prev = prev.clone();
    for (i, c1) in s1.chars().enumerate() {
        curr[0] = i + 1;
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { sub_cost };
            let mut best = prev[j + 1] + 1;
            best = best.min(curr[j] + 1);
            best = best.min(prev[j] + cost);
            if trans
                && i > 0
                && j > 0
                && s1.chars().nth(i - 1) == Some(c2)
                && s1.chars().nth(i) == Some(s2.chars().nth(j - 1).unwrap())
            {
                best = best.min(prev_prev[j - 1] + sub_cost);
            }
            curr[j + 1] = best;
        }
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[yl]
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edit_distance, m)?)?;
    jaro::register_module(m)?;
    scores::register_module(m)?;
    jaccard::register_module(m)?;
    segmentation::register_module(m)?;
    association::register_module(m)?;
    agreement::register_module(m)?;
    spearman::register_module(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ed() {
        assert!((compute_edit_distance("cat", "car", 1, false) as f64 - 1.0).abs() < 0.001);
        assert!((compute_edit_distance("kitten", "sitting", 1, false) as f64 - 3.0).abs() < 0.001);
        assert!((compute_edit_distance("", "a", 1, false) as f64 - 1.0).abs() < 0.001);
        assert!((compute_edit_distance("ab", "ba", 1, true) as f64 - 1.0).abs() < 0.001);
    }
}
