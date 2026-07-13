//! String metrics — Rust implementations of distance functions.
//!
//! Ported from vtext (Apache-2.0) and NLTK algorithms.

use pyo3::prelude::*;

// ═══════════════════════════════════════════════════════════
// edit_distance
// ═══════════════════════════════════════════════════════════

/// Compute the Levenshtein edit distance between two strings.
///
/// Matches NLTK's `nltk.metrics.distance.edit_distance`.
/// Based on the Wagner-Fischer algorithm with O(n*m) time and O(min(n,m)) space.
#[pyfunction(signature = (s1, s2, substitution_cost=1, transpositions=false))]
fn edit_distance(s1: &str, s2: &str, substitution_cost: u32, transpositions: bool) -> f64 {
    compute_edit_distance(s1, s2, substitution_cost as usize, transpositions) as f64
}

/// Core edit distance algorithm.
fn compute_edit_distance(s1: &str, s2: &str, substitution_cost: usize, transpositions: bool) -> usize {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len == 0 {
        return s2_len * substitution_cost.min(1);
    }
    if s2_len == 0 {
        return s1_len * substitution_cost.min(1);
    }

    // Use two-row optimization — only keep previous and current row
    let mut prev: Vec<usize> = (0..=s2_len).collect();
    let mut curr: Vec<usize> = vec![0; s2_len + 1];
    let mut prev_prev: Vec<usize> = (0..=s2_len).collect(); // for transpositions

    for (i, c1) in s1.chars().enumerate() {
        curr[0] = i + 1;

        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { substitution_cost };

            // Standard Levenshtein
            let del = prev[j + 1] + 1;
            let ins = curr[j] + 1;
            let sub = prev[j] + cost;

            let mut min_cost = del.min(ins).min(sub);

            // Damerau transpositions
            if transpositions && i > 0 && j > 0 {
                if s1.chars().nth(i - 1) == Some(c2) && s1.chars().nth(i) == Some(s2.chars().nth(j - 1).unwrap()) {
                    min_cost = min_cost.min(prev_prev[j - 1] + substitution_cost);
                }
            }

            curr[j + 1] = min_cost;
        }

        // Rotate rows
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[s2_len]
}

/// Register the module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edit_distance, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_strings() {
        assert!((compute_edit_distance("", "", 1, false) as f64 - 0.0).abs() < 0.001);
        assert!((compute_edit_distance("a", "", 1, false) as f64 - 1.0).abs() < 0.001);
        assert!((compute_edit_distance("", "a", 1, false) as f64 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_identical() {
        assert!((compute_edit_distance("hello", "hello", 1, false) as f64 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_single_substitution() {
        assert!((compute_edit_distance("cat", "car", 1, false) as f64 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_single_insertion() {
        assert!((compute_edit_distance("cat", "cats", 1, false) as f64 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_single_deletion() {
        assert!((compute_edit_distance("cats", "cat", 1, false) as f64 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_multiple_edits() {
        assert!((compute_edit_distance("kitten", "sitting", 1, false) as f64 - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_yesterday_today() {
        let d = edit_distance("yesterday", "today", 1, false);
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_transpositions() {
        let d = edit_distance("ab", "ba", 1, true);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_substitution_cost_2() {
        let d = edit_distance("cat", "car", 2, false);
        assert!((d - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_unicode() {
        let d = edit_distance("café", "cafe", 1, false);
        assert!((d - 1.0).abs() < 0.001);
    }
}
