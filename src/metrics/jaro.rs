//! Jaro / Jaro-Winkler similarity — ported from vtext (Apache-2.0).

use pyo3::prelude::*;
use std::cmp::{max, min};
use std::collections::HashSet;

/// Jaro similarity. `O(max_len²)` but with O(1) matched-set checks via bool vec.
fn jaro_sim(x: &str, y: &str) -> f64 {
    let xc: Vec<char> = x.chars().collect();
    let yc: Vec<char> = y.chars().collect();
    let xl = xc.len();
    let yl = yc.len();
    if xl == 0 && yl == 0 {
        return 1.0;
    }
    if xl == 0 || yl == 0 {
        return 0.0;
    }
    let bound = max(xl, yl);
    let mut matched_y = vec![false; yl]; // O(1) lookup instead of O(n) Vec::contains
    let mut f1: Vec<usize> = Vec::new();
    let mut f2: Vec<usize> = Vec::new();
    for (i, &c1) in xc.iter().enumerate() {
        let lo = max(0, i as i32 - bound as i32) as usize;
        let hi = min(i + bound, yl - 1);
        for (j, &c2) in yc.iter().enumerate().take(hi + 1).skip(lo) {
            if c1 == c2 && !matched_y[j] {
                matched_y[j] = true;
                f1.push(i);
                f2.push(j);
                break;
            }
        }
    }
    f2.sort_unstable();
    let m = f1.len();
    if m == 0 {
        return 0.0;
    }
    // Transpositions: zip x-indices (in match order) with y-indices (sorted),
    // count character mismatches. Each transposition creates 2 mismatches.
    let t = f1.iter().zip(&f2).filter(|(&i, &j)| xc[i] != yc[j]).count();
    (m as f64 / xl as f64 + m as f64 / yl as f64 + (m as f64 - t as f64 / 2.0) / m as f64) / 3.0
}

fn jaro_winkler_sim(x: &str, y: &str, p: f64, max_l: usize) -> f64 {
    let js = jaro_sim(x, y);
    let mut l = 0usize;
    for (a, b) in x.chars().zip(y.chars()) {
        if a == b && l < max_l {
            l += 1;
        } else {
            break;
        }
    }
    (l as f64 * p).mul_add(1.0 - js, js)
}

#[pyfunction(signature = (x, y))]
fn jaro_similarity(x: &str, y: &str) -> f64 {
    jaro_sim(x, y)
}

#[pyfunction(signature = (x, y, p=0.1, max_l=4))]
fn jaro_winkler_similarity(x: &str, y: &str, p: f64, max_l: usize) -> f64 {
    jaro_winkler_sim(x, y, p, max_l)
}

#[pyfunction(signature = (x, y))]
fn dice_similarity(x: &str, y: &str) -> f64 {
    if x.len() < 2 || y.len() < 2 {
        return 0.0;
    }
    if x == y {
        return 1.0;
    }
    let mut xs: HashSet<(char, char)> = HashSet::new();
    let mut ys: HashSet<(char, char)> = HashSet::new();
    // Avoid Vec<char> allocation: iterate char pairs directly
    {
        let mut prev: Option<char> = None;
        for c in x.chars() {
            if let Some(p) = prev {
                xs.insert((p, c));
            }
            prev = Some(c);
        }
    }
    {
        let mut prev: Option<char> = None;
        for c in y.chars() {
            if let Some(p) = prev {
                ys.insert((p, c));
            }
            prev = Some(c);
        }
    }
    let inter = xs.intersection(&ys).count();
    2.0 * inter as f64 / (xs.len() + ys.len()) as f64
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dice_similarity, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_jaro() {
        let s = jaro_sim("SHACKLEFORD", "SHACKELFORD");
        // With correct transposition counting via zip of f1/f2: L/E swap is a transposition
        assert!((s - 0.970).abs() < 0.01);
    }
    #[test]
    fn test_jaro_winkler() {
        let s = jaro_winkler_sim("SHACKLEFORD", "SHACKELFORD", 0.1, 4);
        // Winkler boost from "SHAC" prefix -> ~0.982
        assert!((s - 0.982).abs() < 0.01);
    }
    #[test]
    fn test_dice() {
        let s = dice_similarity("healed", "sealed");
        assert!((s - 0.8).abs() < 0.02);
    }
}
