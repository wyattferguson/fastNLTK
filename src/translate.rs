//! Translation metrics — BLEU score.
//!
//! Implements the BLEU (Bilingual Evaluation Understudy) score
//! matching NLTK's nltk.translate.bleu_score.

use pyo3::prelude::*;
use std::collections::HashMap;

fn ngram_counts(tokens: &[String], n: usize) -> HashMap<Vec<String>, u64> {
    let mut counts = HashMap::new();
    for ng in tokens.windows(n) {
        *counts.entry(ng.to_vec()).or_insert(0) += 1;
    }
    counts
}

fn clipped_count(candidate: &[String], reference: &[String], n: usize) -> (u64, u64) {
    let ref_counts = ngram_counts(reference, n);
    let cand_counts = ngram_counts(candidate, n);
    let mut clipped = 0u64;
    let mut total = 0u64;
    for (ngram, cnt) in &cand_counts {
        let max_ref = ref_counts.get(ngram).copied().unwrap_or(0);
        clipped += (*cnt).min(max_ref);
        total += cnt;
    }
    (clipped, total)
}

/// Compute BLEU score for one candidate against one reference.
#[pyfunction(signature = (candidate, reference, max_n=4))]
fn bleu_score(candidate: Vec<String>, reference: Vec<String>, max_n: usize) -> f64 {
    let cand_len = candidate.len();
    let ref_len = reference.len();

    if cand_len == 0 || ref_len == 0 {
        return 0.0;
    }

    // Brevity penalty
    let bp = if cand_len < ref_len {
        (1.0 - ref_len as f64 / cand_len as f64).exp()
    } else {
        1.0
    };

    // Geometric mean of ngram precisions
    let mut log_avg = 0.0;
    let mut valid_n = 0;

    for n in 1..=max_n.min(cand_len).min(ref_len) {
        let (clipped, total) = clipped_count(&candidate, &reference, n);
        if total > 0 && clipped > 0 {
            log_avg += (clipped as f64 / total as f64).ln();
            valid_n += 1;
        }
    }

    if valid_n == 0 {
        return 0.0;
    }

    let avg = (log_avg / valid_n as f64).exp();
    bp * avg
}

/// Compute corpus-level BLEU score.
#[pyfunction(signature = (candidates, references, max_n=4))]
fn corpus_bleu(candidates: Vec<Vec<String>>, references: Vec<Vec<String>>, max_n: usize) -> f64 {
    if candidates.is_empty() || references.is_empty() || candidates.len() != references.len() {
        return 0.0;
    }

    let mut total_clipped = vec![0u64; max_n];
    let mut total_counts = vec![0u64; max_n];
    let mut total_cand_len = 0usize;
    let mut total_ref_len = 0usize;

    for (cand, refn) in candidates.iter().zip(references.iter()) {
        total_cand_len += cand.len();
        total_ref_len += refn.len();

        for n in 1..=max_n.min(cand.len()).min(refn.len()) {
            let (clipped, total) = clipped_count(cand, refn, n);
            total_clipped[n - 1] += clipped;
            total_counts[n - 1] += total;
        }
    }

    if total_cand_len == 0 {
        return 0.0;
    }

    let bp = if total_cand_len < total_ref_len {
        (1.0 - total_ref_len as f64 / total_cand_len as f64).exp()
    } else {
        1.0
    };

    let mut log_avg = 0.0;
    let mut valid_n = 0;

    for n in 0..max_n {
        if total_counts[n] > 0 && total_clipped[n] > 0 {
            log_avg += (total_clipped[n] as f64 / total_counts[n] as f64).ln();
            valid_n += 1;
        }
    }

    if valid_n == 0 {
        return 0.0;
    }
    bp * (log_avg / valid_n as f64).exp()
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bleu_score, m)?)?;
    m.add_function(wrap_pyfunction!(corpus_bleu, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_identical() {
        let r = vec!["the".into(), "cat".into()];
        let s = bleu_score(r.clone(), r, 4);
        assert!((s - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bleu_empty() {
        let s = bleu_score(vec![], vec!["the".into()], 4);
        assert!((s - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_bleu_partial() {
        let c = vec!["the".into(), "cat".into()];
        let r = vec!["the".into(), "dog".into()];
        let s = bleu_score(c, r, 4);
        assert!(s > 0.0 && s < 1.0);
    }

    #[test]
    fn test_corpus_bleu() {
        let cs = vec![vec!["the".into(), "cat".into()]];
        let rs = vec![vec!["the".into(), "cat".into()]];
        let s = corpus_bleu(cs, rs, 4);
        assert!((s - 1.0).abs() < 0.01);
    }
}
