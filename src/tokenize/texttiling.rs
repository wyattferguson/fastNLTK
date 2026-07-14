//! TextTiling — topical segmentation of documents.
//!
//! Port of NLTK's nltk.tokenize.texttiling.TextTilingTokenizer.
//! Detects subtopic shifts based on lexical co-occurrence patterns.
//!
//! Algorithm steps:
//! 1. Tokenize into pseudo-sentences (fixed-size blocks)
//! 2. Compute similarity between adjacent blocks
//! 3. Smooth the similarity scores
//! 4. Find valleys (boundaries) in the score curve
//!
//! NLTK equivalent: nltk.tokenize.texttiling.TextTilingTokenizer

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use pyo3::prelude::*;
use regex::Regex;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z]+").unwrap());

#[pyclass(name = "TextTilingTokenizer", module = "fastnltk._rust")]
pub struct TextTilingTokenizer {
    w: usize, // pseudo-sentence size (words)
    k: usize, // block size (pseudo-sentences)
    demo_mode: bool,
}

#[pymethods]
impl TextTilingTokenizer {
    #[new]
    #[pyo3(signature = (w=20, k=10, demo_mode=false))]
    fn new(w: usize, k: usize, demo_mode: bool) -> Self {
        TextTilingTokenizer { w, k, demo_mode }
    }

    /// Tokenize text into topical segments.
    /// Returns (segments, scores, depth_scores, boundary_mask).
    fn tokenize(&self, text: &str) -> PyResult<(Vec<String>, Vec<f64>, Vec<f64>, Vec<u8>)> {
        let words = tokenize_words(text);
        if words.len() < self.w + self.k {
            return Ok((vec![text.to_string()], vec![], vec![], vec![]));
        }

        // Step 1: Build pseudo-sentences
        let pseudo_sents = build_pseudo_sentences(&words, self.w);
        if pseudo_sents.len() < 3 {
            return Ok((vec![text.to_string()], vec![], vec![], vec![]));
        }

        // Step 2: Compute similarity scores between adjacent blocks
        let scores = compute_block_similarity(&pseudo_sents, self.k);

        // Step 3: Smooth scores
        let smoothed = smooth_scores(&scores, 2);
        let smoothed = smooth_scores(&smoothed, 2);

        // Step 4: Find depth scores (valleys)
        let depths = compute_depth_scores(&smoothed);

        // Step 5: Find boundaries
        let boundaries = find_boundaries(&depths, &scores, self.demo_mode);

        // Step 6: Build segments from boundaries
        let segments = build_segments_from_bounds(text, &pseudo_sents, &boundaries);

        Ok((segments, scores, depths, boundaries))
    }
}

fn tokenize_words(text: &str) -> Vec<String> {
    WORD_RE
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn build_pseudo_sentences(words: &[String], w: usize) -> Vec<Vec<String>> {
    if w == 0 {
        return vec![words.to_vec()];
    }
    words.chunks(w).map(|chunk| chunk.to_vec()).collect()
}

fn compute_block_similarity(pseudo_sents: &[Vec<String>], k: usize) -> Vec<f64> {
    let n = pseudo_sents.len();
    if n < 2 * k + 1 {
        return vec![0.0; n];
    }

    let mut scores = Vec::with_capacity(n);
    scores.push(0.0); // first gap has no left block

    for gap in 1..n {
        if gap < k || gap + k > n {
            scores.push(0.0);
            continue;
        }

        // Build left block vocabulary (gap-k..gap)
        let mut left_freq: HashMap<&str, f64> = HashMap::new();
        for ps in &pseudo_sents[gap - k..gap] {
            for w in ps {
                *left_freq.entry(w).or_insert(0.0) += 1.0;
            }
        }

        // Build right block vocabulary (gap..gap+k)
        let mut right_freq: HashMap<&str, f64> = HashMap::new();
        let right_start = gap.min(n - 1);
        let right_end = (gap + k).min(n);
        for ps in &pseudo_sents[right_start..right_end] {
            for w in ps {
                *right_freq.entry(w).or_insert(0.0) += 1.0;
            }
        }

        // Cosine similarity
        let mut dot = 0.0;
        let mut left_mag = 0.0;
        let mut right_mag = 0.0;

        // All unique words from both blocks
        let mut all_words: HashSet<&str> = left_freq.keys().copied().collect();
        all_words.extend(right_freq.keys().copied());

        for w in &all_words {
            let lf = left_freq.get(w).copied().unwrap_or(0.0);
            let rf = right_freq.get(w).copied().unwrap_or(0.0);
            // Use log-frequency weighting
            let lw = if lf > 0.0 { 1.0 + lf.log(2.0) } else { 0.0 };
            let rw = if rf > 0.0 { 1.0 + rf.log(2.0) } else { 0.0 };
            dot += lw * rw;
            left_mag += lw * lw;
            right_mag += rw * rw;
        }

        let sim = if left_mag > 0.0 && right_mag > 0.0 {
            dot / (left_mag.sqrt() * right_mag.sqrt())
        } else {
            0.0
        };

        scores.push(sim);
    }

    scores
}

fn smooth_scores(scores: &[f64], window: usize) -> Vec<f64> {
    let n = scores.len();
    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(n);
        let avg: f64 = scores[start..end].iter().sum::<f64>() / (end - start) as f64;
        smoothed.push(avg);
    }
    smoothed
}

fn compute_depth_scores(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut depths = Vec::with_capacity(n);
    for i in 0..n {
        if i == 0 || i == n - 1 {
            depths.push(0.0);
        } else {
            // Valleys: points where both neighbors are higher
            let left = scores[i - 1];
            let right = scores[i + 1];
            if scores[i] < left && scores[i] < right {
                depths.push((left - scores[i]).min(right - scores[i]));
            } else {
                depths.push(0.0);
            }
        }
    }
    depths
}

fn find_boundaries(depths: &[f64], _scores: &[f64], demo_mode: bool) -> Vec<u8> {
    let n = depths.len();
    if n < 2 {
        return vec![0; n.max(1)];
    }

    let max_depth = depths.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = if demo_mode {
        max_depth * 0.3
    } else {
        max_depth * 0.5
    };

    let mut boundaries = vec![0u8; n];
    for i in 1..n - 1 {
        if depths[i] >= threshold && depths[i] > depths[i - 1] && depths[i] >= depths[i + 1] {
            boundaries[i] = 1;
        }
    }

    boundaries
}

fn build_segments_from_bounds(
    text: &str,
    pseudo_sents: &[Vec<String>],
    boundaries: &[u8],
) -> Vec<String> {
    if boundaries.is_empty() || pseudo_sents.is_empty() {
        return vec![text.to_string()];
    }

    // Approximate character boundaries from pseudo-sentence indices
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![text.to_string()];
    }

    let word_count = words.len();
    let ps_count = pseudo_sents.len();
    let mut segments = Vec::new();
    let mut start = 0;

    for (i, &boundary) in boundaries.iter().enumerate().take(ps_count) {
        if boundary == 1 {
            let word_pos = (i * word_count / ps_count.max(1)).min(word_count);
            if word_pos > start {
                segments.push(words[start..word_pos].join(" "));
                start = word_pos;
            }
        }
    }

    if start < word_count {
        segments.push(words[start..].join(" "));
    }

    if segments.is_empty() {
        segments.push(text.to_string());
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texttiling_short_text() {
        let tt = TextTilingTokenizer::new(20, 10, true);
        let (segments, _, _, _) = tt.tokenize("Hello world.").unwrap();
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_tokenize_words() {
        let words = tokenize_words("Hello, world! This is a test.");
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_pseudo_sentences() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ps = build_pseudo_sentences(&words, 2);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0], vec!["a", "b"]);
    }

    #[test]
    fn test_smooth_scores() {
        let scores = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let smoothed = smooth_scores(&scores, 1);
        assert_eq!(smoothed.len(), 5);
    }
}
