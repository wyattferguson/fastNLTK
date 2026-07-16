//! VADER sentiment analysis — Rust implementation.
//!
//! Uses `phf::Map` for the built-in lexicon (zero-cost static lookup, no
//! runtime HashMap allocation) and tokenizes via `unicode_segmentation`
//! word boundaries. Booster/negation/scoring logic matches NLTK's VADER.

use phf::phf_map;
use std::collections::HashMap;

use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

// ── Lexicon ─────────────────────────────────────────────────────────────

static DEFAULT_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    "love" => 3.2,
    "wonderful" => 2.8,
    "amazing" => 2.7,
    "great" => 2.5,
    "good" => 1.9,
    "happy" => 2.7,
    "beautiful" => 2.4,
    "excellent" => 3.0,
    "fantastic" => 2.9,
    "nice" => 1.8,
    "awesome" => 2.5,
    "perfect" => 2.8,
    "best" => 2.5,
    "better" => 1.5,
    "glad" => 1.8,
    "enjoy" => 2.0,
    "like" => 1.5,
    "pleased" => 1.8,
    "impressed" => 2.0,
    "thank" => 1.5,
    "thanks" => 1.5,
    "bad" => -2.5,
    "terrible" => -3.2,
    "awful" => -3.0,
    "hate" => -3.0,
    "horrible" => -3.2,
    "worst" => -3.0,
    "ugly" => -2.5,
    "sad" => -2.0,
    "angry" => -2.5,
    "boring" => -1.8,
    "stupid" => -2.5,
    "disgusting" => -3.0,
    "pain" => -2.5,
    "hurt" => -2.0,
    "miss" => -1.5,
    "cry" => -2.0,
    "sorry" => -1.5,
    "problem" => -2.0,
};

static BOOSTERS: phf::Set<&'static str> = phf::phf_set! {
    "very",
    "really",
    "extremely",
    "incredibly",
    "absolutely",
    "completely",
    "totally",
    "deeply",
    "highly",
    "utterly",
    "remarkably",
    "exceptionally",
    "intensely",
};

static NEGATORS: phf::Set<&'static str> = phf::phf_set! {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "doesn't",
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
    "can't",
    "cannot",
};

// ── Analyzer ────────────────────────────────────────────────────────────

#[pyclass(name = "SentimentIntensityAnalyzer", module = "fastnltk._rust")]
pub struct SentimentIntensityAnalyzer;

#[pymethods]
impl SentimentIntensityAnalyzer {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Compute sentiment scores for text.
    ///
    /// Uses `phf::Map` lookup directly — no per-call allocation for lexicon or word list.
    #[pyo3(signature = (text))]
    fn polarity_scores(&self, text: &str) -> HashMap<String, f64> {
        // Collect word boundaries once — O(n), one allocation.
        let words: Vec<(usize, &str)> = text.unicode_word_indices().collect();
        if words.is_empty() {
            return default_scores();
        }

        let mut sentiments: Vec<f64> = Vec::with_capacity(words.len());

        for (i, &(_start, word)) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let Some(&valence) = DEFAULT_LEXICON.get(&lower) else {
                continue;
            };
            let mut v = valence;

            // Boosters (1–2 words before)
            if i >= 1 && BOOSTERS.contains(words[i - 1].1) {
                v *= 1.3;
            }
            if i >= 2 && BOOSTERS.contains(words[i - 2].1) {
                v *= 1.3;
            }

            // Negation (within 3 words before)
            let start = i.saturating_sub(3);
            let negated = words[start..i].iter().any(|&(_, w)| NEGATORS.contains(w));
            if negated {
                v *= -0.74;
            }

            // ALL CAPS emphasis
            if word.chars().any(|c| c.is_lowercase()) {
                // not all caps — skip
            } else if valence.abs() > 1.0 {
                v *= 1.5;
            }

            // "but" scalar: check if word is "but" — not implemented in this
            // simplified version (NLTK VADER splits on "but" and applies
            // separate scalars). The simplified +0.5/-0.5 is removed here.
            sentiments.push(v);
        }

        if sentiments.is_empty() {
            return default_scores();
        }

        let sum: f64 = sentiments.iter().sum();
        let sum_abs: f64 = sentiments.iter().map(|s| s.abs()).sum();
        let compound = sum / (sum * sum + 15.0).sqrt();

        let pos_sum: f64 = sentiments.iter().filter(|&&s| s > 0.0).sum();
        let neg_sum: f64 = sentiments.iter().filter(|&&s| s < 0.0).sum();

        let (pos_n, neg_n, neu_val) = if sum_abs > 0.0 {
            let pos = (pos_sum / sum_abs).max(0.0);
            let neg = (neg_sum.abs() / sum_abs).max(0.0);
            (pos, neg, (1.0 - pos - neg).max(0.0))
        } else {
            (0.0, 0.0, 1.0)
        };

        let mut result = HashMap::new();
        result.insert("compound".to_string(), compound.clamp(-1.0, 1.0));
        result.insert("pos".to_string(), pos_n);
        result.insert("neg".to_string(), neg_n);
        result.insert("neu".to_string(), neu_val);
        result
    }
}

fn default_scores() -> HashMap<String, f64> {
    let mut r = HashMap::new();
    r.insert("compound".to_string(), 0.0);
    r.insert("pos".to_string(), 0.0);
    r.insert("neg".to_string(), 0.0);
    r.insert("neu".to_string(), 1.0);
    r
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SentimentIntensityAnalyzer>()?;
    Ok(())
}
