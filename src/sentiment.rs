//! VADER sentiment analysis — Rust implementation.
//!
//! Port of NLTK's nltk.sentiment.vader module.
//! Rule-based sentiment intensity analyzer using a lexicon
//! of valence scores and heuristics (boosters, negators, etc.).

use std::collections::HashMap;

use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

// ═══════════════════════════════════════════════════════════
// VADER Lexicon (built-in subset of common English words)
// ═══════════════════════════════════════════════════════════

fn default_lexicon() -> HashMap<&'static str, f64> {
    let mut lex = HashMap::new();
    // Positive
    lex.insert("love", 3.2);
    lex.insert("wonderful", 2.8);
    lex.insert("amazing", 2.7);
    lex.insert("great", 2.5);
    lex.insert("good", 1.9);
    lex.insert("happy", 2.7);
    lex.insert("beautiful", 2.4);
    lex.insert("excellent", 3.0);
    lex.insert("fantastic", 2.9);
    lex.insert("nice", 1.8);
    lex.insert("awesome", 2.5);
    lex.insert("perfect", 2.8);
    lex.insert("best", 2.5);
    lex.insert("better", 1.5);
    lex.insert("glad", 1.8);
    lex.insert("enjoy", 2.0);
    lex.insert("like", 1.5);
    lex.insert("pleased", 1.8);
    lex.insert("impressed", 2.0);
    lex.insert("thank", 1.5);
    lex.insert("thanks", 1.5);
    // Negative
    lex.insert("bad", -2.5);
    lex.insert("terrible", -3.2);
    lex.insert("awful", -3.0);
    lex.insert("hate", -3.0);
    lex.insert("horrible", -3.2);
    lex.insert("worst", -3.0);
    lex.insert("ugly", -2.5);
    lex.insert("sad", -2.0);
    lex.insert("angry", -2.5);
    lex.insert("boring", -1.8);
    lex.insert("stupid", -2.5);
    lex.insert("ugly", -2.5);
    lex.insert("disgusting", -3.0);
    lex.insert("terrible", -3.2);
    lex.insert("pain", -2.5);
    lex.insert("hurt", -2.0);
    lex.insert("miss", -1.5);
    lex.insert("cry", -2.0);
    lex.insert("sorry", -1.5);
    lex.insert("problem", -2.0);
    lex
}

static BOOSTERS: &[&str] = &[
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
];

static NEGATORS: &[&str] = &[
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
];

// ═══════════════════════════════════════════════════════════
// SentimentIntensityAnalyzer
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "SentimentIntensityAnalyzer", module = "fastnltk._rust")]
pub struct SentimentIntensityAnalyzer {
    lexicon: HashMap<String, f64>,
}

#[pymethods]
impl SentimentIntensityAnalyzer {
    #[new]
    fn new() -> Self {
        let mut lex = HashMap::new();
        for (k, v) in default_lexicon() {
            lex.insert(k.to_string(), v);
        }
        SentimentIntensityAnalyzer { lexicon: lex }
    }

    /// Compute sentiment scores for text.
    #[pyo3(signature = (text))]
    fn polarity_scores(&self, text: &str) -> HashMap<String, f64> {
        let words: Vec<String> = text.unicode_words().map(|w| w.to_lowercase()).collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        let mut sentiments: Vec<f64> = Vec::new();
        let mut i = 0;

        while i < word_refs.len() {
            let word = word_refs[i];
            let mut valence = match self.lexicon.get(word) {
                Some(&v) => v,
                None => {
                    i += 1;
                    continue;
                }
            };

            // Check for booster words before
            if i > 0 {
                if BOOSTERS.contains(&word_refs[i - 1]) {
                    valence *= 1.3;
                }
                if i > 1 && BOOSTERS.contains(&word_refs[i - 2]) {
                    valence *= 1.3;
                }
            }

            // Check for negation before (within 3 words)
            let mut negated = false;
            for j in (0.max(i as i32 - 3) as usize)..i {
                if NEGATORS.contains(&word_refs[j]) {
                    negated = true;
                    break;
                }
            }
            if negated {
                valence *= -0.74;
            }

            // Check capitalization emphasis (ALL CAPS)
            let original_word = text.unicode_words().nth(i).unwrap_or("");
            #[allow(unused_mut)]
            if original_word.chars().all(|c| c.is_uppercase()) && valence.abs() > 1.0 {
                valence *= 1.5;
            }

            // Check for "but" — de-emphasize before, emphasize after
            // (simplified: just apply the valence)
            if valence > 0.0 {
                valence += 0.5;
            } else if valence < 0.0 {
                valence -= 0.5;
            }

            sentiments.push(valence);
            i += 1;
        }

        // Compute compound score
        let compound = if sentiments.is_empty() {
            0.0
        } else {
            let sum: f64 = sentiments.iter().sum();
            sum / (sum.abs() + 15.0).sqrt()
        };

        let sum_abs: f64 = sentiments.iter().map(|s| s.abs()).sum();
        let pos_sum: f64 = sentiments.iter().filter(|&&s| s > 0.0).sum();
        let neg_sum: f64 = sentiments.iter().filter(|&&s| s < 0.0).sum();

        let (pos_n, neg_n, neu_n) = if sum_abs > 0.0 {
            let pos = (pos_sum / sum_abs).max(0.0);
            let neg = (neg_sum.abs() / sum_abs).max(0.0);
            let neu = 1.0 - pos - neg;
            (pos, neg, neu.max(0.0))
        } else {
            (0.0, 0.0, 1.0)
        };

        let compound = compound.max(-1.0).min(1.0);

        let mut result = HashMap::new();
        result.insert("compound".to_string(), compound);
        result.insert("pos".to_string(), pos_n);
        result.insert("neg".to_string(), neg_n);
        result.insert("neu".to_string(), neu_n);
        result
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SentimentIntensityAnalyzer>()?;
    Ok(())
}
