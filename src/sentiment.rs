//! VADER sentiment analysis — Rust implementation.
//!
//! Port of NLTK's nltk.sentiment.vader module.
//! Rule-based sentiment intensity analyzer using a lexicon
//! of valence scores and heuristics (boosters, negators, etc.).

use phf::phf_map;
use std::collections::HashMap;

use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

// ═══════════════════════════════════════════════════════════
// VADER Lexicon (built-in subset of common English words)
// ═══════════════════════════════════════════════════════════
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

fn default_lexicon() -> HashMap<String, f64> {
    DEFAULT_LEXICON.entries().map(|(k, v)| ((*k).to_string(), *v)).collect()
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
        let mut lex: HashMap<String, f64> = HashMap::new();
        for (k, v) in default_lexicon() {
            lex.insert(k.to_string(), v);
        }
        SentimentIntensityAnalyzer { lexicon: lex }
    }

    /// Compute sentiment scores for text.
    #[pyo3(signature = (text))]
    fn polarity_scores(&self, text: &str) -> std::collections::HashMap<String, f64> {
        let words: Vec<String> = text.unicode_words().map(|w| w.to_lowercase()).collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        let mut sentiments: Vec<f64> = Vec::with_capacity(word_refs.len());
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
