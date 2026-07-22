//! VADER sentiment analysis — Rust implementation matching NLTK's
//! scoring algorithm exactly. Uses `phf` maps for zero-cost static lookup.

use phf::phf_map;
use std::collections::HashMap;

use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

// Constants matching NLTK

const C_INCR: f64 = 0.733;
const N_SCALAR: f64 = -0.74;
const ALPHA: f64 = 15.0;

// Lexicon

static DEFAULT_LEXICON: phf::Map<&'static str, f64> = phf_map! {
    // Values from NLTK's vader_lexicon.txt (v3.10)
    "love" => 3.2,
    "wonderful" => 2.7,
    "amazing" => 2.8,
    "great" => 3.1,
    "good" => 1.9,
    "happy" => 2.7,
    "beautiful" => 2.9,
    "excellent" => 2.7,
    "fantastic" => 2.6,
    "okay" => 0.9,
    "nice" => 1.8,
    "awesome" => 3.1,
    "perfect" => 2.7,
    "best" => 3.2,
    "better" => 1.9,
    "glad" => 2.0,
    "enjoy" => 2.2,
    "like" => 1.5,
    "pleased" => 1.9,
    "impressed" => 2.1,
    "thank" => 1.5,
    "thanks" => 1.9,
    "bad" => -2.5,
    "terrible" => -2.1,
    "awful" => -2.0,
    "hate" => -2.7,
    "horrible" => -2.5,
    "worst" => -3.1,
    "ugly" => -2.3,
    "sad" => -2.1,
    "angry" => -2.3,
    "boring" => -1.3,
    "stupid" => -2.4,
    "disgusting" => -2.4,
    "pain" => -2.3,
    "hurt" => -2.4,
    "miss" => -0.6,
    "cry" => -2.1,
    "sorry" => -0.3,
    "problem" => -1.7,
};

/// NLTK `BOOSTER_DICT` entries: word → scalar modifier.
static BOOSTER_DICT: phf::Map<&'static str, f64> = phf::phf_map! {
    "absolutely" => 0.293, "amazingly" => 0.293, "awfully" => 0.293,
    "completely" => 0.293, "considerably" => 0.293, "decidedly" => 0.293,
    "deeply" => 0.293, "effing" => 0.293, "enormously" => 0.293,
    "entirely" => 0.293, "especially" => 0.293, "exceptionally" => 0.293,
    "extremely" => 0.293, "fabulously" => 0.293, "flipping" => 0.293,
    "flippin" => 0.293, "fricking" => 0.293, "frickin" => 0.293,
    "frigging" => 0.293, "friggin" => 0.293, "fully" => 0.293,
    "fucking" => 0.293, "greatly" => 0.293, "hella" => 0.293,
    "highly" => 0.293, "hugely" => 0.293, "incredibly" => 0.293,
    "intensely" => 0.293, "majorly" => 0.293, "more" => 0.293,
    "most" => 0.293, "particularly" => 0.293, "purely" => 0.293,
    "quite" => 0.293, "really" => 0.293, "remarkably" => 0.293,
    "so" => 0.293, "substantially" => 0.293, "thoroughly" => 0.293,
    "totally" => 0.293, "tremendously" => 0.293, "uber" => 0.293,
    "unbelievably" => 0.293, "unusually" => 0.293, "utterly" => 0.293,
    "very" => 0.293,
    "almost" => -0.293, "barely" => -0.293, "hardly" => -0.293,
    "just enough" => -0.293, "kind of" => -0.293, "kinda" => -0.293,
    "kindof" => -0.293, "kind-of" => -0.293, "less" => -0.293,
    "little" => -0.293, "marginally" => -0.293, "occasionally" => -0.293,
    "partly" => -0.293, "scarcely" => -0.293, "slightly" => -0.293,
    "somewhat" => -0.293, "sort of" => -0.293, "sorta" => -0.293,
    "sortof" => -0.293, "sort-of" => -0.293,
};

static NEGATE: phf::Set<&'static str> = phf::phf_set! {
    "ain't", "aint", "aren't", "arent", "can't", "cannot", "cant",
    "couldn't", "couldnt", "daren't", "darent", "despite",
    "didn't", "didnt", "doesn't", "doesnt", "don't", "dont",
    "hadn't", "hadnt", "hasn't", "hasnt", "haven't", "havent",
    "isn't", "isnt", "mightn't", "mightnt", "mustn't", "mustnt",
    "needn't", "neednt", "neither", "never", "none", "nope",
    "nor", "not", "nothing", "nowhere", "oughtn't", "oughtnt",
    "rarely", "seldom", "shan't", "shant", "shouldn't", "shouldnt",
    "uh-uh", "uhuh", "wasn't", "wasnt", "weren't", "werent",
    "without", "won't", "wont", "wouldn't", "wouldnt",
};

/// Check if a word is negated (NLTK's `negated`).
fn is_negated(word: &str) -> bool {
    if NEGATE.contains(word) {
        return true;
    }
    // Check for "n't" suffix (matches "doesn't", "can't", etc. variants)
    if word.len() >= 3 && word.ends_with("n't") {
        return true;
    }
    false
}

// Analyzer

#[pyclass(name = "SentimentIntensityAnalyzer", module = "fastnltk._rust")]
pub struct SentimentIntensityAnalyzer;

#[pymethods]
impl SentimentIntensityAnalyzer {
    #[new]
    const fn new() -> Self {
        Self
    }

    #[pyo3(signature = (text))]
    fn polarity_scores(&self, text: &str) -> HashMap<String, f64> {
        // Tokenize: split on whitespace, filter len > 1 (matches NLTK SentiText)
        let words: Vec<(usize, &str)> =
            text.unicode_word_indices().filter(|(_, w)| w.len() > 1).collect();
        // NLTK SentiText skips standalone single chars like "I" and "a".
        if words.is_empty() {
            return default_scores();
        }

        // Check if text has mixed casing (for ALL CAPS detection)
        let is_cap_diff = words.iter().any(|(_, w)| w.chars().any(char::is_uppercase))
            && words.iter().any(|(_, w)| w.chars().any(char::is_lowercase));

        // Per-word sentiment
        let mut sentiments: Vec<f64> = Vec::with_capacity(words.len());
        for (i, &(_start, word)) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let Some(mut valence) = DEFAULT_LEXICON.get(&lower).copied() else {
                sentiments.push(0.0);
                continue;
            };

            // ALL CAPS boost (C_INCR added to |valence|)
            if word.chars().any(char::is_uppercase)
                && !word.chars().any(char::is_lowercase)
                && is_cap_diff
                && valence.abs() >= 1.0
            {
                valence += if valence > 0.0 { C_INCR } else { -C_INCR };
            }

            // Check preceding words for boosters/negators (NLTK sentiment_valence)
            // NLTK checks up to 3 words back, skipping sentiment-laden words themselves.
            for start_i in 0..3.min(i) {
                let prev = words[i - start_i - 1].1.to_lowercase();
                if DEFAULT_LEXICON.contains_key(&prev) {
                    continue;
                }
                let s = scalar_inc_dec(&prev, valence, is_cap_diff);
                if s != 0.0 {
                    match start_i {
                        0 => valence += s,
                        1 => valence = s.mul_add(0.95, valence),
                        _ => valence = s.mul_add(0.9, valence),
                    }
                }
                // NLTK's _never_check: apply N_SCALAR if preceding word is a negator
                if is_negated(&prev) {
                    valence *= N_SCALAR;
                }
            }

            sentiments.push(valence);
        }

        // "but" check: words before "but" get *0.5, after get *1.5
        if let Some(bi) = words.iter().position(|(_, w)| w.eq_ignore_ascii_case("but")) {
            for (sidx, s) in sentiments.iter_mut().enumerate() {
                if sidx < bi {
                    *s *= 0.5;
                } else if sidx > bi {
                    *s *= 1.5;
                }
            }
        }

        // Punctuation emphasis: each "!" amplifies sum by 0.292
        let excl_count = text.bytes().filter(|&b| b == b'!').count() as f64;
        let punct_emph = excl_count * 0.292;

        let sum_s: f64 = sentiments.iter().sum();
        let sum_s = if sum_s > 0.0 {
            sum_s + punct_emph
        } else if sum_s < 0.0 {
            sum_s - punct_emph
        } else {
            sum_s
        };

        // Compound = sum / sqrt(sum² + ALPHA)  (NLTK normalize)
        let compound = sum_s / (sum_s * sum_s + ALPHA).sqrt();

        // Sift scores (NLTK _sift_sentiment_scores): each sentiment word
        // contributes its value + 1 (for the word itself) to pos_sum, or
        // its value - 1 to neg_sum. Zero-valued words → neu_count.
        let pos_sum: f64 = sentiments.iter().map(|&s| if s > 0.0 { s + 1.0 } else { 0.0 }).sum();
        let neg_sum: f64 = sentiments.iter().map(|&s| if s < 0.0 { s - 1.0 } else { 0.0 }).sum();
        let neu_count = sentiments.iter().filter(|&&s| s == 0.0).count() as f64;

        // Apply punctuation emphasis to the dominant sentiment sum
        let pos_sum = if pos_sum > neg_sum.abs() { pos_sum + punct_emph } else { pos_sum };
        let neg_sum = if neg_sum.abs() > pos_sum { neg_sum - punct_emph } else { neg_sum };

        let total = pos_sum + neg_sum.abs() + neu_count;
        let (pos_n, neg_n, neu_val) = if total > 0.0 {
            (
                (pos_sum / total).max(0.0),
                (neg_sum.abs() / total).max(0.0),
                (neu_count / total).max(0.0),
            )
        } else {
            (0.0, 0.0, 1.0)
        };

        // NLTK rounds to 3 decimal places (4 for compound)
        let scale3 = |v: f64| (v * 1000.0).round() / 1000.0;
        let scale4 = |v: f64| (v * 10000.0).round() / 10000.0;
        let mut result = HashMap::new();
        result.insert("compound".to_string(), scale4(compound.clamp(-1.0, 1.0)));
        result.insert("pos".to_string(), scale3(pos_n));
        result.insert("neg".to_string(), scale3(neg_n));
        result.insert("neu".to_string(), scale3(neu_val));
        result
    }
}

/// NLTK's `scalar_inc_dec`: checks if `word` is a booster/dampener.
/// Returns the scalar delta to apply to valence.
fn scalar_inc_dec(word: &str, valence: f64, is_cap_diff: bool) -> f64 {
    let Some(&scalar) = BOOSTER_DICT.get(word) else {
        return 0.0;
    };
    let s = if valence < 0.0 { -scalar } else { scalar };
    // ALL CAPS booster gets extra C_INCR
    if word.chars().any(char::is_uppercase) && !word.chars().any(char::is_lowercase) && is_cap_diff
    {
        if valence > 0.0 {
            s + C_INCR
        } else {
            s - C_INCR
        }
    } else {
        s
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
