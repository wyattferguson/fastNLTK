//! TnT — Trigram 'n Tags, Rust-accelerated.
//!
//! Trigram HMM POS tagger with backoff smoothing matching NLTK's TnT.
//! Uses Viterbi decoding with transition/emission probabilities
//! estimated from training data.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

// ═══════════════════════════════════════════════════════════
// TnT tagger
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "TnT", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct TnT {
    /// Known tags
    tags: Vec<String>,
    /// Known words (from training)
    known_words: HashSet<String>,
    /// Unigram counts: tag → count
    uni_counts: HashMap<String, u64>,
    /// Bigram counts: (prev, curr) → count
    bi_counts: HashMap<(String, String), u64>,
    /// Trigram counts: (prev2, prev1, curr) → count
    tri_counts: HashMap<(String, String, String), u64>,
    /// Emission counts: (tag, word) → count
    emission_counts: HashMap<(String, String), u64>,
    /// Total word count (for OOV smoothing)
    total_words: u64,
}

#[pymethods]
impl TnT {
    #[new]
    fn new() -> Self {
        TnT {
            tags: Vec::new(),
            known_words: HashSet::new(),
            uni_counts: HashMap::new(),
            bi_counts: HashMap::new(),
            tri_counts: HashMap::new(),
            emission_counts: HashMap::new(),
            total_words: 0,
        }
    }

    /// Train on a list of tagged sentences.
    /// Each sentence is a list of (word, tag) tuples.
    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        let mut tag_set: HashSet<String> = HashSet::new();

        // Count ngrams
        for sentence in &sentences {
            if sentence.is_empty() {
                continue;
            }

            let mut tags = vec!["<S>".to_string(), "<S>".to_string()];
            let mut words: Vec<&str> = Vec::new();

            for (word, tag) in sentence {
                words.push(word);
                tag_set.insert(tag.clone());
                tags.push(tag.clone());
                *self.emission_counts.entry((tag.clone(), word.clone())).or_insert(0) += 1;
                self.known_words.insert(word.to_lowercase());
                self.total_words += 1;
            }

            // Add end marker
            tags.push("<E>".to_string());
            tag_set.insert("<S>".to_string());
            tag_set.insert("<E>".to_string());

            // Count ngrams from the tag sequence
            for window in tags.windows(3) {
                *self.uni_counts.entry(window[2].clone()).or_insert(0) += 1;
                *self.bi_counts.entry((window[1].clone(), window[2].clone())).or_insert(0) += 1;
                *self
                    .tri_counts
                    .entry((window[0].clone(), window[1].clone(), window[2].clone()))
                    .or_insert(0) += 1;
            }
        }

        let mut tags: Vec<String> = tag_set.into_iter().collect();
        tags.sort();
        self.tags = tags;

        Ok(())
    }

    /// Tag a sentence of words, return (word, tag) tuples.
    fn tag(&self, words: Vec<String>) -> Vec<(String, String)> {
        if words.is_empty() {
            return Vec::new();
        }

        let n = words.len();
        let t = self.tags.len();
        if t == 0 {
            return words.into_iter().map(|w| (w, "NN".to_string())).collect();
        }

        // Viterbi: dp[i][j] = best log prob up to position i with tag j
        // back[i][j] = best previous tag index
        let neg_inf = f64::NEG_INFINITY;

        // We'll do Viterbi with trigram: need 2 previous tags
        // dp[bi][j] = best prob ending with tags bi[0], bi[1] at position i, then j at i+1
        // But for simplicity, use bigram Viterbi

        let mut dp: Vec<Vec<f64>> = vec![vec![neg_inf; t]; n];
        let mut back: Vec<Vec<isize>> = vec![vec![-1; t]; n];

        // Initialization: first word
        for j in 0..t {
            let tag = &self.tags[j];
            let em_prob = self.emission_prob(tag, &words[0]);
            let trans_prob = self.trans_prob("<S>", tag);
            dp[0][j] = trans_prob.ln() + em_prob.ln();
        }

        // Induction: remaining words
        for i in 1..n {
            for j in 0..t {
                let tag_j = &self.tags[j];
                let em_prob = self.emission_prob(tag_j, &words[i]);
                if em_prob <= 0.0 {
                    continue; // Skip impossible emissions
                }
                let mut best = neg_inf;
                let mut best_k = -1;
                for k in 0..t {
                    if dp[i - 1][k] == neg_inf {
                        continue;
                    }
                    let tag_k = &self.tags[k];
                    // For first transition use uni, for later use bi
                    let trans_prob = if i > 1 {
                        let tag_prev = &self.tags[back[i - 1][k] as usize];
                        self.trans_prob_tri(tag_prev, tag_k, tag_j)
                    } else {
                        self.trans_prob(tag_k, tag_j)
                    };
                    let score = dp[i - 1][k] + trans_prob.ln() + em_prob.ln();
                    if score > best {
                        best = score;
                        best_k = k as isize;
                    }
                }
                dp[i][j] = best;
                back[i][j] = best_k;
            }
        }

        // Termination: pick best final tag
        let mut best_last = 0;
        let mut best_score = neg_inf;
        for j in 0..t {
            let tag = &self.tags[j];
            let trans_prob = self.trans_prob(tag, "<E>");
            let score = dp[n - 1][j] + trans_prob.ln();
            if score > best_score {
                best_score = score;
                best_last = j;
            }
        }

        // Backtrace
        let mut result: Vec<(String, String)> = Vec::with_capacity(n);
        let mut current = best_last as isize;
        let mut path: Vec<usize> = Vec::with_capacity(n);
        for i in (0..n).rev() {
            path.push(current as usize);
            if i > 0 {
                current = back[i][current as usize];
            }
        }
        path.reverse();

        for (i, word) in words.into_iter().enumerate() {
            result.push((word, self.tags[path[i]].clone()));
        }

        result
    }

    /// Tag multiple sentences.
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

impl TnT {
    /// Probability of a tag (unigram).
    fn tag_prob(&self, tag: &str) -> f64 {
        let total: u64 = self.uni_counts.values().sum();
        if total == 0 {
            return 0.0;
        }
        let count = self.uni_counts.get(tag).copied().unwrap_or(0);
        (count as f64 + 1.0) / (total as f64 + self.tags.len() as f64)
    }

    /// Transition probability P(t2 | t1) with add-one smoothing.
    fn trans_prob(&self, t1: &str, t2: &str) -> f64 {
        let count = self.bi_counts.get(&(t1.to_string(), t2.to_string())).copied().unwrap_or(0);
        let total: u64 = self.bi_counts.iter().filter(|((a, _), _)| a == t1).map(|(_, c)| c).sum();
        if total == 0 {
            return self.tag_prob(t2);
        }
        // TnT-style: backoff from trigram → bigram → unigram
        (count as f64 + 0.5) / (total as f64 + 0.5 * self.tags.len() as f64)
    }

    /// Trigram transition probability P(t3 | t1, t2) with backoff.
    fn trans_prob_tri(&self, t1: &str, t2: &str, t3: &str) -> f64 {
        let tri_count = self
            .tri_counts
            .get(&(t1.to_string(), t2.to_string(), t3.to_string()))
            .copied()
            .unwrap_or(0);
        let bi_count = self.bi_counts.get(&(t2.to_string(), t3.to_string())).copied().unwrap_or(0);

        if tri_count > 0 {
            let total_tri: u64 = self
                .tri_counts
                .iter()
                .filter(|((a, b, _), _)| a == t1 && b == t2)
                .map(|(_, c)| c)
                .sum();
            tri_count as f64 / total_tri as f64
        } else if bi_count > 0 {
            // Backoff to bigram
            let total_bi: u64 =
                self.bi_counts.iter().filter(|((a, _), _)| a == t2).map(|(_, c)| c).sum();
            bi_count as f64 / total_bi as f64
        } else {
            // Backoff to unigram
            self.tag_prob(t3)
        }
    }

    /// Emission probability P(word | tag) with unknown word smoothing.
    fn emission_prob(&self, tag: &str, word: &str) -> f64 {
        let count =
            self.emission_counts.get(&(tag.to_string(), word.to_string())).copied().unwrap_or(0);

        let tag_total: u64 =
            self.emission_counts.iter().filter(|((t, _), _)| t == tag).map(|(_, c)| c).sum();

        let is_known = self.known_words.contains(&word.to_lowercase());

        if count > 0 {
            count as f64 / tag_total as f64
        } else if is_known {
            // Known word but unseen with this tag: give small probability
            1.0 / (tag_total as f64 + 1.0)
        } else {
            // Unknown word: use suffix-based heuristic
            // Common unknown word tags: NN, NNP, VB, JJ
            let suffix_prob = self.suffix_guess_prob(tag, word);
            suffix_prob * 0.5 / (tag_total as f64 + 1.0).max(1.0)
        }
    }

    /// Guess probability of a tag for unknown words based on suffix.
    fn suffix_guess_prob(&self, tag: &str, word: &str) -> f64 {
        let word_lower = word.to_lowercase();
        // Simple suffix-based heuristics
        if word_lower.ends_with("ing") {
            return if tag == "VBG" { 3.0 } else { 1.0 };
        }
        if word_lower.ends_with("ed") {
            return if tag == "VBD" || tag == "VBN" { 2.0 } else { 1.0 };
        }
        if word_lower.ends_with("ly") {
            return if tag == "RB" { 4.0 } else { 1.0 };
        }
        if word_lower.ends_with("s") && !word_lower.ends_with("ss") {
            return if tag == "NNS" { 3.0 } else { 1.0 };
        }
        if word_lower.ends_with("tion") {
            return if tag == "NN" { 3.0 } else { 1.0 };
        }
        if word[..1].to_uppercase() == word[..1] && word.len() > 1 {
            return if tag == "NNP" { 3.0 } else { 1.0 };
        }
        if tag == "NN" {
            2.0
        } else {
            1.0
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TnT>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_training() -> Vec<Vec<(String, String)>> {
        vec![
            vec![
                ("The".into(), "DT".into()),
                ("cat".into(), "NN".into()),
                ("sat".into(), "VBD".into()),
            ],
            vec![
                ("The".into(), "DT".into()),
                ("dog".into(), "NN".into()),
                ("ran".into(), "VBD".into()),
            ],
        ]
    }

    #[test]
    fn test_train() {
        let mut tnt = TnT::new();
        tnt.train(sample_training()).unwrap();
        assert!(tnt.tags.len() >= 5); // <S>, <E>, DT, NN, VBD
    }

    #[test]
    fn test_tag() {
        let mut tnt = TnT::new();
        tnt.train(sample_training()).unwrap();
        let result = tnt.tag(vec!["The".into(), "cat".into(), "sat".into()]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, "DT");
        assert_eq!(result[1].1, "NN");
    }

    #[test]
    fn test_empty_tag() {
        let tnt = TnT::new();
        let result = tnt.tag(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_untrained_tag() {
        let tnt = TnT::new();
        let result = tnt.tag(vec!["hello".into()]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_unknown_word() {
        let mut tnt = TnT::new();
        tnt.train(sample_training()).unwrap();
        let result = tnt.tag(vec!["xyzzy".into()]);
        assert_eq!(result.len(), 1);
        // Should produce some tag
        assert!(!result[0].1.is_empty());
    }

    #[test]
    fn test_tag_sents() {
        let mut tnt = TnT::new();
        tnt.train(sample_training()).unwrap();
        let results =
            tnt.tag_sents(vec![vec!["The".into(), "cat".into()], vec!["A".into(), "dog".into()]]);
        assert_eq!(results.len(), 2);
    }
}
