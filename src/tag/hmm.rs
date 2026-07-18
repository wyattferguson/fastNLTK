//! HMM tagger — supervised Hidden Markov Model for POS tagging.
//!
//! Estimates transition and emission probabilities from labeled training data.
//! Uses Viterbi decoding with integer tag IDs for O(N × T²) without allocations.

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use smol_str::SmolStr;

#[pyclass(name = "HiddenMarkovModelTagger", module = "fastnltk._rust")]
pub struct HiddenMarkovModelTagger {
    /// Transition log-probabilities: `trans_mat[from_tag_id][to_tag_id]`
    trans_mat: Vec<Vec<f64>>,
    /// Emission log-probabilities: `emission[tag_id][word_hash]` → log_prob
    emission: Vec<HashMap<u64, f64>>,
    /// Tag → index for Viterbi
    tag_index: HashMap<SmolStr, usize>,
    /// All known tags (index → tag name)
    tag_names: Vec<SmolStr>,
    /// Start tag ID (always position 0)
    start_id: usize,
    /// End tag ID (always position 1)
    end_id: usize,
    /// Whether model has been trained
    trained: bool,
}

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Simple string hash for emission lookup (no need for crypto security).
fn word_hash(w: &str) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write(w.as_bytes());
    h.finish()
}

#[pymethods]
impl HiddenMarkovModelTagger {
    #[new]
    fn new() -> Self {
        let mut tag_index: HashMap<SmolStr, usize> = HashMap::new();
        // Reserve IDs 0 and 1 for <s>, </s>
        let tag_names = vec![SmolStr::new_inline("<s>"), SmolStr::new_inline("</s>")];
        tag_index.insert(SmolStr::new_inline("<s>"), 0);
        tag_index.insert(SmolStr::new_inline("</s>"), 1);
        Self {
            trans_mat: Vec::new(),
            emission: Vec::new(),
            tag_index,
            tag_names,
            start_id: 0,
            end_id: 1,
            trained: false,
        }
    }

    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        if sentences.is_empty() {
            return Err(PyValueError::new_err("No training data"));
        }

        // Build tag vocabulary (beyond the built-in <s>, </s>)
        for sent in &sentences {
            for (_, tag) in sent {
                let t = SmolStr::new(tag);
                let next_id = self.tag_names.len();
                self.tag_index.entry(t).or_insert_with(|| {
                    self.tag_names.push(SmolStr::new(tag));
                    next_id
                });
            }
        }

        let k = self.tag_names.len();

        // Count transitions (int -> int), emissions (int -> word), and tag occurrences.
        let mut trans_counts: Vec<Vec<f64>> = vec![vec![0.0f64; k]; k];
        let mut tag_totals: Vec<f64> = vec![0.0f64; k]; // transition totals (from tag)
        let mut tag_occs: Vec<f64> = vec![0.0f64; k]; // occurrence totals (as current tag)
        let mut emiss_counts_raw: Vec<HashMap<u64, f64>> = vec![HashMap::new(); k];

        for sent in &sentences {
            let mut prev_id = self.start_id;
            for (word, tag) in sent {
                let tag_id = self.tag_index[&SmolStr::new(tag)];
                trans_counts[prev_id][tag_id] += 1.0;
                tag_totals[prev_id] += 1.0;
                tag_occs[tag_id] += 1.0;
                *emiss_counts_raw[tag_id].entry(word_hash(word)).or_insert(0.0) += 1.0;
                prev_id = tag_id;
            }
            // Sentence end
            trans_counts[prev_id][self.end_id] += 1.0;
            tag_totals[prev_id] += 1.0;
        }

        // Convert counts to log-probabilities with add-1 smoothing
        self.trans_mat = vec![vec![NEG_INF; k]; k];
        for from in 0..k {
            let total = tag_totals[from];
            if total == 0.0 {
                continue;
            }
            for to in 0..k {
                let count = trans_counts[from][to];
                if count > 0.0 || from == self.start_id {
                    let prob = (count + 1.0) / (total + k as f64);
                    self.trans_mat[from][to] = prob.ln();
                }
            }
        }

        self.emission = vec![HashMap::new(); k];
        for tag_id in 0..k {
            let total = tag_occs[tag_id];
            if total == 0.0 {
                continue;
            }
            for (wh, count) in &emiss_counts_raw[tag_id] {
                let prob = (count + 1.0) / (total + k as f64);
                self.emission[tag_id].insert(*wh, prob.ln());
            }
        }

        self.trained = true;
        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> PyResult<Vec<(String, String)>> {
        if !self.trained {
            return Err(PyValueError::new_err("Model not trained"));
        }
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = self.tag_names.len();
        if k < 3 {
            return Err(PyValueError::new_err("No tags in model"));
        }

        // Pre-compute word hashes
        let word_hashes: Vec<u64> = tokens.iter().map(|w| word_hash(w)).collect();

        // Viterbi: dp[i][j] = best log-prob for prefix[0..=i] ending in tag j
        let mut dp: Vec<Vec<f64>> = vec![vec![NEG_INF; k]; n];
        let mut back: Vec<Vec<usize>> = vec![vec![0; k]; n];

        // Init: first word
        let wh0 = word_hashes[0];
        for j in 2..k {
            // skip <s> (0) and </s> (1) — they're not real tags
            let trans = self.trans_mat[self.start_id][j];
            if trans == NEG_INF {
                continue;
            }
            let emiss = self.emission[j].get(&wh0).copied().unwrap_or(NEG_INF);
            if emiss == NEG_INF {
                continue;
            }
            dp[0][j] = trans + emiss;
        }

        // Induction
        for i in 1..n {
            let wh = word_hashes[i];
            for j in 2..k {
                let emiss = self.emission[j].get(&wh).copied().unwrap_or(NEG_INF);
                if emiss == NEG_INF {
                    continue;
                }
                let mut best = NEG_INF;
                let mut best_p = 0usize;
                for p in 2..k {
                    let prev = dp[i - 1][p];
                    if prev == NEG_INF {
                        continue;
                    }
                    let trans = self.trans_mat[p][j];
                    if trans == NEG_INF {
                        continue;
                    }
                    let score = prev + trans;
                    if score > best {
                        best = score;
                        best_p = p;
                    }
                }
                dp[i][j] = best + emiss;
                back[i][j] = best_p;
            }
        }

        // Termination
        let mut best_last = 2usize;
        let mut best_score = NEG_INF;
        for j in 2..k {
            let score = dp[n - 1][j];
            if score == NEG_INF {
                continue;
            }
            let end_trans = self.trans_mat[j][self.end_id];
            if end_trans == NEG_INF {
                continue;
            }
            let total = score + end_trans;
            if total > best_score {
                best_score = total;
                best_last = j;
            }
        }

        // Backtrace
        let mut path: Vec<usize> = Vec::with_capacity(n);
        let mut cur = best_last;
        path.push(cur);
        for i in (1..n).rev() {
            cur = back[i][cur];
            path.push(cur);
        }
        path.reverse();

        let result: Vec<(String, String)> = tokens
            .into_iter()
            .zip(path.into_iter().map(|j| self.tag_names[j].to_string()))
            .collect();

        Ok(result)
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> PyResult<Vec<Vec<(String, String)>>> {
        sentences.into_iter().map(|t| self.tag(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_simple() {
        let mut tagger = HiddenMarkovModelTagger::new();
        let training = vec![vec![
            ("the".to_string(), "DT".to_string()),
            ("cat".to_string(), "NN".to_string()),
        ]];
        tagger.train(training).unwrap();
        let result = tagger.tag(vec!["the".to_string(), "cat".to_string()]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, "DT");
        assert_eq!(result[1].1, "NN");
    }

    #[test]
    fn test_hmm_empty() {
        let tagger = HiddenMarkovModelTagger::new();
        let result = tagger.tag(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hmm_not_trained() {
        let tagger = HiddenMarkovModelTagger::new();
        assert!(tagger.tag(vec!["hello".to_string()]).is_err());
    }

    #[test]
    fn test_hmm_multiple_sents() {
        let mut tagger = HiddenMarkovModelTagger::new();
        tagger
            .train(vec![
                vec![("the".to_string(), "DT".to_string()), ("cat".to_string(), "NN".to_string())],
                vec![("a".to_string(), "DT".to_string()), ("dog".to_string(), "NN".to_string())],
            ])
            .unwrap();
        let result = tagger.tag_sents(vec![
            vec!["the".to_string(), "cat".to_string()],
            vec!["a".to_string(), "dog".to_string()],
        ]);
        assert!(result.is_ok());
        let sents = result.unwrap();
        assert_eq!(sents.len(), 2);
    }
}
