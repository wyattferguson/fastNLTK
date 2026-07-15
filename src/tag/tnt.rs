//! `TnT` — Trigram 'n Tags, Rust-accelerated.

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use smol_str::SmolStr;

#[pyclass(name = "TnT", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct TnT {
    tags: Vec<String>,
    known_words: HashSet<SmolStr>,
    uni_counts: HashMap<SmolStr, u64>,
    bi_counts: HashMap<(SmolStr, SmolStr), u64>,
    tri_counts: HashMap<(SmolStr, SmolStr, SmolStr), u64>,
    emission_counts: HashMap<(SmolStr, SmolStr), u64>,
    total_words: u64,
}

#[pymethods]
impl TnT {
    #[new]
    fn new() -> Self {
        Self {
            tags: Vec::new(),
            known_words: HashSet::new(),
            uni_counts: HashMap::new(),
            bi_counts: HashMap::new(),
            tri_counts: HashMap::new(),
            emission_counts: HashMap::new(),
            total_words: 0,
        }
    }

    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        let mut tag_set: HashSet<SmolStr> = HashSet::new();

        for sentence in &sentences {
            if sentence.is_empty() {
                continue;
            }

            let mut tags: Vec<SmolStr> =
                vec![SmolStr::new_inline("<S>"), SmolStr::new_inline("<S>")];

            for (word, tag) in sentence {
                let t = SmolStr::new(tag);
                let w = SmolStr::new(word);
                tag_set.insert(t.clone());
                tags.push(t.clone());
                *self.emission_counts.entry((t.clone(), w.clone())).or_insert(0) += 1;
                self.known_words.insert(SmolStr::new(word.to_lowercase()));
                self.total_words += 1;
            }

            tags.push(SmolStr::new_inline("<E>"));
            tag_set.insert(SmolStr::new_inline("<S>"));
            tag_set.insert(SmolStr::new_inline("<E>"));

            for window in tags.windows(3) {
                *self.uni_counts.entry(window[2].clone()).or_insert(0) += 1;
                *self.bi_counts.entry((window[1].clone(), window[2].clone())).or_insert(0) += 1;
                *self
                    .tri_counts
                    .entry((window[0].clone(), window[1].clone(), window[2].clone()))
                    .or_insert(0) += 1;
            }
        }

        let mut tag_list: Vec<SmolStr> = tag_set.into_iter().collect();
        tag_list.sort();
        self.tags = tag_list.into_iter().map(|s| s.to_string()).collect();

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

        let neg_inf = f64::NEG_INFINITY;

        // Precompute emission probs
        let word_smols: Vec<SmolStr> = words.iter().map(|w| SmolStr::new(w)).collect();
        let em_probs: Vec<f64> = self
            .tags
            .iter()
            .map(|tag| {
                let em = self.emission_prob_smol(&SmolStr::new(tag), &word_smols[0]);
                em
            })
            .collect();

        let mut dp: Vec<Vec<f64>> = vec![vec![neg_inf; t]; n];
        let mut back: Vec<Vec<isize>> = vec![vec![-1; t]; n];

        // Init
        let start_smol = SmolStr::new_inline("<S>");
        for (j, tag) in self.tags.iter().enumerate() {
            let trans_prob = self.trans_prob_smol(&start_smol, &SmolStr::new(tag));
            dp[0][j] = trans_prob.ln() + em_probs[j].ln();
        }

        // Induction
        for i in 1..n {
            let word_smol = &word_smols[i];
            for (j, tag_j) in self.tags.iter().enumerate() {
                let em_prob = self.emission_prob_smol(&SmolStr::new(tag_j), word_smol);
                if em_prob <= 0.0 {
                    continue;
                }
                let mut best = neg_inf;
                let mut best_k: isize = -1;
                for k in 0..t {
                    if (dp[i - 1][k] - neg_inf).abs() < f64::EPSILON {
                        continue;
                    }
                    let tag_k_smol = SmolStr::new(&self.tags[k]);
                    let tag_j_smol = SmolStr::new(tag_j);
                    let trans_prob = if i > 1 {
                        let prev_k = back[i - 1][k];
                        if prev_k < 0 {
                            continue;
                        }
                        let tag_prev_smol = SmolStr::new(&self.tags[prev_k as usize]);
                        self.trans_prob_tri_smol(&tag_prev_smol, &tag_k_smol, &tag_j_smol)
                    } else {
                        self.trans_prob_smol(&tag_k_smol, &tag_j_smol)
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

        // Termination
        let end_smol = SmolStr::new_inline("<E>");
        let mut best_last: isize = 0;
        let mut best_score = neg_inf;
        for (j, tag) in self.tags.iter().enumerate() {
            let trans_prob = self.trans_prob_smol(&SmolStr::new(tag), &end_smol);
            let score = dp[n - 1][j] + trans_prob.ln();
            if score > best_score {
                best_score = score;
                best_last = j as isize;
            }
        }

        // Backtrace
        let mut result: Vec<(String, String)> = Vec::with_capacity(n);
        let mut current = best_last;
        let mut path: Vec<usize> = Vec::with_capacity(n);
        for i in (0..n).rev() {
            if current < 0 {
                path.push(0);
                current = 0;
            } else {
                path.push(current as usize);
            }
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

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

impl TnT {
    fn tag_prob_smol(&self, tag: &SmolStr) -> f64 {
        let total: u64 = self.uni_counts.values().sum();
        if total == 0 {
            return 0.0;
        }
        let count = self.uni_counts.get(tag).copied().unwrap_or(0);
        (count as f64 + 1.0) / (total as f64 + self.tags.len() as f64)
    }

    fn trans_prob_smol(&self, t1: &SmolStr, t2: &SmolStr) -> f64 {
        let count = self.bi_counts.get(&(t1.clone(), t2.clone())).copied().unwrap_or(0);
        let total: u64 = self.bi_counts.iter().filter(|((a, _), _)| a == t1).map(|(_, c)| c).sum();
        if total == 0 {
            return self.tag_prob_smol(t2);
        }
        (count as f64 + 0.5) / 0.5f64.mul_add(self.tags.len() as f64, total as f64)
    }

    fn trans_prob_tri_smol(&self, t1: &SmolStr, t2: &SmolStr, t3: &SmolStr) -> f64 {
        let tri_count =
            self.tri_counts.get(&(t1.clone(), t2.clone(), t3.clone())).copied().unwrap_or(0);
        let bi_count = self.bi_counts.get(&(t2.clone(), t3.clone())).copied().unwrap_or(0);
        if bi_count == 0 {
            return self.trans_prob_smol(t2, t3);
        }
        let lambda = 0.5;
        let tri_prob =
            (tri_count as f64 + lambda) / (bi_count as f64 + lambda * self.tags.len() as f64);
        let bi_prob = self.trans_prob_smol(t2, t3);
        lambda.mul_add(tri_prob, (1.0 - lambda) * bi_prob)
    }

    fn emission_prob_smol(&self, tag: &SmolStr, word: &SmolStr) -> f64 {
        let tag_count = self.uni_counts.get(tag).copied().unwrap_or(0);
        if tag_count == 0 {
            return 0.0;
        }
        let emit_count =
            self.emission_counts.get(&(tag.clone(), word.clone())).copied().unwrap_or(0);
        if emit_count > 0 {
            return emit_count as f64 / tag_count as f64;
        }
        let oov_penalty = if self.known_words.contains(word) { 0.0001 } else { 0.001 };
        oov_penalty / tag_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sentences() -> Vec<Vec<(String, String)>> {
        vec![
            vec![
                ("the".into(), "DT".into()),
                ("cat".into(), "NN".into()),
                ("sat".into(), "VBD".into()),
            ],
            vec![
                ("the".into(), "DT".into()),
                ("dog".into(), "NN".into()),
                ("ran".into(), "VBD".into()),
            ],
        ]
    }

    #[test]
    fn test_train() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|_py| {
            let mut tnt = TnT::new();
            tnt.train(make_sentences()).unwrap();
            assert!(!tnt.tags.is_empty());
            assert!(tnt.tags.contains(&"DT".to_string()));
            assert!(tnt.tags.contains(&"NN".to_string()));
            assert!(tnt.tags.contains(&"VBD".to_string()));
        })
        .expect("GIL");
    }

    #[test]
    fn test_tag() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|_py| {
            let mut tnt = TnT::new();
            tnt.train(make_sentences()).unwrap();
            let result = tnt.tag(vec!["the".into(), "cat".into()]);
            assert_eq!(result.len(), 2);
        })
        .expect("GIL");
    }

    #[test]
    fn test_tag_sents() {
        let mut tnt = TnT::new();
        tnt.train(make_sentences()).unwrap();
        let result = tnt.tag_sents(vec![vec!["the".into(), "dog".into()]]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
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
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|_py| {
            let mut tnt = TnT::new();
            tnt.train(make_sentences()).unwrap();
            let result = tnt.tag(vec!["flibbertigibbet".into()]);
            assert_eq!(result.len(), 1);
        })
        .expect("GIL");
    }
}
