//! `TnT` — Trigram 'n Tags, integer-tag-ID Viterbi.
//!
//! Stores transition counts in flat Vec<Vec<u64>> arrays indexed by
//! u16 tag ID instead of `HashMap`<(`SmolStr`, ...), u64>.  Eliminates
//! `SmolStr::new()` + `clone()` calls in the O(N x T^2) Viterbi loop.

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use smol_str::SmolStr;

#[pyclass(name = "TnT", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct TnT {
    /// Tag names indexed by ID.
    tags: Vec<SmolStr>,
    /// Tag name → ID lookup (built during train).
    tag_id: FxHashMap<SmolStr, u16>,
    known_words: HashSet<SmolStr>,
    /// `uni_counts`[`tag_id`] = count.
    uni_counts: Vec<u64>,
    /// Pre-computed sum of all `uni_counts`.
    uni_total: u64,
    /// `bi_counts`[`t1_id`][t2_id] = count.
    bi_counts: Vec<Vec<u64>>,
    /// Pre-computed sum per row: `bi_totals`[t1_id] = `sum(bi_counts`[t1_id][*]).
    bi_totals: Vec<u64>,
    /// `tri_counts`[`t1_id`][t2_id][`t3_id`] = count.
    tri_counts: Vec<Vec<Vec<u64>>>,
    /// `emission_counts`[`tag_id`][normalized_word] = count.
    emission_counts: Vec<HashMap<SmolStr, u64>>,
    total_words: u64,
}

impl TnT {
    /// Look up a tag ID, panicking if missing (should not happen after train).
    #[inline]
    fn tid(&self, tag: &SmolStr) -> u16 {
        self.tag_id[tag]
    }

    /// Unigram probability (add-1 smoothed).
    #[inline]
    fn tag_prob(&self, t: u16) -> f64 {
        let count = self.uni_counts[t as usize];
        (count as f64 + 1.0) / (self.uni_total as f64 + self.tags.len() as f64)
    }

    /// Bigram transition probability (add-0.5 smoothed).
    #[inline]
    fn trans_prob(&self, t1: u16, t2: u16) -> f64 {
        let count = self.bi_counts[t1 as usize][t2 as usize];
        let total = self.bi_totals[t1 as usize];
        if total == 0 {
            return self.tag_prob(t2);
        }
        (count as f64 + 0.5) / 0.5f64.mul_add(self.tags.len() as f64, total as f64)
    }

    /// Trigram transition probability with linear interpolation.
    #[inline]
    fn trans_prob_tri(&self, t1: u16, t2: u16, t3: u16) -> f64 {
        let tri = self.tri_counts[t1 as usize][t2 as usize][t3 as usize];
        let bi = self.bi_counts[t2 as usize][t3 as usize];
        if bi == 0 {
            return self.trans_prob(t2, t3);
        }
        let lambda = 0.5;
        let tri_prob = (tri as f64 + lambda) / (bi as f64 + lambda * self.tags.len() as f64);
        let bi_prob = self.trans_prob(t2, t3);
        (1.0 - lambda).mul_add(bi_prob, lambda * tri_prob)
    }

    /// Emission probability for tag→word.
    #[inline]
    fn em_prob(&self, tag: u16, word: &SmolStr) -> f64 {
        let tag_count = self.uni_counts[tag as usize];
        if tag_count == 0 {
            return 0.0;
        }
        let e = &self.emission_counts[tag as usize];
        if let Some(emit) = e.get(word) {
            return *emit as f64 / tag_count as f64;
        }
        let oov = if self.known_words.contains(word) { 0.0001 } else { 0.001 };
        oov / tag_count as f64
    }
}

#[pymethods]
impl TnT {
    #[new]
    fn new() -> Self {
        Self {
            tags: Vec::new(),
            tag_id: FxHashMap::default(),
            known_words: HashSet::new(),
            uni_counts: Vec::new(),
            uni_total: 0,
            bi_counts: Vec::new(),
            bi_totals: Vec::new(),
            tri_counts: Vec::new(),
            emission_counts: Vec::new(),
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
                tag_set.insert(t.clone());
                tags.push(t.clone());
                self.known_words.insert(SmolStr::new(word.to_lowercase()));
                self.total_words += 1;
            }
            tags.push(SmolStr::new_inline("<E>"));
            tag_set.insert(SmolStr::new_inline("<S>"));
            tag_set.insert(SmolStr::new_inline("<E>"));
        }

        // Build tag→ID mapping
        let mut tag_list: Vec<SmolStr> = tag_set.into_iter().collect();
        tag_list.sort();
        let tag_id: FxHashMap<SmolStr, u16> =
            tag_list.iter().enumerate().map(|(i, t)| (t.clone(), i as u16)).collect();
        let t = tag_list.len();

        self.tags = tag_list;
        self.tag_id = tag_id;
        self.uni_counts = vec![0u64; t];
        self.bi_counts = vec![vec![0u64; t]; t];
        self.bi_totals = vec![0u64; t];
        self.tri_counts = vec![vec![vec![0u64; t]; t]; t];
        self.emission_counts = vec![HashMap::new(); t];

        // Second pass: count
        for sentence in &sentences {
            if sentence.is_empty() {
                continue;
            }
            let mut tag_ids: Vec<u16> =
                vec![self.tid(&SmolStr::new_inline("<S>")), self.tid(&SmolStr::new_inline("<S>"))];

            for (word, tag) in sentence {
                let t_id = self.tid(&SmolStr::new(tag));
                tag_ids.push(t_id);
                let word_s = SmolStr::new(word);
                *self.emission_counts[t_id as usize].entry(word_s).or_insert(0) += 1;
            }

            tag_ids.push(self.tid(&SmolStr::new_inline("<E>")));

            for window in tag_ids.windows(3) {
                let t1 = window[0] as usize;
                let t2 = window[1] as usize;
                let t3 = window[2] as usize;
                self.uni_counts[t3] += 1;
                self.bi_counts[t1][t2] += 1;
                self.tri_counts[t1][t2][t3] += 1;
            }
        }

        // Pre-compute totals
        self.uni_total = self.uni_counts.iter().sum();
        for i in 0..t {
            self.bi_totals[i] = self.bi_counts[i].iter().sum();
        }

        Ok(())
    }

    fn tag(&self, words: Vec<String>) -> Vec<(String, String)> {
        let n = words.len();
        let t = self.tags.len();
        if t == 0 || n == 0 {
            return words.into_iter().map(|w| (w, "NN".to_string())).collect();
        }

        // Pre-compute word SmolStrs
        let word_smols: Vec<SmolStr> = words.iter().map(SmolStr::new).collect();
        let start_id = self.tid(&SmolStr::new_inline("<S>"));
        let end_id = self.tid(&SmolStr::new_inline("<E>"));

        // Emission probs for first word
        let mut em_probs: Vec<f64> = Vec::with_capacity(t);
        for j in 0..t {
            em_probs.push(self.em_prob(j as u16, &word_smols[0]));
        }

        let neg_inf = f64::NEG_INFINITY;
        let mut dp: Vec<Vec<f64>> = vec![vec![neg_inf; t]; n];
        let mut back: Vec<Vec<isize>> = vec![vec![-1isize; t]; n];

        // Init: P(tag_j | <S>) + P(word | tag_j)
        for j in 0..t {
            if em_probs[j] <= 0.0 {
                continue;
            }
            let tp = self.trans_prob(start_id, j as u16);
            if tp <= 0.0 {
                continue;
            }
            dp[0][j] = tp.ln() + em_probs[j].ln();
        }

        // Induction
        for i in 1..n {
            let w = &word_smols[i];
            for j in 0..t {
                let em = self.em_prob(j as u16, w);
                if em <= 0.0 {
                    continue;
                }
                let em_ln = em.ln();
                let mut best = neg_inf;
                let mut best_k: isize = -1;

                for k in 0..t {
                    let prev = dp[i - 1][k];
                    if prev == neg_inf {
                        continue;
                    }
                    let tp = if i > 1 {
                        let pk = back[i - 1][k];
                        if pk < 0 {
                            continue;
                        }
                        self.trans_prob_tri(pk as u16, k as u16, j as u16)
                    } else {
                        self.trans_prob(k as u16, j as u16)
                    };
                    if tp <= 0.0 {
                        continue;
                    }
                    let score = prev + tp.ln() + em_ln;
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
        let mut best_last: isize = 0;
        let mut best_score = neg_inf;
        for j in 0..t {
            let tp = self.trans_prob(j as u16, end_id);
            if tp <= 0.0 {
                continue;
            }
            let score = dp[n - 1][j] + tp.ln();
            if score > best_score {
                best_score = score;
                best_last = j as isize;
            }
        }

        // Backtrace
        let mut path: Vec<usize> = Vec::with_capacity(n);
        let mut cur = best_last;
        for i in (0..n).rev() {
            if cur < 0 {
                path.push(0);
                cur = 0;
            } else {
                path.push(cur as usize);
            }
            if i > 0 {
                cur = back[i][cur as usize];
            }
        }
        path.reverse();

        let mut result: Vec<(String, String)> = Vec::with_capacity(n);
        for (i, word) in words.into_iter().enumerate() {
            result.push((word, self.tags[path[i]].to_string()));
        }
        result
    }

    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
}

// Tests

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
            assert!(tnt.tags.contains(&SmolStr::new("DT")));
            assert!(tnt.tags.contains(&SmolStr::new("NN")));
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
