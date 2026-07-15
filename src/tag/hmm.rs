//! HMM tagger — supervised Hidden Markov Model for POS tagging.
//!
//! Estimates transition and emission probabilities from labeled training data.
//! Uses Viterbi decoding for prediction.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyclass(name = "HiddenMarkovModelTagger", module = "fastnltk._rust")]
pub struct HiddenMarkovModelTagger {
    /// Transition log-probabilities: (prev_tag, tag) → log_prob
    transitions: FxHashMap<(String, String), f64>,
    /// Emission log-probabilities: (tag, word) → log_prob
    emissions: FxHashMap<(String, String), f64>,
    /// Tag → index for Viterbi
    tag_index: FxHashMap<String, usize>,
    /// All known tags
    tags: Vec<String>,
    /// Start tag
    start_tag: String,
    /// Whether model has been trained
    trained: bool,
}

#[pymethods]
impl HiddenMarkovModelTagger {
    #[new]
    fn new() -> Self {
        Self {
            transitions: FxHashMap::default(),
            emissions: FxHashMap::default(),
            tag_index: FxHashMap::default(),
            tags: Vec::new(),
            start_tag: "<s>".to_string(),
            trained: false,
        }
    }

    fn train(&mut self, sentences: Vec<Vec<(String, String)>>) -> PyResult<()> {
        if sentences.is_empty() {
            return Err(PyValueError::new_err("No training data"));
        }

        // Collect tag vocabulary
        let mut tag_set = rustc_hash::FxHashSet::default();
        for sent in &sentences {
            for (_, tag) in sent {
                tag_set.insert(tag.clone());
            }
        }
        self.tags = tag_set.into_iter().collect();
        self.tags.sort();
        for (i, tag) in self.tags.iter().enumerate() {
            self.tag_index.insert(tag.clone(), i);
        }

        // Count transitions and emissions
        let mut transition_counts: FxHashMap<(String, String), f64> = FxHashMap::default();
        let mut emission_counts: FxHashMap<(String, String), f64> = FxHashMap::default();
        let mut tag_totals: FxHashMap<String, f64> = FxHashMap::default();

        for sent in &sentences {
            let mut prev_tag = self.start_tag.clone();
            for (word, tag) in sent {
                *transition_counts.entry((prev_tag.clone(), tag.clone())).or_insert(0.0) += 1.0;
                *tag_totals.entry(prev_tag.clone()).or_insert(0.0) += 1.0;
                *emission_counts.entry((tag.clone(), word.clone())).or_insert(0.0) += 1.0;
                prev_tag = tag.clone();
            }
            // Transition to end
            *transition_counts.entry((prev_tag, "</s>".to_string())).or_insert(0.0) += 1.0;
            *tag_totals.entry("</s>".to_string()).or_insert(0.0) += 1.0;
        }

        // Convert counts to log-probabilities with add-1 smoothing
        let tag_count = self.tags.len() as f64;
        for ((prev, tag), count) in &transition_counts {
            let total = tag_totals.get(prev).copied().unwrap_or(1.0);
            self.transitions
                .insert((prev.clone(), tag.clone()), ((count + 1.0) / (total + tag_count)).ln());
        }
        for ((tag, word), count) in &emission_counts {
            let total = tag_totals.get(tag).copied().unwrap_or(1.0);
            self.emissions
                .insert((tag.clone(), word.clone()), ((count + 1.0) / (total + tag_count)).ln());
        }

        self.trained = true;
        Ok(())
    }

    fn tag(&self, tokens: Vec<String>) -> PyResult<Vec<(String, String)>> {
        if !self.trained {
            return Err(PyValueError::new_err("Model not trained"));
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if self.tags.is_empty() {
            return Err(PyValueError::new_err("No tags in model"));
        }

        let n = tokens.len();
        let k = self.tags.len();

        // Viterbi matrix: log-probabilities
        let mut v: Vec<Vec<f64>> = vec![vec![f64::NEG_INFINITY; k]; n];
        // Backpointers
        let mut bp: Vec<Vec<usize>> = vec![vec![0; k]; n];

        // Initialize first token
        let w0 = &tokens[0];
        for (j, tag) in self.tags.iter().enumerate() {
            let trans = self
                .transitions
                .get(&(self.start_tag.clone(), tag.clone()))
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let emiss = self
                .emissions
                .get(&(tag.clone(), w0.clone()))
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            v[0][j] = trans + emiss;
        }

        // Fill Viterbi
        for i in 1..n {
            let wi = &tokens[i];
            for (j, tag_j) in self.tags.iter().enumerate() {
                let emiss = self
                    .emissions
                    .get(&(tag_j.clone(), wi.clone()))
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for (p, tag_p) in self.tags.iter().enumerate() {
                    let trans = self
                        .transitions
                        .get(&(tag_p.clone(), tag_j.clone()))
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    let score = v[i - 1][p] + trans;
                    if score > best_score {
                        best_score = score;
                        best_prev = p;
                    }
                }
                v[i][j] = best_score + emiss;
                bp[i][j] = best_prev;
            }
        }

        // Backtrack
        let mut best_last = 0;
        let mut best_last_score = f64::NEG_INFINITY;
        for (j, tag_j) in self.tags.iter().enumerate() {
            let end_trans = self
                .transitions
                .get(&(tag_j.clone(), "</s>".to_string()))
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let score = v[n - 1][j] + end_trans;
            if score > best_last_score {
                best_last_score = score;
                best_last = j;
            }
        }

        let mut path = vec![best_last];
        for i in (1..n).rev() {
            path.push(bp[i][path[path.len() - 1]]);
        }
        path.reverse();

        let result: Vec<(String, String)> =
            tokens.into_iter().zip(path.into_iter().map(|j| self.tags[j].clone())).collect();

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
        assert!(result.is_err()); // Not trained
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
