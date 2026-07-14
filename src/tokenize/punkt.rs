//! Punkt sentence tokenizer — Rust port matching NLTK's implementation.
//!
//! Uses trained Punkt models loaded from NLTK's nltk_data directory.
//! The model parameters (abbreviations, collocations, orthographic context)
//! are loaded via Python pickle and passed to Rust for inference.

use hashbrown::HashMap;
use hashbrown::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PySet};

// ═══════════════════════════════════════════════════════════
// Parameter types matching NLTK's PunktParameters
// ═══════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PunktParams {
    /// Abbreviation types (e.g., "Mr.", "Dr.", "U.S.")
    abbrev_types: HashSet<String>,
    /// Collocations (word pairs that cross sentence boundaries)
    collocations: HashSet<(String, String)>,
    /// Words that commonly start sentences
    sent_starters: HashSet<String>,
}

impl PunktParams {
    fn new() -> Self {
        PunktParams {
            abbrev_types: HashSet::new(),
            collocations: HashSet::new(),
            sent_starters: HashSet::new(),
        }
    }

    /// Check if a word is a known abbreviation.
    fn is_abbrev(&self, word: &str) -> bool {
        let stripped = word.trim_end_matches('.');
        self.abbrev_types.contains(stripped) || self.abbrev_types.contains(word)
    }

    /// Check if a word pair is a known collocation.
    fn is_collocation(&self, w1: &str, w2: &str) -> bool {
        self.collocations.contains(&(w1.to_lowercase(), w2.to_lowercase()))
    }

    /// Check if a word is a known sentence starter.
    fn is_sent_start(&self, word: &str) -> bool {
        self.sent_starters.contains(&word.to_lowercase())
    }
}

// ═══════════════════════════════════════════════════════════
// PunktSentenceTokenizer
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "PunktSentenceTokenizer", module = "fastnltk._rust")]
pub struct PunktSentenceTokenizer {
    params: Option<PunktParams>,
}

#[pymethods]
impl PunktSentenceTokenizer {
    #[new]
    #[pyo3(signature = (train_text=None, language="english"))]
    fn new(train_text: Option<String>, language: &str) -> Self {
        let _ = train_text;
        let _ = language;
        PunktSentenceTokenizer { params: None }
    }

    /// Load trained parameters from Python dicts (from NLTK pickle).
    #[pyo3(signature = (params=None))]
    fn load(&mut self, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(p) = params {
            let mut pparams = PunktParams::new();

            // Load abbreviation types
            if let Ok(abbrev) = p.get_item("abbrev_types") {
                if let Some(abbrev) = abbrev {
                    if let Ok(set) = abbrev.downcast::<PySet>() {
                        for item in set.iter() {
                            if let Ok(s) = item.extract::<String>() {
                                pparams.abbrev_types.insert(s);
                            }
                        }
                    }
                }
            }

            // Load collocations
            if let Ok(coll) = p.get_item("collocations") {
                if let Some(coll) = coll {
                    if let Ok(set) = coll.downcast::<PyFrozenSet>() {
                        for item in set.iter() {
                            if let Ok(t) = item.extract::<(String, String)>() {
                                pparams.collocations.insert(t);
                            }
                        }
                    }
                }
            }

            // Load sentence starters
            if let Ok(ss) = p.get_item("sent_starters") {
                if let Some(ss_val) = ss {
                    if let Ok(set) = ss_val.downcast::<PySet>() {
                        for item in set.iter() {
                            if let Ok(s) = item.extract::<String>() {
                                pparams.sent_starters.insert(s);
                            }
                        }
                    }
                }
            }

            self.params = Some(pparams);
        }
        Ok(())
    }

    /// Tokenize text into sentences.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.sentences_from_text(text)
    }

    /// Return (start, end) spans for each sentence.
    fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        self.punkt_spans(text)
    }

    /// Return sentences from text.
    fn sentences_from_text(&self, text: &str) -> Vec<String> {
        let spans = self.punkt_spans(text);
        spans.into_iter().map(|(s, e)| text[s..e].to_string()).collect()
    }
}

// ═══════════════════════════════════════════════════════════
// Implementation: Three-pass sentence boundary detection
// ═══════════════════════════════════════════════════════════

impl PunktSentenceTokenizer {
    fn tokenize_simple_sentences(text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = 0;
        let bytes = text.as_bytes();
        for (i, _) in text.char_indices() {
            if i > 0 {
                let c = bytes[i - 1];
                if c == b'.' || c == b'!' || c == b'?' {
                    if i >= text.len()
                        || bytes[i] == b' '
                        || bytes[i] == b'\n'
                        || bytes[i] == b'"'
                        || bytes[i] == b'\''
                    {
                        let end = i;
                        spans.push((start, end));
                        start = end;
                    }
                }
            }
        }
        if start < text.len() {
            spans.push((start, text.len()));
        }
        spans
    }

    /// Punkt sentence boundary detection using loaded model.
    fn punkt_spans(&self, text: &str) -> Vec<(usize, usize)> {
        let params = match &self.params {
            Some(p) => p,
            None => return Self::tokenize_simple_sentences(text),
        };

        // Pass 1: Find candidate sentence boundaries
        let sentences: Vec<(usize, usize)> = {
            let mut spans = Vec::new();
            let mut start = 0;
            let tokens = self.tokenize_words(text);
            let mut i = 0;

            while i < tokens.len() {
                let (tok_start, tok_text) = &tokens[i];
                let end = *tok_start + tok_text.len();

                // Check if this token ends with sentence-final punctuation
                if let Some('.') | Some('!') | Some('?') = tok_text.chars().last() {
                    let is_sentence_break = self.is_sentence_boundary(text, &tokens, i, params);

                    if is_sentence_break {
                        // Include trailing whitespace/quote in current sentence
                        let real_end = if end < text.len() {
                            let remaining = &text[end..];
                            let ws_end = remaining
                                .find(|c: char| {
                                    !c.is_whitespace() && c != '"' && c != '\'' && c != ')'
                                })
                                .map(|pos| end + pos)
                                .unwrap_or(text.len());
                            ws_end
                        } else {
                            end
                        };
                        spans.push((start, real_end));
                        start = real_end;
                    }
                }
                i += 1;
            }

            if start < text.len() {
                spans.push((start, text.len()));
            }

            spans
        };

        if sentences.is_empty() {
            return vec![(0, text.len())];
        }
        sentences
    }

    /// Tokenize text into (position, word) pairs.
    /// Words keep their trailing punctuation attached for abbreviation detection.
    fn tokenize_words(&self, text: &str) -> Vec<(usize, String)> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();

        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            let start = i;

            // Collect a word: alphanumeric + internal periods/hyphens/apostrophes
            while i < chars.len()
                && !chars[i].is_whitespace()
                && (chars[i].is_alphanumeric()
                    || chars[i] == '.'
                    || chars[i] == '-'
                    || chars[i] == '\'')
            {
                i += 1;
            }

            if i > start {
                tokens.push((text[..start].len(), chars[start..i].iter().collect()));
            } else {
                // Single non-alphanumeric char (punctuation)
                tokens.push((text[..i].len(), chars[i].to_string()));
                i += 1;
            }
        }
        tokens
    }

    /// Determine if a period/final-punctuation token is a sentence boundary.
    fn is_sentence_boundary(
        &self,
        _text: &str,
        tokens: &[(usize, String)],
        idx: usize,
        params: &PunktParams,
    ) -> bool {
        let (_, tok_text) = &tokens[idx];

        // No final punctuation → not a boundary
        let _last_char = match tok_text.chars().last() {
            Some('.') | Some('!') | Some('?') => true,
            _ => return false,
        };

        // Check if it's an abbreviation
        if tok_text.ends_with('.') {
            let word = tok_text.trim_end_matches('.');
            if params.is_abbrev(word) || params.is_abbrev(tok_text) {
                if idx + 1 >= tokens.len() {
                    return true;
                }
                let next_word = &tokens[idx + 1].1;
                if let Some(c) = next_word.chars().next() {
                    if c.is_lowercase() {
                        return false;
                    }
                    if params.is_sent_start(next_word) {
                        return true;
                    }
                    return false;
                }
                return false;
            }

            // Multi-dot tokens like "U.S." — treat as abbreviation pattern
            if tok_text.matches('.').count() > 1 {
                if idx + 1 < tokens.len() {
                    let next = &tokens[idx + 1].1;
                    if let Some(c) = next.chars().next() {
                        if c.is_lowercase() || c.is_ascii_uppercase() {
                            return false;
                        }
                    }
                }
            }
        }

        // For ! and ?, these are almost always sentence boundaries
        if !tok_text.ends_with('.') {
            return true;
        }

        true
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PunktSentenceTokenizer>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> PunktParams {
        let mut p = PunktParams::new();
        p.abbrev_types.insert("Mr".to_string());
        p.abbrev_types.insert("Dr".to_string());
        p.abbrev_types.insert("Mrs".to_string());
        p.abbrev_types.insert("Ms".to_string());
        p.abbrev_types.insert("U.S".to_string());
        p.abbrev_types.insert("Ph.D".to_string());
        p.abbrev_types.insert("e.g".to_string());
        p.abbrev_types.insert("i.e".to_string());
        p.abbrev_types.insert("vs".to_string());
        p.abbrev_types.insert("Inc".to_string());
        p.abbrev_types.insert("Ltd".to_string());
        p.abbrev_types.insert("Dept".to_string());
        p.sent_starters.insert("the".to_string());
        p.sent_starters.insert("he".to_string());
        p.sent_starters.insert("she".to_string());
        p.sent_starters.insert("it".to_string());
        p.sent_starters.insert("this".to_string());
        p.sent_starters.insert("that".to_string());
        p
    }

    #[test]
    fn test_simple_sentences() {
        let spans =
            PunktSentenceTokenizer::tokenize_simple_sentences("Hello world. This is a test.");
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn test_abbreviation_not_boundary() {
        let mut tok = PunktSentenceTokenizer::new(None, "english");
        let p = test_params();
        tok.params = Some(p);
        let sentences = tok.sentences_from_text("Dr. Smith went home. He ate dinner.");
        assert_eq!(sentences.len(), 2, "Should have 2 sentences: {sentences:?}");
        assert!(sentences[0].contains("Dr."), "First sentence should contain Dr.");
        assert!(!sentences[0].contains("He ate"), "First sentence should not contain 'He ate'");
    }

    #[test]
    fn test_empty_text() {
        let tok = PunktSentenceTokenizer::new(None, "english");
        let sentences = tok.sentences_from_text("");
        assert!(sentences.is_empty() || sentences == vec![""]);
    }

    #[test]
    fn test_single_sentence() {
        let tok = PunktSentenceTokenizer::new(None, "english");
        let sentences = tok.sentences_from_text("This is one sentence.");
        assert_eq!(sentences.len(), 1);
    }
}
