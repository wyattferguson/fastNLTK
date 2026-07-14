//! Sequential taggers — Rust-accelerated lookup-based POS taggers.
//!
//! Implements NLTK's sequential backoff taggers:
//!   - `DefaultTagger`: assign same tag to every word
//!   - `NgramTagger` (Unigram/Bigram/Trigram): lookup tag from training data
//!   - `AffixTagger`: tag based on word suffix/prefix
//!   - `RegexpTagger`: tag based on regex pattern match on the word

pub mod taggers;
pub use taggers::{UnigramTagger, BigramTagger, TrigramTagger, AffixTagger, RegexpTagger};

use pyo3::prelude::*;

/// `DefaultTagger` — assign the same tag to every word.
#[pyclass(name = "DefaultTagger", module = "fastnltk._rust")]
pub struct DefaultTagger {
    tag: String,
}

#[pymethods]
impl DefaultTagger {
    #[new]
    fn new(tag: &str) -> Self { Self { tag: tag.to_string() } }
    fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        tokens.into_iter().map(|w| (w, self.tag.clone())).collect()
    }
    fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
        sentences.into_iter().map(|s| self.tag(s)).collect()
    }
    fn evaluate(&self, tagged_sentences: Vec<Vec<(String, String)>>) -> f64 {
        let total = tagged_sentences.iter().map(std::vec::Vec::len).sum::<usize>();
        if total == 0 { return 0.0; }
        let correct = tagged_sentences.iter().flatten().filter(|(_, t)| t == &self.tag).count();
        correct as f64 / total as f64
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DefaultTagger>()?;
    m.add_class::<taggers::UnigramTagger>()?;
    m.add_class::<taggers::BigramTagger>()?;
    m.add_class::<taggers::TrigramTagger>()?;
    m.add_class::<taggers::AffixTagger>()?;
    m.add_class::<taggers::RegexpTagger>()?;
    Ok(())
}
