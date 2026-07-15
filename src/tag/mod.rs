//! POS tagging — Rust-accelerated taggers matching NLTK's API.

pub mod hmm;
pub mod perceptron;
pub mod sequential;
pub mod tnt;

use pyo3::prelude::*;

/// Register all tagger classes with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<perceptron::PerceptronTagger>()?;
    m.add_class::<sequential::DefaultTagger>()?;
    m.add_class::<sequential::UnigramTagger>()?;
    m.add_class::<sequential::BigramTagger>()?;
    m.add_class::<sequential::TrigramTagger>()?;
    m.add_class::<sequential::AffixTagger>()?;
    m.add_class::<sequential::RegexpTagger>()?;
    m.add_class::<tnt::TnT>()?;
    m.add_class::<hmm::HiddenMarkovModelTagger>()?;
    Ok(())
}
