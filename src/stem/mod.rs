//! Stemming — Rust-accelerated stemmers matching NLTK's API.

pub mod arlstem;
pub mod cistem;
pub mod isri;
pub mod lancaster;
pub mod porter;
pub mod regexp;
pub mod rslp;
pub mod snowball;
pub mod wordnet;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<snowball::SnowballStemmer>()?;
    m.add_class::<porter::PorterStemmer>()?;
    m.add_class::<lancaster::LancasterStemmer>()?;
    m.add_class::<isri::ISRIStemmer>()?;
    m.add_class::<cistem::Cistem>()?;
    m.add_class::<rslp::RSLPStemmer>()?;
    m.add_class::<regexp::RegexpStemmer>()?;
    m.add_class::<wordnet::WordNetLemmatizer>()?;
    m.add_class::<arlstem::ARLSTem>()?;
    m.add_class::<arlstem::ARLSTem2>()?;
    Ok(())
}
