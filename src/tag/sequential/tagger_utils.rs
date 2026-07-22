//! Shared utilities for sequential taggers.

use std::hash::Hasher;

use hashbrown::HashMap as FastMap;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyList, PyListMethods};
use pyo3::Bound;
use rustc_hash::FxHasher;
use smol_str::SmolStr;

/// Hash a word string to a u64 using FxHash.
pub fn hash_word(w: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(w.as_bytes());
    h.finish()
}

/// Ensure a tag string has an integer ID, inserting if new.
pub fn ensure_tag(
    tag: &str,
    tag_to_id: &mut FastMap<SmolStr, u16>,
    tag_names: &mut Vec<SmolStr>,
) -> u16 {
    let next = tag_to_id.len() as u16;
    let id = *tag_to_id.entry(SmolStr::new(tag)).or_insert(next);
    if id == next {
        tag_names.push(SmolStr::new(tag));
    }
    id
}

/// Run a tagging fn over sentences, optionally parallel.
pub fn maybe_parallel<T, F>(sentences: Vec<Vec<String>>, f: F) -> Vec<Vec<(String, T)>>
where
    T: Clone + Default + Send,
    F: Fn(Vec<String>) -> Vec<(String, T)> + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        sentences.into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        sentences.into_iter().map(f).collect()
    }
}

/// Evaluate a tagger against tagged sentences.
pub fn evaluate<F>(sentences: &Bound<'_, PyList>, tag: F) -> f64
where
    F: Fn(Vec<String>) -> Vec<(String, String)>,
{
    let mut correct = 0u64;
    let mut total = 0u64;
    for item in sentences.iter() {
        let sent: Vec<(String, String)> = item.extract().unwrap_or_default();
        let words: Vec<String> = sent.iter().map(|(w, _)| w.clone()).collect();
        let gold_tags: Vec<String> = sent.iter().map(|(_, t)| t.clone()).collect();
        let pred = tag(words);
        for (p, g) in pred.iter().zip(gold_tags.iter()) {
            if &p.1 == g {
                correct += 1;
            }
        }
        total += sent.len() as u64;
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}
