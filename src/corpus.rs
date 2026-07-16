//! Corpus readers — Rust-accelerated file I/O + tokenization.

use std::fs;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;

use crate::tokenize::{self};

// PlaintextCorpusReader

#[pyclass(name = "PlaintextCorpusReader", module = "fastnltk._rust")]
pub struct PlaintextCorpusReader {
    root: PathBuf,
    fileids: Vec<String>,
}

#[pymethods]
impl PlaintextCorpusReader {
    #[new]
    #[pyo3(signature = (root, fileids=None, _encoding=None))]
    fn new(root: &str, fileids: Option<Vec<String>>, _encoding: Option<&str>) -> PyResult<Self> {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            return Err(PyFileNotFoundError::new_err(format!(
                "Corpus directory not found: {root}"
            )));
        }

        let resolved: Vec<String> = fileids.unwrap_or_else(|| {
            let mut ids: Vec<String> = Vec::new();
            if let Ok(entries) = fs::read_dir(root_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(name) = path.file_name() {
                            ids.push(name.to_string_lossy().to_string());
                        }
                    }
                }
            }
            ids.sort();
            ids
        });

        Ok(Self { root: root_path.to_path_buf(), fileids: resolved })
    }

    /// Return the list of file IDs in this corpus.
    fn fileids(&self) -> Vec<String> {
        self.fileids.clone()
    }

    /// Read the raw contents of a file.
    #[pyo3(signature = (fileids=None))]
    fn raw(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<String>> {
        let ids = fileids.unwrap_or_else(|| self.fileids.clone());
        let mut contents = Vec::with_capacity(ids.len());
        for fid in &ids {
            let path = self.root.join(fid);
            let text = fs::read_to_string(&path)
                .map_err(|e| PyFileNotFoundError::new_err(format!("Cannot read {fid}: {e}")))?;
            contents.push(text);
        }
        Ok(contents)
    }

    /// Read words (tokenized using `word_tokenize`).
    #[pyo3(signature = (fileids=None))]
    fn words(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<String>> {
        let contents = self.raw(fileids)?;
        let mut all_words = Vec::new();
        Python::try_attach(|py| {
            for text in &contents {
                let mut words = tokenize::word_tokenize_py(py, text, "english", false)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                all_words.append(&mut words);
            }
            Ok(all_words)
        })
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Failed to acquire Python GIL"))?
    }

    /// Read sentences (tokenized using `sent_tokenize`).
    #[pyo3(signature = (fileids=None))]
    fn sents(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<Vec<String>>> {
        let contents = self.raw(fileids)?;
        let mut all_sents = Vec::new();
        Python::try_attach(|py| {
            for text in &contents {
                let sentences = tokenize::sent_tokenize_py(py, text, "english")
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                for sent in &sentences {
                    let words = tokenize::word_tokenize_py(py, sent, "english", false)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    all_sents.push(words);
                }
            }
            Ok(all_sents)
        })
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Failed to acquire Python GIL"))?
    }

    /// Read the number of words in the corpus.
    #[pyo3(signature = (fileids=None))]
    fn word_count(&self, fileids: Option<Vec<String>>) -> PyResult<usize> {
        self.words(fileids).map(|w| w.len())
    }

    /// Read the number of sentences.
    #[pyo3(signature = (fileids=None))]
    fn sent_count(&self, fileids: Option<Vec<String>>) -> PyResult<usize> {
        self.sents(fileids).map(|s| s.len())
    }
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlaintextCorpusReader>()?;
    Ok(())
}
