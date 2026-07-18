//! Corpus readers — Rust-accelerated file I/O + tokenization.

use std::fs;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;

use crate::tokenize;

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

    fn fileids(&self) -> Vec<String> {
        self.fileids.clone()
    }

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

    #[pyo3(signature = (fileids=None))]
    fn word_count(&self, fileids: Option<Vec<String>>) -> PyResult<usize> {
        self.words(fileids).map(|w| w.len())
    }

    #[pyo3(signature = (fileids=None))]
    fn sent_count(&self, fileids: Option<Vec<String>>) -> PyResult<usize> {
        self.sents(fileids).map(|s| s.len())
    }
}

// ── TaggedCorpusReader ──
//
// Reads word/tag pairs from files in the format: word/tag word/tag ...

#[pyclass(name = "TaggedCorpusReader", module = "fastnltk._rust")]
pub struct TaggedCorpusReader {
    root: PathBuf,
    fileids: Vec<String>,
    sep: String,
}

#[pymethods]
impl TaggedCorpusReader {
    #[new]
    #[pyo3(signature = (root, fileids, sep="/"))]
    fn new(root: &str, fileids: Vec<String>, sep: &str) -> PyResult<Self> {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            return Err(PyFileNotFoundError::new_err(format!(
                "Corpus directory not found: {root}"
            )));
        }
        Ok(Self { root: root_path.to_path_buf(), fileids, sep: sep.to_string() })
    }

    fn fileids(&self) -> Vec<String> {
        self.fileids.clone()
    }

    /// Raw text of files.
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

    /// Return tagged words as list of (word, tag) tuples.
    #[pyo3(signature = (fileids=None))]
    fn tagged_words(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<(String, String)>> {
        let raw = self.raw(fileids)?;
        let sep = &self.sep;
        let mut result = Vec::new();
        for text in &raw {
            for token in text.split_whitespace() {
                if let Some(pos) = token.rfind(sep) {
                    let word = token[..pos].to_string();
                    let tag = token[pos + sep.len()..].to_string();
                    result.push((word, tag));
                }
            }
        }
        Ok(result)
    }

    /// Return tagged sentences.
    #[pyo3(signature = (fileids=None))]
    fn tagged_sents(
        &self,
        fileids: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<(String, String)>>> {
        let raw = self.raw(fileids)?;
        let sep = &self.sep;
        let mut result = Vec::new();
        for text in &raw {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let mut sent = Vec::new();
                for token in line.split_whitespace() {
                    if let Some(pos) = token.rfind(sep) {
                        let word = token[..pos].to_string();
                        let tag = token[pos + sep.len()..].to_string();
                        sent.push((word, tag));
                    }
                }
                if !sent.is_empty() {
                    result.push(sent);
                }
            }
        }
        Ok(result)
    }

    /// Return just the words (no tags).
    #[pyo3(signature = (fileids=None))]
    fn words(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<String>> {
        Ok(self.tagged_words(fileids)?.into_iter().map(|(w, _)| w).collect())
    }
}

// ── CategorizedPlaintextCorpusReader ──
//
// Reads plaintext files organized in category subdirectories.

#[pyclass(name = "CategorizedPlaintextCorpusReader", module = "fastnltk._rust")]
pub struct CategorizedPlaintextCorpusReader {
    root: PathBuf,
    /// Map: category → list of fileids in that category.
    cat_map: Vec<(String, Vec<String>)>,
    /// Flattened fileids.
    all_fileids: Vec<String>,
}

#[pymethods]
impl CategorizedPlaintextCorpusReader {
    #[new]
    #[pyo3(signature = (root, fileids_map))]
    fn new(root: &str, fileids_map: Vec<(String, Vec<String>)>) -> PyResult<Self> {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            return Err(PyFileNotFoundError::new_err(format!(
                "Corpus directory not found: {root}"
            )));
        }
        let mut all_fileids = Vec::new();
        for (_, fids) in &fileids_map {
            all_fileids.extend(fids.iter().cloned());
        }
        all_fileids.sort();
        Ok(Self { root: root_path.to_path_buf(), cat_map: fileids_map, all_fileids })
    }

    fn fileids(&self) -> Vec<String> {
        self.all_fileids.clone()
    }

    fn categories(&self) -> Vec<String> {
        self.cat_map.iter().map(|(c, _)| c.clone()).collect()
    }

    fn fileids_by_category(&self, category: &str) -> Vec<String> {
        self.cat_map
            .iter()
            .find(|(c, _)| c == category)
            .map(|(_, fids)| fids.clone())
            .unwrap_or_default()
    }

    #[pyo3(signature = (fileids=None))]
    fn raw(&self, fileids: Option<Vec<String>>) -> PyResult<Vec<String>> {
        let ids = fileids.unwrap_or_else(|| self.all_fileids.clone());
        let mut contents = Vec::with_capacity(ids.len());
        for fid in &ids {
            let path = self.root.join(fid);
            let text = fs::read_to_string(&path)
                .map_err(|e| PyFileNotFoundError::new_err(format!("Cannot read {fid}: {e}")))?;
            contents.push(text);
        }
        Ok(contents)
    }

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
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlaintextCorpusReader>()?;
    m.add_class::<TaggedCorpusReader>()?;
    m.add_class::<CategorizedPlaintextCorpusReader>()?;
    Ok(())
}
