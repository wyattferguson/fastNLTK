//! S-Expression tokenizer — Rust implementation matching NLTK's `SExprTokenizer`.
//!
//! Divides a string into parenthesized expressions (including nested ones)
//! and other whitespace-separated tokens. Customizable open/close parens
//! and strict/non-strict mode for mismatched parentheses.
//!
//! NLTK equivalent: nltk.tokenize.sexpr.SExprTokenizer

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "SExprTokenizer", module = "fastnltk._rust")]
pub struct SExprTokenizer {
    open_paren: String,
    close_paren: String,
    strict: bool,
}

#[pymethods]
impl SExprTokenizer {
    #[new]
    #[pyo3(signature = (parens="()", strict=true))]
    fn new(parens: &str, strict: bool) -> PyResult<Self> {
        if parens.len() != 2 {
            return Err(PyValueError::new_err("parens must contain exactly two characters"));
        }
        let chars: Vec<char> = parens.chars().collect();
        Ok(Self { open_paren: chars[0].to_string(), close_paren: chars[1].to_string(), strict })
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        let mut result: Vec<String> = Vec::new();
        let mut pos: usize = 0;
        let mut depth: i32 = 0;
        let bytes = text.as_bytes();
        let open_byte = self.open_paren.as_bytes()[0];
        let close_byte = self.close_paren.as_bytes()[0];

        for (i, &b) in bytes.iter().enumerate() {
            if b == open_byte || b == close_byte {
                if depth == 0 {
                    // Flush any non-paren tokens before this paren
                    let before = &text[pos..i];
                    for token in before.split_whitespace() {
                        if !token.is_empty() {
                            result.push(token.to_string());
                        }
                    }
                    pos = i;
                }
                if b == open_byte {
                    depth += 1;
                }
                if b == close_byte {
                    if self.strict && depth == 0 {
                        return Err(PyValueError::new_err(format!(
                            "Un-matched close paren at char {i}"
                        )));
                    }
                    depth = (depth - 1).max(0);
                    if depth == 0 {
                        result.push(text[pos..=i].to_string());
                        pos = i + 1;
                    }
                }
            }
        }

        if self.strict && depth > 0 {
            return Err(PyValueError::new_err(format!("Un-matched open paren at char {pos}")));
        }

        if pos < text.len() {
            let remaining = &text[pos..];
            for token in remaining.split_whitespace() {
                if !token.is_empty() {
                    result.push(token.to_string());
                }
            }
        }

        Ok(result)
    }
}
