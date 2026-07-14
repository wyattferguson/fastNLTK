//! CCG — Combinatory Categorial Grammar types + API.
//!
//! NLTK equivalent: nltk.ccg.api

pub mod combinator;
pub mod lexicon;
pub mod chart;

use std::fmt;
use pyo3::prelude::*;

/// A CCG category: primitive (N, NP, S, PP) or functional (A/B, A\B).
#[pyclass(name = "Category", module = "fastnltk._rust")]
#[derive(Clone, Debug, PartialEq)]
pub struct Category {
    inner: CategoryKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CategoryKind {
    Primitive(String),
    Functional {
        result: Box<CategoryKind>,
        argument: Box<CategoryKind>,
        is_forward: bool, // true = A/B, false = A\B
    },
}

#[pymethods]
impl Category {
    fn is_primitive(&self) -> bool {
        matches!(self.inner, CategoryKind::Primitive(_))
    }

    fn is_functional(&self) -> bool {
        matches!(self.inner, CategoryKind::Functional { .. })
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl Category {
    pub(crate) fn new(kind: CategoryKind) -> Self {
        Category { inner: kind }
    }

    pub(crate) fn kind(&self) -> &CategoryKind {
        &self.inner
    }

    pub fn primitive(label: &str) -> Self {
        Category {
            inner: CategoryKind::Primitive(label.to_string()),
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            CategoryKind::Primitive(label) => write!(f, "{label}"),
            CategoryKind::Functional { result, argument, is_forward } => {
                let result_s = fmt_kind(result);
                let arg_s = fmt_kind(argument);
                if *is_forward {
                    write!(f, "{result_s}/{arg_s}")
                } else {
                    write!(f, "{result_s}\\{arg_s}")
                }
            }
        }
    }
}

fn fmt_kind(k: &CategoryKind) -> String {
    match k {
        CategoryKind::Primitive(l) => l.clone(),
        CategoryKind::Functional { result, argument, is_forward } => {
            let rs = fmt_kind(result);
            let as_ = fmt_kind(argument);
            if *is_forward {
                format!("({rs}/{as_})")
            } else {
                format!("({rs}\\{as_})")
            }
        }
    }
}

/// Parse a CCG category string like "NP/N" or "S\NP" or "(S\NP)/NP".
pub fn parse_category(s: &str) -> Option<Category> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    parse_inner(s).map(|(kind, _)| Category::new(kind))
}

fn parse_inner(s: &str) -> Option<(CategoryKind, usize)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Check for parentheses
    if s.starts_with('(') {
        let mut depth = 1;
        let mut i = 1;
        while i < s.len() && depth > 0 {
            if s.as_bytes()[i] == b'(' { depth += 1; }
            else if s.as_bytes()[i] == b')' { depth -= 1; }
            i += 1;
        }
        if depth == 0 {
            // Parse inner expression
            let inner = &s[1..i - 1];
            if let Some((kind, _used)) = parse_inner(inner) {
                // Check if there's a / or \ after
                let rest = &s[i..].trim();
                if rest.is_empty() {
                    return Some((kind, i));
                }
                if let Some(dir) = rest.chars().next() {
                    if dir == '/' || dir == '\\' {
                        let after = &rest[1..].trim();
                        if let Some((arg_kind, _)) = parse_inner(after) {
                            let total = i + 1 + (rest.len() - after.len());
                            return Some((CategoryKind::Functional {
                                result: Box::new(kind),
                                argument: Box::new(arg_kind),
                                is_forward: dir == '/',
                            }, total));
                        }
                    }
                }
                return Some((kind, i));
            }
        }
        return None;
    }

    // Find / or \
    let mut slash_pos = None;
    for (i, c) in s.char_indices() {
        if c == '/' || c == '\\' {
            slash_pos = Some((i, c));
            break;
        }
    }

    match slash_pos {
        Some((pos, dir)) => {
            let left = &s[..pos];
            let right = &s[pos + 1..];
            if let Some(result) = parse_inner(left) {
                if let Some((arg_kind, _)) = parse_inner(right) {
                    return Some((CategoryKind::Functional {
                        result: Box::new(result.0),
                        argument: Box::new(arg_kind),
                        is_forward: dir == '/',
                    }, s.len()));
                }
            }
            None
        }
        None => {
            // Primitive category
            let label: String = s.chars().take_while(|c| c.is_alphabetic() || c.is_ascii_punctuation()).collect();
            if label.is_empty() { None }
            else { Some((CategoryKind::Primitive(label.clone()), label.len())) }
        }
    }
}

#[pyfunction]
fn from_string(s: &str) -> PyResult<Category> {
    parse_category(s).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Invalid category: {s}")))
}

/// Register the CCG module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Category>()?;
    m.add_function(wrap_pyfunction!(from_string, m)?)?;
    chart::register_module(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_np() {
        let cat = parse_category("NP").unwrap();
        assert!(cat.is_primitive());
        assert_eq!(cat.to_string(), "NP");
    }

    #[test]
    fn test_forward_functional() {
        let cat = parse_category("NP/N").unwrap();
        assert!(cat.is_functional());
        assert_eq!(cat.to_string(), "NP/N");
    }

    #[test]
    fn test_backward_functional() {
        let cat = parse_category("S\\NP").unwrap();
        assert_eq!(cat.to_string(), "S\\NP");
    }

    #[test]
    fn test_nested() {
        let cat = parse_category("(S\\NP)/NP").unwrap();
        assert!(cat.is_functional());
    }

    #[test]
    fn test_invalid() {
        assert!(parse_category("").is_none());
    }
}
