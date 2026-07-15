//! CCG — Combinatory Categorial Grammar types + API.
//!
//! NLTK equivalent: nltk.ccg.api

pub mod chart;
pub mod combinator;
pub mod lexicon;

use pyo3::prelude::*;
use std::fmt;

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
        result: Box<Self>,
        argument: Box<Self>,
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
        Self { inner: kind }
    }

    pub(crate) fn kind(&self) -> &CategoryKind {
        &self.inner
    }

    pub fn primitive(label: &str) -> Self {
        Self { inner: CategoryKind::Primitive(label.to_string()) }
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

fn find_matching_paren(bytes: &[u8]) -> Option<usize> {
    let mut depth = 1u32;
    for (i, &b) in bytes.iter().enumerate().skip(1) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_inner(s: &str) -> Option<(CategoryKind, usize)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    // Parenthesized sub-expression
    if bytes[0] == b'(' {
        let close = find_matching_paren(bytes)?;
        let kind = parse_inner(&s[1..close])?.0;
        let after = &s[close + 1..];
        if after.is_empty() {
            return Some((kind, close + 1));
        }
        let rest = after.trim_start();
        if rest.is_empty() {
            return Some((kind, close + 1));
        }
        let rbytes = rest.as_bytes();
        if rbytes[0] == b'/' || rbytes[0] == b'\\' {
            let tail = rest[1..].trim_start();
            let arg_kind = parse_inner(tail)?.0;
            return Some((
                CategoryKind::Functional {
                    result: Box::new(kind),
                    argument: Box::new(arg_kind),
                    is_forward: rbytes[0] == b'/',
                },
                close + 1 + 1 + (rest.len() - tail.len()),
            ));
        }
        return Some((kind, close + 1));
    }

    // Scan for first / or \ (byte-level — CCG input is ASCII)
    match bytes.iter().position(|&b| b == b'/' || b == b'\\') {
        Some(pos) => {
            let left = &s[..pos];
            let right = s[pos + 1..].trim_start();
            let result = parse_inner(left)?;
            let arg_kind = parse_inner(right)?.0;
            Some((
                CategoryKind::Functional {
                    result: Box::new(result.0),
                    argument: Box::new(arg_kind),
                    is_forward: bytes[pos] == b'/',
                },
                s.len(),
            ))
        }
        None => {
            // Primitive category — ASCII alphabetic label
            let label_len =
                bytes.iter().position(|&b| !b.is_ascii_alphabetic()).unwrap_or(bytes.len());
            if label_len == 0 {
                return None;
            }
            let label = s[..label_len].to_string();
            Some((CategoryKind::Primitive(label), label_len))
        }
    }
}

#[pyfunction]
fn from_string(s: &str) -> PyResult<Category> {
    parse_category(s)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Invalid category: {s}")))
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
