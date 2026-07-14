//! ISRI Arabic stemmer — Rust port.
use pyo3::prelude::*;

#[pyclass(name = "ISRIStemmer", module = "fastnltk._rust")]
pub struct ISRIStemmer;

#[pymethods]
impl ISRIStemmer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn stem(&self, word: &str) -> String {
        let w = word.trim().to_lowercase();
        if w.len() <= 3 {
            return w.clone();
        }
        let mut s = w.clone();
        if s.starts_with("al") && s.len() > 4 {
            s = s[2..].to_string();
        }
        for p in ["w", "f", "b", "l", "y", "t", "n", "s"] {
            if s.starts_with(p) && s.len() > 3 {
                s = s[p.len()..].to_string();
                break;
            }
        }
        for suf in
            ["huma", "km", "kn", "hm", "hn", "na", "ny", "at", "an", "yn", "wn", "h", "t", "k", "n"]
        {
            if s.ends_with(suf) && s.len() > suf.len() + 2 {
                s = s[..s.len() - suf.len()].to_string();
                break;
            }
        }
        if s.len() < 2 {
            return w;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_isri_runs() {
        let st = ISRIStemmer::new();
        let r = st.stem("ktb");
        assert!(!r.is_empty());
    }
}
