//! Cistem — German stemmer.
use pyo3::prelude::*;

#[pyclass(name = "Cistem", module = "fastnltk._rust")]
pub struct Cistem;

#[pymethods]
impl Cistem {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn stem(&self, word: &str) -> String {
        let w = word.to_lowercase();
        if w.len() <= 3 {
            return w;
        }
        let mut s = w;
        // CISTEM: ^ge(.{4,}) — remove ge- only if remaining stem >= 4 chars
        if s.starts_with("ge") && s.len() > 5 {
            s = s[2..].to_string();
        }
        // Check suffixes longest-first to avoid partial suffix matches
        for suf in ["est", "et", "en", "em", "er", "es", "st", "t", "e", "n", "s"] {
            if s.ends_with(suf) && s.len() > suf.len() + 2 {
                s = s[..s.len() - suf.len()].to_string();
                break;
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cistem_runs() {
        let st = Cistem::new();
        let r = st.stem("laufen");
        assert!(r.len() <= "laufen".len());
    }

    #[test]
    fn test_cistem_empty() {
        let st = Cistem::new();
        assert_eq!(st.stem(""), "");
    }

    #[test]
    fn test_cistem_short() {
        let st = Cistem::new();
        assert_eq!(st.stem("ab"), "ab");
        assert_eq!(st.stem("der"), "der");
    }

    #[test]
    fn test_cistem_ge_prefix() {
        let st = Cistem::new();
        let result = st.stem("gelaufen");
        assert!(!result.starts_with("ge"));
    }
}
