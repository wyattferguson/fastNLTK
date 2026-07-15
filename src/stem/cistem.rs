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
        if s.starts_with("ge") && s.len() > 4 {
            s = s[2..].to_string();
        }
        for suf in ["n", "s", "e", "t", "st", "et", "est", "en", "em", "er", "es"] {
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
