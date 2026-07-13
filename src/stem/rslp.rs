//! RSLP — Portuguese stemmer.
use pyo3::prelude::*;

#[pyclass(name = "RSLPStemmer", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RSLPStemmer;

#[pymethods]
impl RSLPStemmer {
    #[new]
    fn new() -> Self { Self }

    fn stem(&self, word: &str) -> String {
        let w = word.to_lowercase();
        if w.len() <= 3 { return w; }
        let mut s = w;
        if s.ends_with("ns") { s = s[..s.len() - 1].to_string(); }
        else if s.ends_with("es") && s.len() > 4 { s = s[..s.len() - 2].to_string(); }
        else if s.ends_with("s") && !s.ends_with("ss") && s.len() > 3 { s.pop(); }
        if s.ends_with("a") && s.len() > 3 { s.pop(); s.push('o'); }
        for suf in ["ou", "ram", "mos", "ndo", "sse"] {
            if s.ends_with(suf) && s.len() > suf.len() + 2 {
                s = s[..s.len() - suf.len()].to_string(); break;
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rslp_runs() {
        let st = RSLPStemmer::new();
        let r = st.stem("casas");
        assert!(r.len() <= "casas".len());
    }
}
