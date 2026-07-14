//! Multi-Word Expression tokenizer — Rust implementation.
//!
//! NLTK equivalent: nltk.tokenize.mwe.MWETokenizer

use hashbrown::HashMap;
use pyo3::prelude::*;

#[derive(Clone, Default)]
struct TrieNode {
    children: HashMap<String, TrieNode>,
    is_end: bool,
}

#[pyclass(name = "MWETokenizer", module = "fastnltk._rust")]
pub struct MWETokenizer {
    root: TrieNode,
    separator: String,
}

#[pymethods]
impl MWETokenizer {
    #[new]
    #[pyo3(signature = (mwes=None, separator="_"))]
    fn new(mwes: Option<Vec<Vec<String>>>, separator: &str) -> Self {
        let mut tok = MWETokenizer { root: TrieNode::default(), separator: separator.to_string() };
        if let Some(expressions) = mwes {
            for mwe in expressions {
                tok.add_mwe(mwe);
            }
        }
        tok
    }

    fn add_mwe(&mut self, mwe: Vec<String>) {
        if mwe.is_empty() {
            return;
        }
        let mut node = &mut self.root;
        for word in &mwe {
            node = node.children.entry(word.clone()).or_default();
        }
        node.is_end = true;
    }

    fn tokenize(&self, text: Vec<String>) -> Vec<String> {
        let mut result = Vec::with_capacity(text.len());
        let mut i = 0;
        while i < text.len() {
            let mut matched_end = None;
            let mut node = &self.root;
            for j in i..text.len() {
                match node.children.get(&text[j]) {
                    Some(next) => {
                        node = next;
                        if node.is_end {
                            matched_end = Some(j);
                        }
                    }
                    None => break,
                }
            }
            match matched_end {
                Some(end) => {
                    result.push(text[i..=end].join(&self.separator));
                    i = end + 1;
                }
                None => {
                    result.push(text[i].clone());
                    i += 1;
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> MWETokenizer {
        MWETokenizer::new(
            Some(vec![
                vec!["a".into(), "little".into()],
                vec!["a".into(), "little".into(), "bit".into()],
                vec!["a".into(), "lot".into()],
                vec!["in".into(), "spite".into(), "of".into()],
            ]),
            "_",
        )
    }

    #[test]
    fn test_no_match() {
        let t = tok();
        let text: Vec<String> =
            "This is a test in spite".split_whitespace().map(String::from).collect();
        assert_eq!(t.tokenize(text), vec!["This", "is", "a", "test", "in", "spite"]);
    }

    #[test]
    fn test_merged() {
        let t = tok();
        let text: Vec<String> = "In a little or a little bit or a lot in spite of"
            .split_whitespace()
            .map(String::from)
            .collect();
        assert_eq!(
            t.tokenize(text),
            vec!["In", "a_little", "or", "a_little_bit", "or", "a_lot", "in_spite_of"]
        );
    }

    #[test]
    fn test_empty() {
        let t = MWETokenizer::new(None, "_");
        assert_eq!(t.tokenize(vec!["a".into(), "b".into()]), vec!["a", "b"]);
    }

    #[test]
    fn test_add() {
        let mut t = MWETokenizer::new(None, "_");
        t.add_mwe(vec!["a".into(), "b".into()]);
        assert_eq!(t.tokenize(vec!["a".into(), "b".into(), "c".into()]), vec!["a_b", "c"]);
    }

    #[test]
    fn test_longest_match() {
        let mut t = MWETokenizer::new(None, "_");
        t.add_mwe(vec!["a".into(), "b".into()]);
        t.add_mwe(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(t.tokenize(vec!["a".into(), "b".into(), "c".into()]), vec!["a_b_c"]);
    }
}
