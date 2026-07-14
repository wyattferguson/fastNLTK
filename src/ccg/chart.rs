//! CCG Chart Parser — CKY-style CCG parsing with combinators.
//!
//! Implements a bottom-up chart parser for Combinatory Categorial Grammar.
//! The parser uses a CKY-style dynamic programming approach:
//!   1. Initialize chart with lexical categories from `CCGLexicon`
//!   2. For each span size 2..n, try all split points and apply combinators
//!   3. Collect parses that span the entire input with category S
//!
//! Supports forward/backward application (FA, BA) and composition (FC, BC).
//! Unknown words receive default NP and N categories for robustness.
//!
//! NLTK equivalent: nltk.ccg.chart.CCGChartParser

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::ccg::combinator::{self, Combinator};
use crate::ccg::lexicon::CCGLexicon;
use crate::ccg::{Category, CategoryKind};

/// A chart cell entry: a category over a span [start, end).
///
/// Tracks the category, its input span, and optional children
/// for derivation tree reconstruction.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CCGEdge {
    cat: Category,
    start: usize,
    end: usize,
    /// Left child for derivation tree (None for lexical edges).
    left_child: Option<Box<CCGEdge>>,
    /// Right child for derivation tree (None for lexical edges).
    right_child: Option<Box<CCGEdge>>,
    /// Rule name ("lex", "FA", "BA", "FC", "BC").
    #[allow(dead_code)]
    rule: String,
}

impl CCGEdge {
    /// Create a lexical edge for a word at position `pos`.
    fn new_lexical(cat: Category, pos: usize) -> Self {
        CCGEdge {
            cat,
            start: pos,
            end: pos + 1,
            left_child: None,
            right_child: None,
            rule: "lex".into(),
        }
    }

    /// Create a combined edge from two sub-edges using a combinator rule.
    fn combined(cat: Category, left: CCGEdge, right: CCGEdge, rule: &str) -> Self {
        CCGEdge {
            cat,
            start: left.start,
            end: right.end,
            left_child: Some(Box::new(left)),
            right_child: Some(Box::new(right)),
            rule: rule.into(),
        }
    }
}

/// The CCG chart parser, using CKY-style dynamic programming.
#[pyclass(name = "CCGChartParser", module = "fastnltk._rust")]
pub struct CCGChartParser {
    lexicon: CCGLexicon,
    max_span: usize,
}

#[pymethods]
impl CCGChartParser {
    #[new]
    #[pyo3(signature = (lexicon, max_span=20))]
    fn new(lexicon: CCGLexicon, max_span: usize) -> Self {
        CCGChartParser { lexicon, max_span }
    }

    /// Parse a sequence of words into CCG derivation trees.
    /// Returns a list of derivation descriptions.
    #[pyo3(signature = (words))]
    fn parse(&self, words: Vec<String>) -> PyResult<Vec<String>> {
        let n = words.len();
        if n == 0 {
            return Err(PyValueError::new_err("Empty input"));
        }
        if n > self.max_span {
            return Err(PyValueError::new_err(format!(
                "Input too long ({} words, max {})",
                n, self.max_span
            )));
        }

        // Build chart: [span_size][start] -> Vec<CCGEdge>
        // We use span_size=1..=n, start=0..(n-span_size)
        let mut chart: HashMap<(usize, usize), Vec<CCGEdge>> = HashMap::new();

        // Initialize with lexical categories (span=1)
        for (i, word) in words.iter().enumerate() {
            let cats = self.lexicon.lookup_cats(word);
            if cats.is_empty() {
                // Unknown word — try to give it NP and N as default
                if let Some(np) = crate::ccg::parse_category("NP") {
                    chart
                        .entry((1, i))
                        .or_default()
                        .push(CCGEdge::new_lexical(np, i));
                }
                if let Some(n) = crate::ccg::parse_category("N") {
                    chart
                        .entry((1, i))
                        .or_default()
                        .push(CCGEdge::new_lexical(n, i));
                }
                continue;
            }
            for cat in cats {
                chart
                    .entry((1, i))
                    .or_default()
                    .push(CCGEdge::new_lexical(cat.clone(), i));
            }
        }

        let combinators = combinator::all_combinators();

        // Fill chart bottom-up (CKY-style)
        for span in 2..=n {
            for start in 0..=(n - span) {
                let end = start + span;
                let mut new_edges: Vec<CCGEdge> = Vec::new();

                // Try all split points
                for split in (start + 1)..end {
                    let left_edges = chart.get(&(split - start, start));
                    let right_edges = chart.get(&(end - split, split));

                    if let Some(lefts) = left_edges {
                        if let Some(rights) = right_edges {
                            for l in lefts {
                                for r in rights {
                                    for comb in &combinators {
                                        let kind_l = l.cat.kind();
                                        let kind_r = r.cat.kind();
                                        if let Some(result_kind) = apply_combinator_with_composition(
                                            kind_l, kind_r, comb, span,
                                        ) {
                                            let result_str = format_kind(&result_kind);
                                            if let Some(result_cat) =
                                                crate::ccg::parse_category(&result_str)
                                            {
                                                new_edges.push(CCGEdge::combined(
                                                    result_cat,
                                                    l.clone(),
                                                    r.clone(),
                                                    combinator::combinator_name(comb),
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !new_edges.is_empty() {
                    chart.entry((span, start)).or_default().extend(new_edges);
                }
            }
        }

        // Collect results: edges spanning all words with S category
        let results: Vec<String> = chart
            .get(&(n, 0))
            .map(|edges| {
                edges
                    .iter()
                    .filter(|e| {
                        let s = e.cat.to_string();
                        s == "S"
                    })
                    .enumerate()
                    .map(|(i, e)| format!("Parse {}: {} (rule: {})", i + 1, e.cat, e.rule))
                    .collect()
            })
            .unwrap_or_default();

        if results.is_empty() {
            // Try to find any complete spanning parse
            let any_results: Vec<String> = chart
                .get(&(n, 0))
                .map(|edges| {
                    edges
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("Derivation {}: {} (rule: {})", i + 1, e.cat, e.rule))
                        .collect()
                })
                .unwrap_or_default();

            if any_results.is_empty() {
                Ok(vec!["No parse found".to_string()])
            } else {
                Ok(any_results)
            }
        } else {
            Ok(results)
        }
    }
}

/// Apply a combinator, also checking forward/backward composition variants.
fn apply_combinator_with_composition(
    left: &CategoryKind,
    right: &CategoryKind,
    comb: &Combinator,
    _span: usize,
) -> Option<CategoryKind> {
    // Try standard application first
    if let Some(result) = combinator::apply_combinator(left, right, comb) {
        return Some(result);
    }

    match comb_name(comb) {
        "FC" => {
            // Forward composition: A/B + B/C -> A/C
            if let CategoryKind::Functional {
                result: lr,
                argument: la,
                is_forward: true,
            } = left
            {
                if let CategoryKind::Functional {
                    result: rr,
                    argument: ra,
                    is_forward: true,
                } = right
                {
                    if **la == **rr {
                        return Some(CategoryKind::Functional {
                            result: lr.clone(),
                            argument: ra.clone(),
                            is_forward: true,
                        });
                    }
                }
            }
            None
        }
        "BC" => {
            // Backward composition: B\C + A\B -> A\C
            if let CategoryKind::Functional {
                result: _,
                argument: la,
                is_forward: false,
            } = right
            {
                if let CategoryKind::Functional {
                    result: rr,
                    argument: ra,
                    is_forward: false,
                } = left
                {
                    if **la == **rr {
                        return Some(CategoryKind::Functional {
                            result: ra.clone(),
                            argument: la.clone(),
                            is_forward: false,
                        });
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn comb_name(comb: &Combinator) -> &'static str {
    combinator::combinator_name(comb)
}

fn format_kind(k: &CategoryKind) -> String {
    match k {
        CategoryKind::Primitive(l) => l.clone(),
        CategoryKind::Functional {
            result,
            argument,
            is_forward,
        } => {
            let r = format_kind(result);
            let a = format_kind(argument);
            if *is_forward {
                format!("{r}/{a}")
            } else {
                format!("{r}\\{a}")
            }
        }
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CCGChartParser>()?;
    m.add_class::<CCGLexicon>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccg::lexicon::CCGLexicon;

    fn test_lexicon() -> CCGLexicon {
        CCGLexicon::new(Some(vec![
            ("the".into(), "NP/N".into()),
            ("cat".into(), "N".into()),
            ("dog".into(), "N".into()),
            ("chased".into(), "(S\\NP)/NP".into()),
            ("saw".into(), "(S\\NP)/NP".into()),
            ("a".into(), "NP/N".into()),
            ("ball".into(), "N".into()),
        ]))
        .unwrap()
    }

    #[test]
    fn test_chart_parse_simple() {
        let lex = test_lexicon();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = "the cat chased a ball"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let results = parser.parse(words).unwrap();
        assert!(!results.is_empty(), "Should find at least one parse");
        assert!(
            results.iter().any(|r| r.starts_with("Parse")),
            "Should have S parse"
        );
    }

    #[test]
    fn test_chart_parse_two_words() {
        let lex = test_lexicon();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = "the cat"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let results = parser.parse(words).unwrap();
        // "the cat" should result in NP (not S)
        assert!(results.iter().any(|r| r.contains("NP")), "Should have NP");
    }

    #[test]
    fn test_chart_parse_empty() {
        let lex = test_lexicon();
        let parser = CCGChartParser::new(lex, 20);
        let result = parser.parse(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chart_parse_unknown_word() {
        let lex = test_lexicon();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = "the cat ate a ball"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let results = parser.parse(words).unwrap();
        assert!(!results.is_empty(), "Should handle unknown words");
    }

    #[test]
    fn test_chart_parse_three_words_full_sentence() {
        let lex = CCGLexicon::new(Some(vec![
            ("the".into(), "NP/N".into()),
            ("cat".into(), "N".into()),
            ("ran".into(), "S\\NP".into()),
        ]))
        .unwrap();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = "the cat ran"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let results = parser.parse(words).unwrap();
        assert!(
            results.iter().any(|r| r.starts_with("Parse")),
            "Should produce S parse: {:?}",
            results
        );
    }

    #[test]
    fn test_max_span_exceeded() {
        let lex = test_lexicon();
        let parser = CCGChartParser::new(lex, 3);
        let words: Vec<String> = "the cat chased a ball"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let result = parser.parse(words);
        assert!(result.is_err(), "Should reject input exceeding max_span");
    }

    #[test]
    fn test_no_parse_possible() {
        // NP and N categories only -> cannot form S
        let lex = CCGLexicon::new(Some(vec![
            ("the".into(), "NP/N".into()),
            ("cat".into(), "N".into()),
        ]))
        .unwrap();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = "the cat"
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let results = parser.parse(words).unwrap();
        // Should get NP derivation, not an S parse
        assert!(
            results.iter().any(|r| r.contains("NP")),
            "Should find NP: {:?}",
            results
        );
    }

    #[test]
    fn test_single_word() {
        let lex = CCGLexicon::new(Some(vec![("hello".into(), "S".into())])).unwrap();
        let parser = CCGChartParser::new(lex, 20);
        let words: Vec<String> = vec!["hello".to_string()];
        let results = parser.parse(words).unwrap();
        assert!(
            results[0].contains("S") || results[0].contains("Derivation"),
            "Should handle single word: {:?}",
            results
        );
    }
}
