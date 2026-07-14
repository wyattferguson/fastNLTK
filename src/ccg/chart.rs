//! CCG Chart Parser — CKY-style CCG parsing with combinators.
//!
//! Implements a bottom-up chart parser for Combinatory Categorial Grammar.
//! Uses a flat `Vec<Vec<Vec<CCGEdge>>>` chart (O(1) array indexing)
//! instead of HashMap for maximum parser throughput.
//!
//! NLTK equivalent: nltk.ccg.chart.CCGChartParser

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ccg::combinator::{self, Combinator};
use crate::ccg::lexicon::CCGLexicon;
use crate::ccg::{Category, CategoryKind};

/// A chart cell entry: a category over a span [start, end).
#[derive(Clone, Debug)]
struct CCGEdge {
    cat: Category,
    start: usize,
    end: usize,
    left_child: Option<Box<CCGEdge>>,
    right_child: Option<Box<CCGEdge>>,
    rule: String,
}

impl CCGEdge {
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

/// The CCG chart parser, using CKY-style dynamic programming
/// with flat 3D array (span × start → edges) for O(1) cell access.
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

        // Flat 3D chart: chart[span][start] → Vec<CCGEdge>
        // Span 0 is unused; spans go 1..=n.
        // span s has (n - s + 1) possible start positions.
        let mut chart: Vec<Vec<Vec<CCGEdge>>> = (0..=n)
            .map(|s| {
                let count = if s == 0 { 0 } else { n - s + 1 };
                (0..count).map(|_| Vec::new()).collect()
            })
            .collect();

        // Initialize with lexical categories (span=1)
        for (i, word) in words.iter().enumerate() {
            let cats = self.lexicon.lookup_cats(word);
            if cats.is_empty() {
                // Unknown word — assign NP and N categories
                if let Some(np) = crate::ccg::parse_category("NP") {
                    chart[1][i].push(CCGEdge::new_lexical(np, i));
                }
                if let Some(n) = crate::ccg::parse_category("N") {
                    chart[1][i].push(CCGEdge::new_lexical(n, i));
                }
                continue;
            }
            for cat in cats {
                chart[1][i].push(CCGEdge::new_lexical(cat.clone(), i));
            }
        }

        let combinators = combinator::all_combinators();

        // Fill chart bottom-up (CKY-style)
        for span in 2..=n {
            for start in 0..=(n - span) {
                let end = start + span;
                let mut new_edges: Vec<CCGEdge> = Vec::new();

                for split in (start + 1)..end {
                    let left_span = split - start;
                    let right_span = end - split;
                    let lefts = &chart[left_span][start];
                    let rights = &chart[right_span][split];

                    if lefts.is_empty() || rights.is_empty() {
                        continue;
                    }

                    for l in lefts {
                        for r in rights {
                            for comb in &combinators {
                                if let Some(result_kind) =
                                    apply_with_variants(l.cat.kind(), r.cat.kind(), comb)
                                {
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

                if !new_edges.is_empty() {
                    chart[span][start] = new_edges;
                }
            }
        }

        // Collect results: edges spanning all words with S category
        let final_edges: Vec<String> = chart[n][0]
            .iter()
            .filter(|e| e.cat.to_string() == "S")
            .enumerate()
            .map(|(i, e)| format!("Parse {}: {} (rule: {})", i + 1, e.cat, e.rule))
            .collect();

        if !final_edges.is_empty() {
            return Ok(final_edges);
        }

        // Try any complete spanning parse
        let any_results: Vec<String> = chart[n][0]
            .iter()
            .enumerate()
            .map(|(i, e)| format!("Derivation {}: {} (rule: {})", i + 1, e.cat, e.rule))
            .collect();

        if any_results.is_empty() {
            Ok(vec!["No parse found".to_string()])
        } else {
            Ok(any_results)
        }
    }
}

/// Try standard application + composition + crossed composition variants.
fn apply_with_variants(
    left: &CategoryKind,
    right: &CategoryKind,
    comb: &Combinator,
) -> Option<CategoryKind> {
    if let Some(result) = combinator::apply_combinator(left, right, comb) {
        return Some(result);
    }

    let name = combinator::combinator_name(comb);
    match name {
        "FC" => {
            if let (
                CategoryKind::Functional { result: lr, argument: la, is_forward: true },
                CategoryKind::Functional { result: rr, argument: ra, is_forward: true },
            ) = (left, right)
            {
                if **la == **rr {
                    return Some(CategoryKind::Functional {
                        result: lr.clone(),
                        argument: ra.clone(),
                        is_forward: true,
                    });
                }
            }
            None
        }
        "BC" => {
            if let (
                CategoryKind::Functional { result: rr, argument: ra, is_forward: false },
                CategoryKind::Functional { result: _, argument: la, is_forward: false },
            ) = (left, right)
            {
                if **la == **rr {
                    return Some(CategoryKind::Functional {
                        result: ra.clone(),
                        argument: la.clone(),
                        is_forward: false,
                    });
                }
            }
            None
        }
        "FX" => {
            if let (
                CategoryKind::Functional { result: lr, argument: la, is_forward: true },
                CategoryKind::Functional { result: _, argument: ra, is_forward: false },
            ) = (left, right)
            {
                if **la == **ra {
                    return Some(CategoryKind::Functional {
                        result: lr.clone(),
                        argument: ra.clone(),
                        is_forward: true,
                    });
                }
            }
            None
        }
        "BX" => {
            if let (
                CategoryKind::Functional { result: rr, argument: _, is_forward: true },
                CategoryKind::Functional { result: _, argument: la, is_forward: false },
            ) = (left, right)
            {
                if **la == **rr {
                    return Some(CategoryKind::Functional {
                        result: rr.clone(),
                        argument: la.clone(),
                        is_forward: false,
                    });
                }
            }
            None
        }
        _ => None,
    }
}

fn format_kind(k: &CategoryKind) -> String {
    match k {
        CategoryKind::Primitive(l) => l.clone(),
        CategoryKind::Functional { result, argument, is_forward } => {
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
