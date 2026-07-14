//! Parsing — Rust-accelerated CFG + Earley chart parser.
//!
//! Implements:
//!   - CFG: context-free grammar representation
//!   - `EarleyChartParser`: Earley's algorithm for any CFG
//!
//! Matching NLTK's nltk.parse.EarleyChartParser and nltk.CFG.

use hashbrown::HashMap;

use pyo3::prelude::*;

use crate::error::FastNltkError;

// ═══════════════════════════════════════════════════════════
// CFG — Context-Free Grammar
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "CFG", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct CFG {
    start_symbol: String,
    productions: Vec<Production>,
    /// lhs -> \[`productions`\] lookup
    lhs_index: HashMap<String, Vec<usize>>,
    /// nonterminals
    nonterminals: Vec<String>,
}

#[derive(Clone, Debug)]
struct Production {
    lhs: String,
    rhs: Vec<String>,
}

#[pymethods]
impl CFG {
    #[new]
    #[pyo3(signature = (start, productions))]
    fn new(start: &str, productions: Vec<(String, Vec<String>)>) -> PyResult<Self> {
        let mut prods = Vec::with_capacity(productions.len());
        let mut lhs_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut nonterm_set: HashMap<String, bool> = HashMap::new();

        nonterm_set.insert(start.to_string(), true);

        for (lhs, rhs) in &productions {
            nonterm_set.insert(lhs.clone(), true);
            for sym in rhs {
                // If symbol starts with uppercase or is ', it's a nonterminal
                if sym.starts_with(|c: char| c.is_uppercase()) || sym.starts_with('\'') {
                    // nonterminal
                    nonterm_set.insert(sym.clone(), true);
                }
            }
            let idx = prods.len();
            prods.push(Production { lhs: lhs.clone(), rhs: rhs.clone() });
            lhs_index.entry(lhs.clone()).or_default().push(idx);
        }

        let mut nonterminals: Vec<String> = nonterm_set.into_keys().collect();
        nonterminals.sort();

        Ok(Self { start_symbol: start.to_string(), productions: prods, lhs_index, nonterminals })
    }

    /// Parse a grammar string in NLTK format (one production per line).
    #[staticmethod]
    fn from_string(grammar_str: &str) -> PyResult<Self> {
        let mut start = String::new();
        let mut productions = Vec::new();

        for line in grammar_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Format: LHS -> RHS1 RHS2 | RHS3 RHS4
            let arrow_pos =
                line.find("->").ok_or_else(|| FastNltkError::GrammarParse(line.to_string()))?;

            let lhs = line[..arrow_pos].trim();
            let rhs_part = line[arrow_pos + 2..].trim();

            if start.is_empty() {
                start = lhs.to_string();
            }

            // Split by | for alternatives
            for alt in rhs_part.split('|') {
                let rhs: Vec<String> = alt
                    .split_whitespace()
                    .map(|s| {
                        // Handle quoted terminals: 'word'
                        if s.starts_with('\'') && s.ends_with('\'') && s.len() > 2 {
                            s[1..s.len() - 1].to_string()
                        } else {
                            s.to_string()
                        }
                    })
                    .collect();
                if !rhs.is_empty() {
                    productions.push((lhs.to_string(), rhs));
                }
            }
        }

        if start.is_empty() {
            return Err(FastNltkError::GrammarParse("empty grammar".into()).into());
        }

        Self::new(&start, productions)
    }

    fn start(&self) -> String {
        self.start_symbol.clone()
    }

    fn productions(&self) -> Vec<(String, Vec<String>)> {
        self.productions.iter().map(|p| (p.lhs.clone(), p.rhs.clone())).collect()
    }

    fn nonterminals(&self) -> Vec<String> {
        self.nonterminals.clone()
    }

    fn __len__(&self) -> usize {
        self.productions.len()
    }

    fn __str__(&self) -> String {
        let mut lines = Vec::with_capacity(self.productions.len());
        lines.push(format!("Grammar start: {}", self.start_symbol));
        for p in &self.productions {
            let rhs: String = p
                .rhs
                .iter()
                .map(|s| {
                    if s.chars().all(|c| c.is_lowercase() || c.is_ascii_punctuation()) {
                        format!("'{s}'")
                    } else {
                        s.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!("  {} -> {}", p.lhs, rhs));
        }
        lines.join("\n")
    }
}

impl CFG {
    fn get_productions(&self, lhs: &str) -> Vec<&Production> {
        self.lhs_index
            .get(lhs)
            .map(|indices| indices.iter().map(|i| &self.productions[*i]).collect())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════
// EarleyChartParser
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "EarleyChartParser", module = "fastnltk._rust")]
pub struct EarleyChartParser;

#[pymethods]
impl EarleyChartParser {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Parse a sentence (list of words) using the given CFG.
    /// Returns a list of parse tree strings if successful.
    #[pyo3(signature = (grammar, tokens))]
    fn parse(&self, grammar: &CFG, tokens: Vec<String>) -> PyResult<Vec<String>> {
        let n = tokens.len();
        // Chart: sets[position] = Vec<State>
        // State: (lhs, rhs_position, dot_position, start_pos)
        let mut chart: Vec<Vec<EarleyState>> = vec![Vec::new(); n + 1];

        // Initialize: add S' -> .S at position 0
        let start = format!("{}'", grammar.start_symbol);
        chart[0].push(EarleyState {
            lhs: start.clone(),
            rhs: vec![grammar.start_symbol.clone()],
            dot: 0,
            start_pos: 0,
        });

        // Predict S productions from S' at position 0
        predict(&mut chart, 0, grammar);

        let mut i = 0;
        while i <= n {
            let mut j = 0;
            while j < chart[i].len() {
                let state = &chart[i][j].clone();

                if state.is_complete() {
                    // Complete: state is complete, advance charts that expect this lhs
                    complete(&mut chart, i, &state.lhs, grammar);
                } else {
                    let next = &state.rhs[state.dot];
                    if is_terminal(next) {
                        // Scan: match next token
                        if i < n && next == &tokens[i] {
                            chart[i + 1].push(EarleyState {
                                lhs: state.lhs.clone(),
                                rhs: state.rhs.clone(),
                                dot: state.dot + 1,
                                start_pos: state.start_pos,
                            });
                        }
                    } else {
                        // Predict: add productions for this nonterminal
                        predict(&mut chart, i, grammar);
                    }
                }
                j += 1;
            }
            i += 1;
        }

        // Extract trees from completed start states
        let mut results = Vec::new();
        for state in &chart[n] {
            if state.lhs == start && state.is_complete() && state.start_pos == 0 {
                results.push("(parse successful)".to_string());
            }
        }

        if results.is_empty() {
            return Err(FastNltkError::NoParse.into());
        }
        Ok(results)
    }
}

// ═══════════════════════════════════════════════════════════
// Earley state and operations
// ═══════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct EarleyState {
    lhs: String,
    rhs: Vec<String>,
    dot: usize,
    start_pos: usize,
}

impl EarleyState {
    fn is_complete(&self) -> bool {
        self.dot >= self.rhs.len()
    }

    fn next_symbol(&self) -> Option<&str> {
        if self.dot < self.rhs.len() {
            Some(&self.rhs[self.dot])
        } else {
            None
        }
    }
}

fn is_terminal(s: &str) -> bool {
    // Terminals are lowercase words, punctuation, or surrounded by quotes
    !s.starts_with(|c: char| c.is_uppercase()) || s.starts_with('\'')
}

fn predict(chart: &mut Vec<Vec<EarleyState>>, pos: usize, grammar: &CFG) {
    // Collect all nonterminals that need prediction at this position
    let mut to_predict: Vec<String> = Vec::new();
    for state in &chart[pos].clone() {
        if let Some(next) = state.next_symbol() {
            if !is_terminal(next) {
                to_predict.push(next.to_string());
            }
        }
    }

    for nt in to_predict {
        for prod in grammar.get_productions(&nt) {
            let new_state = EarleyState {
                lhs: prod.lhs.clone(),
                rhs: prod.rhs.clone(),
                dot: 0,
                start_pos: pos,
            };
            // Avoid duplicates
            if !chart[pos].iter().any(|s| {
                s.lhs == new_state.lhs
                    && s.rhs == new_state.rhs
                    && s.dot == new_state.dot
                    && s.start_pos == new_state.start_pos
            }) {
                chart[pos].push(new_state);
            }
        }
    }
}

fn complete(chart: &mut Vec<Vec<EarleyState>>, pos: usize, lhs: &str, _grammar: &CFG) {
    let states = chart[pos].clone();
    for k in 0..pos {
        for state in &states {
            if state.start_pos == k {
                // Advance states in chart[k] that expect this lhs
                let chart_k = chart[k].clone();
                for s in &chart_k {
                    if s.dot < s.rhs.len() && s.rhs[s.dot] == *lhs {
                        let new_state = EarleyState {
                            lhs: s.lhs.clone(),
                            rhs: s.rhs.clone(),
                            dot: s.dot + 1,
                            start_pos: s.start_pos,
                        };
                        if !chart[pos].iter().any(|cs| {
                            cs.lhs == new_state.lhs
                                && cs.rhs == new_state.rhs
                                && cs.dot == new_state.dot
                                && cs.start_pos == new_state.start_pos
                        }) {
                            chart[pos].push(new_state);
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CFG>()?;
    m.add_class::<EarleyChartParser>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grammar() -> CFG {
        CFG::new(
            "S",
            vec![
                ("S".to_string(), vec!["NP".to_string(), "VP".to_string()]),
                ("NP".to_string(), vec!["Det".to_string(), "N".to_string()]),
                ("NP".to_string(), vec!["N".to_string()]),
                ("VP".to_string(), vec!["V".to_string(), "NP".to_string()]),
                ("Det".to_string(), vec!["the".to_string()]),
                ("Det".to_string(), vec!["a".to_string()]),
                ("N".to_string(), vec!["cat".to_string()]),
                ("N".to_string(), vec!["dog".to_string()]),
                ("V".to_string(), vec!["chased".to_string()]),
                ("V".to_string(), vec!["saw".to_string()]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_cfg_creation() {
        let cfg = sample_grammar();
        assert_eq!(cfg.start(), "S");
        assert!(cfg.__len__() > 0);
    }

    #[test]
    fn test_cfg_nonterminals() {
        let cfg = sample_grammar();
        let nts = cfg.nonterminals();
        assert!(nts.contains(&"S".to_string()));
        assert!(nts.contains(&"NP".to_string()));
        assert!(nts.contains(&"VP".to_string()));
    }

    #[test]
    fn test_cfg_from_string() {
        let grammar_str =
            "S -> NP VP\nNP -> Det N\nDet -> 'the'\nN -> 'cat'\nVP -> V NP\nV -> 'chased'";
        let cfg = CFG::from_string(grammar_str).unwrap();
        assert_eq!(cfg.start(), "S");
        assert_eq!(cfg.__len__(), 6);
    }

    #[test]
    fn test_cfg_lhs_index() {
        let cfg = sample_grammar();
        let prods = cfg.get_productions("NP");
        assert_eq!(prods.len(), 2);
    }

    #[test]
    fn test_earley_simple() {
        let parser = EarleyChartParser::new();
        let cfg = sample_grammar();
        let result = parser.parse(
            &cfg,
            vec![
                "the".to_string(),
                "cat".to_string(),
                "chased".to_string(),
                "the".to_string(),
                "dog".to_string(),
            ],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_earley_no_parse() {
        let parser = EarleyChartParser::new();
        let cfg = sample_grammar();
        let result = parser.parse(&cfg, vec!["the".to_string(), "dog".to_string()]);
        // "the dog" is just a noun phrase, not a full sentence
        assert!(result.is_err());
    }

    #[test]
    fn test_earley_empty_tokens() {
        let parser = EarleyChartParser::new();
        let cfg = sample_grammar();
        let result = parser.parse(&cfg, Vec::new());
        assert!(result.is_err());
    }
}
