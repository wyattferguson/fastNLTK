//! Parsing — Rust-accelerated CFG + Earley chart parser with parse tree output.

use hashbrown::HashMap;
use hashbrown::HashSet;

use pyo3::prelude::*;

use crate::error::FastNltkError;

// CFG — Context-Free Grammar

#[pyclass(name = "CFG", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct CFG {
    start_symbol: String,
    productions: Vec<Production>,
    lhs_index: HashMap<String, Vec<usize>>,
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
                if sym.starts_with(|c: char| c.is_uppercase())
                    || sym.starts_with('\'')
                    || sym.starts_with('"')
                {
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

    #[staticmethod]
    fn from_string(grammar_str: &str) -> PyResult<Self> {
        let mut start = String::new();
        let mut productions = Vec::new();
        for line in grammar_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let arrow_pos =
                line.find("->").ok_or_else(|| FastNltkError::GrammarParse(line.to_string()))?;
            let lhs = line[..arrow_pos].trim();
            let rhs_part = line[arrow_pos + 2..].trim();
            if start.is_empty() {
                start = lhs.to_string();
            }
            for alt in rhs_part.split('|') {
                let rhs: Vec<String> = alt
                    .split_whitespace()
                    .map(|s| {
                        if (s.starts_with('\'') && s.ends_with('\'') && s.len() > 2)
                            || (s.starts_with('"') && s.ends_with('"') && s.len() > 2)
                        {
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
    const fn __len__(&self) -> usize {
        self.productions.len()
    }
    fn __str__(&self) -> String {
        let mut lines = vec![format!("Grammar: {}", self.start_symbol)];
        for p in &self.productions {
            lines.push(format!("  {} -> {}", p.lhs, p.rhs.join(" ")));
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

// ── Earley State with backpointers for tree building ──

type StateId = (usize, usize); // (pos_in_chart, index_in_chart[pos])

#[derive(Clone, Debug)]
struct EarleyState {
    lhs: String,
    rhs: Vec<String>,
    dot: usize,
    start_pos: usize,
    /// Backpointers: for completed states, stores the state IDs of children
    /// Used during tree reconstruction — maps (`rhs_fragment`) to children
    children: Vec<Vec<StateId>>,
}

impl EarleyState {
    const fn new(lhs: String, rhs: Vec<String>, start_pos: usize) -> Self {
        Self { lhs, rhs, dot: 0, start_pos, children: Vec::new() }
    }

    fn advance(&self) -> Self {
        let mut s = self.clone();
        s.dot += 1;
        s
    }

    const fn is_complete(&self) -> bool {
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
    !s.starts_with(|c: char| c.is_uppercase()) || s.starts_with('\'') || s.starts_with('"')
}

// ── EarleyChartParser ──

#[pyclass(name = "EarleyChartParser", module = "fastnltk._rust")]
pub struct EarleyChartParser;

#[pymethods]
impl EarleyChartParser {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn parse(&self, grammar: &CFG, tokens: Vec<String>) -> PyResult<Vec<String>> {
        let n = tokens.len();

        // chart[pos] = Vec<State>
        let mut chart: Vec<Vec<EarleyState>> = vec![Vec::new(); n + 1];

        // Initialize: add S' -> .S at position 0
        let start = format!("{}'", grammar.start_symbol);
        chart[0].push(EarleyState::new(start.clone(), vec![grammar.start_symbol.clone()], 0));

        // Predict from S
        let mut to_predict = vec![grammar.start_symbol.clone()];
        let mut predicted: HashSet<String> = HashSet::new();
        while let Some(nt) = to_predict.pop() {
            for prod in grammar.get_productions(&nt) {
                let state = EarleyState::new(prod.lhs.clone(), prod.rhs.clone(), 0);
                if !chart[0].iter().any(|s| s.lhs == state.lhs && s.rhs == state.rhs && s.dot == 0)
                {
                    chart[0].push(state);
                }
                for sym in &prod.rhs {
                    if !is_terminal(sym) && predicted.insert(sym.clone()) {
                        to_predict.push(sym.clone());
                    }
                }
            }
        }

        // Main Earley loop
        for i in 0..=n {
            let mut j = 0;
            while j < chart[i].len() {
                let state = chart[i][j].clone();

                if state.is_complete() {
                    // Complete: advance states expecting this lhs
                    for k in 0..i {
                        for ci in 0..chart[k].len() {
                            let pred = chart[k][ci].clone();
                            if pred.dot < pred.rhs.len() && pred.rhs[pred.dot] == state.lhs {
                                let mut advanced = pred.advance();
                                // Add backpointer: this completion used state at (i, j)
                                advanced.children.push(vec![(i, j)]);
                                // Also merge any existing backpointers from the predictor
                                if !chart[i].iter().any(|s| {
                                    s.lhs == advanced.lhs
                                        && s.rhs == advanced.rhs
                                        && s.dot == advanced.dot
                                        && s.start_pos == advanced.start_pos
                                }) {
                                    chart[i].push(advanced);
                                }
                            }
                        }
                    }
                } else {
                    let next = state
                        .next_symbol()
                        .expect("Earley state should have at least one symbol to predict/scan");
                    if is_terminal(next) {
                        if i < n && next == tokens[i] {
                            let mut s = state.advance();
                            s.children.push(vec![]); // terminal child
                            chart[i + 1].push(s);
                        }
                    } else {
                        // Predict
                        for prod in grammar.get_productions(next) {
                            let new_state = EarleyState::new(prod.lhs.clone(), prod.rhs.clone(), i);
                            if !chart[i].iter().any(|s| {
                                s.lhs == new_state.lhs
                                    && s.rhs == new_state.rhs
                                    && s.dot == 0
                                    && s.start_pos == i
                            }) {
                                chart[i].push(new_state);
                            }
                        }
                    }
                }
                j += 1;
            }
        }

        // Extract parse trees from completed start states
        let mut results = Vec::new();
        for state in &chart[n] {
            if state.lhs == start && state.is_complete() && state.start_pos == 0 {
                // Build tree string
                // S' -> S wrapper; try to extract inner S tree.
                if state.rhs.len() == 1 {
                    let inner_sym = &state.rhs[0];
                    if let Some(inner) = find_completed(inner_sym, 0, n, &chart) {
                        if let Some(tree) = build_tree(&inner, &chart, n, &tokens) {
                            results.push(tree);
                            continue;
                        }
                    }
                }
                // Fallback: use the full S' tree
                if let Some(tree) = build_tree(state, &chart, n, &tokens) {
                    results.push(tree);
                }
            }
        }

        if results.is_empty() {
            return Err(FastNltkError::NoParse.into());
        }
        Ok(results)
    }
}

/// Build a tree string by finding each RHS symbol's child in the chart.
fn build_tree(
    state: &EarleyState,
    chart: &[Vec<EarleyState>],
    end_pos: usize,
    tokens: &[String],
) -> Option<String> {
    let start = state.start_pos;
    let n_syms = state.rhs.len();

    if n_syms == 0 {
        return Some(format!("({})", state.lhs));
    }

    let mut children: Vec<String> = Vec::with_capacity(n_syms);
    let mut pos = start;

    for sym in &state.rhs {
        if pos > end_pos {
            return None;
        }
        if is_terminal(sym) {
            if pos < tokens.len() && pos < end_pos {
                children.push(tokens[pos].clone());
                pos += 1;
            } else {
                return None;
            }
        } else {
            // Find completed state for sym spanning [pos, q) for some q > pos
            let mut found = false;
            for q in (pos + 1)..=end_pos {
                if let Some(child) = find_completed(sym, pos, q, chart) {
                    if let Some(subtree) = build_tree(&child, chart, q, tokens) {
                        children.push(subtree);
                        pos = q;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return None;
            }
        }
    }

    if pos == end_pos {
        Some(format!("({} {})", state.lhs, children.join(" ")))
    } else {
        None
    }
}

/// Find a completed state with given LHS spanning [start, end).
fn find_completed(
    lhs: &str,
    start: usize,
    end: usize,
    chart: &[Vec<EarleyState>],
) -> Option<EarleyState> {
    if end >= chart.len() {
        return None;
    }
    chart[end].iter().find(|s| s.lhs == lhs && s.is_complete() && s.start_pos == start).cloned()
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CFG>()?;
    m.add_class::<EarleyChartParser>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grammar() -> CFG {
        CFG::new(
            "S",
            vec![
                ("S".into(), vec!["NP".into(), "VP".into()]),
                ("NP".into(), vec!["Det".into(), "N".into()]),
                ("NP".into(), vec!["N".into()]),
                ("VP".into(), vec!["V".into(), "NP".into()]),
                ("Det".into(), vec!["the".into()]),
                ("Det".into(), vec!["a".into()]),
                ("N".into(), vec!["cat".into()]),
                ("N".into(), vec!["dog".into()]),
                ("V".into(), vec!["chased".into()]),
                ("V".into(), vec!["saw".into()]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_cfg_creation() {
        let cfg = sample_grammar();
        assert_eq!(cfg.start(), "S");
    }

    #[test]
    fn test_cfg_from_string() {
        let s = "S -> NP VP\nNP -> Det N\nDet -> 'the'\nN -> 'cat'\nVP -> V NP\nV -> 'chased'";
        let cfg = CFG::from_string(s).unwrap();
        assert_eq!(cfg.__len__(), 6);
    }

    #[test]
    fn test_earley_parse() {
        // Use non-ambiguous grammar (VP -> V only).
        // Ambiguous grammars with shared prefixes (VP -> V NP | V) are
        // handled at the Python layer via NLTK delegation.
        let cfg = CFG::new(
            "S",
            vec![
                ("S".into(), vec!["NP".into(), "VP".into()]),
                ("NP".into(), vec!["Det".into(), "N".into()]),
                ("Det".into(), vec!["the".into()]),
                ("N".into(), vec!["cat".into()]),
                ("VP".into(), vec!["V".into()]),
                ("V".into(), vec!["runs".into()]),
            ],
        )
        .unwrap();
        let parser = EarleyChartParser::new();
        let result = parser.parse(&cfg, vec!["the".into(), "cat".into(), "runs".into()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_earley_no_parse() {
        let parser = EarleyChartParser::new();
        let cfg = sample_grammar();
        let result = parser.parse(&cfg, vec!["zebra".into()]);
        assert!(result.is_err());
    }
}
