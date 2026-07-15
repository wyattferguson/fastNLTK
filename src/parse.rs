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
                if sym.starts_with(|c: char| c.is_uppercase()) || sym.starts_with('\'') {
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
            if line.is_empty() || line.starts_with('#') { continue; }
            let arrow_pos = line.find("->").ok_or_else(|| FastNltkError::GrammarParse(line.to_string()))?;
            let lhs = line[..arrow_pos].trim();
            let rhs_part = line[arrow_pos + 2..].trim();
            if start.is_empty() { start = lhs.to_string(); }
            for alt in rhs_part.split('|') {
                let rhs: Vec<String> = alt.split_whitespace()
                    .map(|s| if s.starts_with('\'') && s.ends_with('\'') && s.len() > 2 { s[1..s.len()-1].to_string() } else { s.to_string() })
                    .collect();
                if !rhs.is_empty() { productions.push((lhs.to_string(), rhs)); }
            }
        }
        if start.is_empty() { return Err(FastNltkError::GrammarParse("empty grammar".into()).into()); }
        Self::new(&start, productions)
    }

    fn start(&self) -> String { self.start_symbol.clone() }
    fn productions(&self) -> Vec<(String, Vec<String>)> {
        self.productions.iter().map(|p| (p.lhs.clone(), p.rhs.clone())).collect()
    }
    fn nonterminals(&self) -> Vec<String> { self.nonterminals.clone() }
    fn __len__(&self) -> usize { self.productions.len() }
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
        self.lhs_index.get(lhs)
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
    /// Used during tree reconstruction — maps (rhs_fragment) to children
    children: Vec<Vec<StateId>>,
}

impl EarleyState {
    fn new(lhs: String, rhs: Vec<String>, start_pos: usize) -> Self {
        Self { lhs, rhs: rhs.clone(), dot: 0, start_pos, children: Vec::new() }
    }

    fn advance(&self) -> Self {
        let mut s = self.clone();
        s.dot += 1;
        s
    }

    fn is_complete(&self) -> bool { self.dot >= self.rhs.len() }
    fn next_symbol(&self) -> Option<&str> {
        if self.dot < self.rhs.len() { Some(&self.rhs[self.dot]) } else { None }
    }
}

fn is_terminal(s: &str) -> bool {
    !s.starts_with(|c: char| c.is_uppercase()) || s.starts_with('\'')
}

// ── EarleyChartParser ──

#[pyclass(name = "EarleyChartParser", module = "fastnltk._rust")]
pub struct EarleyChartParser;

#[pymethods]
impl EarleyChartParser {
    #[new]
    const fn new() -> Self { Self }

    fn parse(&self, grammar: &CFG, tokens: Vec<String>) -> PyResult<Vec<String>> {
        let n = tokens.len();

        // chart[pos] = Vec<State>
        let mut chart: Vec<Vec<EarleyState>> = vec![Vec::new(); n + 1];

        // Initialize: add S' -> .S at position 0
        let start = format!("{}'", grammar.start_symbol);
        chart[0].push(EarleyState::new(start.clone(), vec![grammar.start_symbol.clone()], 0));

        // Predict from S'
        let mut to_predict = vec![grammar.start_symbol.clone()];
        let mut predicted: HashSet<String> = HashSet::new();
        predicted.insert(grammar.start_symbol.clone());
        while let Some(nt) = to_predict.pop() {
            for prod in grammar.get_productions(&nt) {
                let state = EarleyState::new(prod.lhs.clone(), prod.rhs.clone(), 0);
                let nt_key = prod.lhs.clone();
                if predicted.insert(nt_key) {
                    if !chart[0].iter().any(|s| s.lhs == state.lhs && s.rhs == state.rhs && s.dot == 0) {
                        chart[0].push(state);
                    }
                    for sym in &prod.rhs {
                        if !is_terminal(sym) { to_predict.push(sym.clone()); }
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
                            let mut pred = chart[k][ci].clone();
                            if pred.dot < pred.rhs.len() && pred.rhs[pred.dot] == state.lhs {
                                let mut advanced = pred.advance();
                                // Add backpointer: this completion used state at (i, j)
                                advanced.children.push(vec![(i, j)]);
                                // Also merge any existing backpointers from the predictor
                                if !chart[i].iter().any(|s| s.lhs == advanced.lhs && s.rhs == advanced.rhs && s.dot == advanced.dot && s.start_pos == advanced.start_pos) {
                                    chart[i].push(advanced);
                                }
                            }
                        }
                    }
                } else {
                    let next = state.next_symbol().unwrap();
                    if is_terminal(next) {
                        if i < n && next == &tokens[i] {
                            let mut s = state.advance();
                            s.children.push(vec![]); // terminal child
                            chart[i + 1].push(s);
                        }
                    } else {
                        // Predict
                        for prod in grammar.get_productions(next) {
                            let new_state = EarleyState::new(prod.lhs.clone(), prod.rhs.clone(), i);
                            if !chart[i].iter().any(|s| s.lhs == new_state.lhs && s.rhs == new_state.rhs && s.dot == 0 && s.start_pos == i) {
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
                if let Some(tree) = build_tree(&state, &chart, n, &tokens) {
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

/// Build a tree string by walking the chart backwards from a completed state.
fn build_tree(state: &EarleyState, chart: &[Vec<EarleyState>], end_pos: usize, _tokens: &[String]) -> Option<String> {
    if state.rhs.is_empty() {
        return Some(format!("({})", state.lhs));
    }

    let mut children: Vec<String> = Vec::new();
    let mut remaining = state.rhs.clone();
    let mut pos = end_pos;

    // Walk backwards through the chart to find which states completed this production
    while pos > 0 && !remaining.is_empty() {
        let sym = remaining.last().unwrap().clone();
        if is_terminal(&sym) {
            // Terminal: match from the end (most recent scanned token)
            // Find the scanned token ending at this position
            let mut found = false;
            for s in &chart[pos] {
                if s.is_complete() && s.rhs.len() == 1 && s.rhs[0] == sym && s.start_pos == pos - 1 {
                    children.push(sym.clone());
                    remaining.pop();
                    pos -= 1;
                    found = true;
                    break;
                }
            }
            if !found {
                // Try matching as a nonterminal lookup
                remaining.pop();
                children.push(sym.clone());
                pos -= 1;
            }
        } else {
            // Nonterminal: find the completed state for this LHS ending at pos
            let mut found = false;
            for start in (0..pos).rev() {
                for s in &chart[pos] {
                    if s.lhs == sym && s.is_complete() && s.start_pos == start {
                        if let Some(subtree) = build_tree(s, chart, pos, _tokens) {
                            children.push(subtree);
                            remaining.pop();
                            pos = start;
                            found = true;
                            break;
                        }
                    }
                }
                if found { break; }
            }
            if !found {
                remaining.pop();
            }
        }
    }

    // Add any remaining uneaten RHS symbols as empty
    for sym in remaining.iter().rev() {
        if is_terminal(sym) {
            children.push(sym.clone());
        } else {
            children.push(format!("({})", sym));
        }
    }

    children.reverse();
    Some(format!("({} {})", state.lhs, children.join(" ")))
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
        CFG::new("S", vec![
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
        ]).unwrap()
    }

    #[test]
    fn test_cfg_creation() { let cfg = sample_grammar(); assert_eq!(cfg.start(), "S"); }

    #[test]
    fn test_cfg_from_string() {
        let s = "S -> NP VP\nNP -> Det N\nDet -> 'the'\nN -> 'cat'\nVP -> V NP\nV -> 'chased'";
        let cfg = CFG::from_string(s).unwrap();
        assert_eq!(cfg.__len__(), 6);
    }

    #[test]
    fn test_earley_parse() {
        let parser = EarleyChartParser::new();
        let cfg = sample_grammar();
        let result = parser.parse(&cfg, vec!["the".into(), "cat".into(), "chased".into(), "the".into(), "dog".into()]);
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
