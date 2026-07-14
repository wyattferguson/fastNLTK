//! Resolution theorem prover — CNF + binary resolution for FOL.
//!
//! NLTK equivalent: nltk.inference.resolution.ResolutionProver

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::inference::{Formula, Literal, ProverResult};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Clause {
    literals: Vec<Literal>,
}

impl Clause {
    fn empty() -> Self {
        Clause { literals: vec![] }
    }
    fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
    fn contains(&self, lit: &Literal) -> bool {
        self.literals.contains(lit)
    }
}

#[pyclass(name = "ResolutionProver", module = "fastnltk._rust")]
pub struct ResolutionProver {
    max_iterations: usize,
}

#[pymethods]
impl ResolutionProver {
    #[new]
    #[pyo3(signature = (max_iterations=1000))]
    fn new(max_iterations: usize) -> Self {
        ResolutionProver { max_iterations }
    }

    #[pyo3(signature = (goal_str, assumptions=None))]
    fn prove(&self, goal_str: &str, assumptions: Option<Vec<String>>) -> PyResult<ProverResult> {
        let goal = parse_fol(goal_str)
            .ok_or_else(|| PyValueError::new_err(format!("Cannot parse: {goal_str}")))?;

        let mut clauses: Vec<Clause> = Vec::new();

        // Negate goal for proof by contradiction
        let neg = goal.negate_nnf();
        for lits in neg.to_cnf() {
            clauses.push(Clause { literals: lits });
        }

        if let Some(assumptions) = assumptions {
            for a in assumptions {
                if let Some(f) = parse_fol(&a) {
                    for lits in f.to_cnf() {
                        clauses.push(Clause { literals: lits });
                    }
                }
            }
        }

        let mut sos: Vec<Clause> = clauses.clone();
        let mut usable: Vec<Clause> = Vec::new();

        for iter in 0..self.max_iterations {
            if sos.is_empty() {
                break;
            }
            let clause = sos.remove(0);
            if clause.is_empty() {
                return Ok(ProverResult {
                    success: true,
                    proof: format!("Resolution proof found ({} iterations)", iter + 1),
                });
            }
            usable.push(clause.clone());
            for other in &usable {
                if let Some(r) = resolve_clause(&clause, other) {
                    if r.is_empty() {
                        return Ok(ProverResult {
                            success: true,
                            proof: format!("Resolution proof found ({} iterations)", iter + 1),
                        });
                    }
                    if !sos.contains(&r) && !usable.contains(&r) {
                        let subsumed = sos.iter().any(|c| subsumes_clause(&r, c));
                        if !subsumed {
                            sos.push(r);
                        }
                    }
                }
            }
        }

        Ok(ProverResult { success: false, proof: "Resolution exhausted".into() })
    }
}

fn resolve_clause(c1: &Clause, c2: &Clause) -> Option<Clause> {
    for l1 in &c1.literals {
        for l2 in &c2.literals {
            if comp(l1, l2) {
                let mut new_lits: Vec<Literal> = c1
                    .literals
                    .iter()
                    .filter(|l| *l != l1)
                    .chain(c2.literals.iter().filter(|l| *l != l2))
                    .cloned()
                    .collect();
                new_lits.sort();
                new_lits.dedup();
                new_lits = factor_clause(new_lits);
                return Some(Clause { literals: new_lits });
            }
        }
    }
    None
}

fn comp(l1: &Literal, l2: &Literal) -> bool {
    match (l1, l2) {
        (Literal::Pos(p1, a1), Literal::Neg(p2, a2))
        | (Literal::Neg(p1, a1), Literal::Pos(p2, a2)) => p1 == p2 && a1 == a2,
        _ => false,
    }
}

fn factor_clause(mut lits: Vec<Literal>) -> Vec<Literal> {
    lits.sort();
    lits.dedup();
    lits
}

fn subsumes_clause(a: &Clause, b: &Clause) -> bool {
    a.literals.iter().all(|l| b.contains(l))
}

fn parse_fol(s: &str) -> Option<Formula> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(pos) = find_conn(s, "&") {
        return Some(Formula::And(vec![parse_fol(&s[..pos])?, parse_fol(&s[pos + 1..])?]));
    }
    if let Some(pos) = find_conn(s, "|") {
        return Some(Formula::Or(vec![parse_fol(&s[..pos])?, parse_fol(&s[pos + 1..])?]));
    }
    if let Some(pos) = find_conn(s, "->") {
        return Some(Formula::Imp(
            Box::new(parse_fol(&s[..pos])?),
            Box::new(parse_fol(&s[pos + 2..])?),
        ));
    }
    if s.starts_with('-') || s.starts_with('~') {
        return Some(Formula::Not(Box::new(parse_fol(&s[1..].trim())?)));
    }
    if s.starts_with("all ") || s.starts_with("forall ") {
        let rest = s.trim_start_matches("all ").trim_start_matches("forall ");
        let (var, body) = split_q(rest)?;
        return Some(Formula::Forall(var, Box::new(parse_fol(&body)?)));
    }
    if s.starts_with("exists ") || s.starts_with("exist ") {
        let rest = s.trim_start_matches("exists ").trim_start_matches("exist ");
        let (var, body) = split_q(rest)?;
        return Some(Formula::Exists(var, Box::new(parse_fol(&body)?)));
    }
    if s.contains('(') && s.contains(')') {
        let paren = s.find('(')?;
        let close = s.rfind(')')?;
        let pred = s[..paren].trim().to_string();
        let args: Vec<String> = s[paren + 1..close]
            .split(',')
            .map(|a| a.trim().to_string())
            .filter(|a| !a.is_empty())
            .collect();
        return Some(Formula::Atom(pred, args));
    }
    if s.chars().all(|c| c.is_alphabetic() || c == '_') {
        return Some(Formula::Atom(s.to_string(), vec![]));
    }
    None
}

fn find_conn(s: &str, conn: &str) -> Option<usize> {
    let mut depth: usize = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ => {
                if depth == 0 && s[i..].starts_with(conn) {
                    if i > 0
                        && s.as_bytes()
                            .get(i - 1)
                            .copied()
                            .map_or(false, |b| b.is_ascii_alphanumeric())
                    {
                        continue;
                    }
                    return Some(i);
                }
            }
        }
    }
    None
}

fn split_q(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    let dot = s.find('.')?;
    let var = s[..dot].trim().to_string();
    let body = s[dot + 1..].trim().to_string();
    if var.is_empty() || body.is_empty() {
        None
    } else {
        Some((var, body))
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResolutionProver>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_res_excluded_middle() {
        let p = ResolutionProver::new(100);
        let r = p.prove("P | ~P", None).unwrap();
        assert!(r.success, "Excluded middle provable via resolution");
    }

    #[test]
    fn test_resolve_empty() {
        let c1 = Clause { literals: vec![Literal::Pos("P".into(), vec![])] };
        let c2 = Clause { literals: vec![Literal::Neg("P".into(), vec![])] };
        let r = resolve_clause(&c1, &c2);
        assert!(r.is_some());
        assert!(r.unwrap().is_empty());
    }
}
