//! Tableau theorem prover — connected tableau calculus for FOL.
//!
//! NLTK equivalent: nltk.inference.tableau.TableauProver

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::inference::{Formula, ProverResult};

#[derive(Clone, Debug)]
struct SignedFormula {
    formula: Formula,
    sign: bool,
}

type Branch = Vec<SignedFormula>;

#[pyclass(name = "TableauProver", module = "fastnltk._rust")]
pub struct TableauProver {
    max_depth: usize,
}

#[pymethods]
impl TableauProver {
    #[new]
    #[pyo3(signature = (max_depth=200))]
    fn new(max_depth: usize) -> Self {
        TableauProver { max_depth }
    }

    #[pyo3(signature = (goal_str, assumptions=None))]
    fn prove(&self, goal_str: &str, assumptions: Option<Vec<String>>) -> PyResult<ProverResult> {
        let goal = parse_fol(goal_str)
            .ok_or_else(|| PyValueError::new_err(format!("Cannot parse: {goal_str}")))?;

        let mut branch = Branch::new();
        if let Some(assumptions) = assumptions {
            for a in assumptions {
                if let Some(f) = parse_fol(&a) {
                    branch.push(SignedFormula {
                        formula: f,
                        sign: true,
                    });
                }
            }
        }
        branch.push(SignedFormula {
            formula: goal.negate_nnf(),
            sign: true,
        });

        let branches = &mut vec![branch];
        let found = self.search(branches, 0);
        if found {
            Ok(ProverResult {
                success: true,
                proof: "Tableau proof found".into(),
            })
        } else {
            Ok(ProverResult {
                success: false,
                proof: "Tableau exhausted".into(),
            })
        }
    }
}

impl TableauProver {
    fn search(&self, branches: &mut Vec<Branch>, depth: usize) -> bool {
        if depth > self.max_depth || branches.is_empty() {
            return false;
        }

        // Take the first open branch
        let branch = branches.remove(0);

        // Check for closure
        if is_closed_simple(&branch) {
            if branches.is_empty() {
                return true; // All branches closed
            }
            return self.search(branches, depth + 1);
        }

        // If atomic and not closed, fail
        if is_atomic_simple(&branch) {
            return false;
        }

        // Find first non-atomic formula
        if let Some(idx) = branch.iter().position(|sf| !is_literal_simple(&sf.formula)) {
            let sf = branch[idx].clone();
            let mut base: Branch = branch.clone();
            base.remove(idx);
            let f = sf.formula.nnf();

            match f {
                Formula::And(children) => {
                    for child in children {
                        base.push(SignedFormula {
                            formula: child,
                            sign: sf.sign,
                        });
                    }
                    branches.insert(0, base);
                    self.search(branches, depth + 1)
                }
                Formula::Or(children) => {
                    for child in children {
                        let mut new_branch = base.clone();
                        new_branch.push(SignedFormula {
                            formula: child,
                            sign: sf.sign,
                        });
                        branches.insert(0, new_branch);
                    }
                    self.search(branches, depth + 1)
                }
                Formula::Imp(l, r) => {
                    // P -> Q becomes ~P | Q
                    let not_l = Formula::Not(l);
                    for child in vec![not_l, *r] {
                        let mut new_branch = base.clone();
                        new_branch.push(SignedFormula {
                            formula: child,
                            sign: sf.sign,
                        });
                        branches.insert(0, new_branch);
                    }
                    self.search(branches, depth + 1)
                }
                Formula::Forall(v, body) => {
                    let fresh = format!("_c{depth}");
                    let inst = sub_var(&body, &v, &fresh);
                    base.push(SignedFormula {
                        formula: inst,
                        sign: sf.sign,
                    });
                    base.push(SignedFormula {
                        formula: Formula::Forall(v, body),
                        sign: sf.sign,
                    });
                    branches.insert(0, base);
                    self.search(branches, depth + 1)
                }
                Formula::Exists(v, body) => {
                    let fresh = format!("_sk{depth}");
                    let inst = sub_var(&body, &v, &fresh);
                    base.push(SignedFormula {
                        formula: inst,
                        sign: sf.sign,
                    });
                    branches.insert(0, base);
                    self.search(branches, depth + 1)
                }
                _ => {
                    branches.insert(0, base);
                    self.search(branches, depth + 1)
                }
            }
        } else {
            false
        }
    }
}

fn is_literal_simple(f: &Formula) -> bool {
    matches!(
        f,
        Formula::Atom(_, _)
            | Formula::Not(_)
            | Formula::True
            | Formula::False
            | Formula::Equal(_, _)
    )
}

fn is_atomic_simple(b: &[SignedFormula]) -> bool {
    b.iter().all(|sf| is_literal_simple(&sf.formula))
}

fn is_closed_simple(b: &[SignedFormula]) -> bool {
    for i in 0..b.len() {
        for j in (i + 1)..b.len() {
            if contradict(&b[i].formula, &b[j].formula) {
                return true;
            }
        }
    }
    false
}

fn contradict(a: &Formula, b: &Formula) -> bool {
    match (a, b) {
        (Formula::Atom(p1, a1), Formula::Not(inner))
        | (Formula::Not(inner), Formula::Atom(p1, a1)) => {
            if let Formula::Atom(p2, a2) = inner.as_ref() {
                p1 == p2 && a1 == a2
            } else {
                false
            }
        }
        (Formula::True, Formula::False) | (Formula::False, Formula::True) => true,
        _ => false,
    }
}

fn sub_var(f: &Formula, var: &str, repl: &str) -> Formula {
    match f {
        Formula::Atom(p, args) => {
            let new_args: Vec<String> = args
                .iter()
                .map(|a| if a == var { repl.into() } else { a.clone() })
                .collect();
            Formula::Atom(p.clone(), new_args)
        }
        Formula::Not(inner) => Formula::Not(Box::new(sub_var(inner, var, repl))),
        Formula::And(children) => {
            Formula::And(children.iter().map(|c| sub_var(c, var, repl)).collect())
        }
        Formula::Or(children) => {
            Formula::Or(children.iter().map(|c| sub_var(c, var, repl)).collect())
        }
        Formula::Forall(v, body) if v != var => {
            Formula::Forall(v.clone(), Box::new(sub_var(body, var, repl)))
        }
        Formula::Exists(v, body) if v != var => {
            Formula::Exists(v.clone(), Box::new(sub_var(body, var, repl)))
        }
        Formula::Equal(a, b) => Formula::Equal(
            if a == var { repl.into() } else { a.clone() },
            if b == var { repl.into() } else { b.clone() },
        ),
        other => other.clone(),
    }
}

fn parse_fol(s: &str) -> Option<Formula> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(pos) = find_conn(s, "&") {
        return Some(Formula::And(vec![
            parse_fol(&s[..pos])?,
            parse_fol(&s[pos + 1..])?,
        ]));
    }
    if let Some(pos) = find_conn(s, "|") {
        return Some(Formula::Or(vec![
            parse_fol(&s[..pos])?,
            parse_fol(&s[pos + 1..])?,
        ]));
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
    m.add_class::<TableauProver>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom() {
        let f = parse_fol("P(a)").unwrap();
        assert_eq!(f, Formula::Atom("P".into(), vec!["a".into()]));
    }

    #[test]
    fn test_parse_and() {
        let f = parse_fol("P & Q").unwrap();
        assert!(matches!(f, Formula::And(_)));
    }

    #[test]
    fn test_parse_forall() {
        let f = parse_fol("all x.P(x)").unwrap();
        assert!(matches!(f, Formula::Forall(_, _)));
    }

    #[test]
    fn test_tableau_excluded_middle() {
        let p = TableauProver::new(50);
        let r = p.prove("P | ~P", None).unwrap();
        assert!(r.success, "Excluded middle should be provable");
    }
}
