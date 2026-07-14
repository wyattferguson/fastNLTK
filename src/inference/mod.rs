//! Inference — Tableau + Resolution theorem provers.
//!
//! Proves first-order logic formulas using:
//! - Connected tableau calculus with free variables
//! - Resolution with CNF + saturation
//!
//! NLTK equivalent: nltk.inference.api + nltk.inference.tableau + nltk.inference.resolution

use pyo3::prelude::*;

pub mod tableau;
pub mod resolution;

#[pyclass(name = "ProverResult", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct ProverResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub proof: String,
}

#[pymethods]
impl ProverResult {
    fn __str__(&self) -> String {
        if self.success {
            format!("True (proof: {})", self.proof)
        } else {
            format!("Unknown: {}", self.proof)
        }
    }
}

/// Simple FOL formula types used by both provers.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Formula {
    True,
    False,
    Atom(String, Vec<String>),
    Not(Box<Formula>),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Imp(Box<Formula>, Box<Formula>),
    Iff(Box<Formula>, Box<Formula>),
    Forall(String, Box<Formula>),
    Exists(String, Box<Formula>),
    Equal(String, String),
}

impl Formula {
    pub fn nnf(self) -> Self {
        match self {
            Formula::Not(inner) => Self::negate_nnf(*inner),
            other => other,
        }
    }

    pub fn negate_nnf(self) -> Self {
        match self {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Atom(p, args) => Formula::Not(Box::new(Formula::Atom(p, args))),
            Formula::Not(inner) => inner.nnf(),
            Formula::And(children) => Formula::Or(children.into_iter().map(|c| c.negate_nnf()).collect()),
            Formula::Or(children) => Formula::And(children.into_iter().map(|c| c.negate_nnf()).collect()),
            Formula::Imp(l, r) => Formula::And(vec![l.nnf(), r.negate_nnf()]),
            Formula::Iff(l, r) => Formula::Or(vec![
                Formula::And(vec![l.clone().nnf(), r.clone().nnf()]),
                Formula::And(vec![l.negate_nnf(), r.negate_nnf()]),
            ]),
            Formula::Forall(v, body) => Formula::Exists(v, Box::new(body.negate_nnf())),
            Formula::Exists(v, body) => Formula::Forall(v, Box::new(body.negate_nnf())),
            Formula::Equal(a, b) => Formula::Not(Box::new(Formula::Equal(a, b))),
        }
    }

    pub fn to_cnf(self) -> Vec<Vec<Literal>> {
        let nnf = self.nnf();
        let dis = nnf.distribute();
        let mut clauses = Vec::new();
        push_and_clauses(dis, &mut clauses);
        clauses
    }

    fn distribute(self) -> Self {
        match self {
            Formula::Or(children) => {
                let mut distributed: Vec<Vec<Formula>> = vec![vec![]];
                for child in children {
                    let child = child.distribute();
                    match child {
                        Formula::And(sub_children) => {
                            let mut new_dist = Vec::new();
                            for existing in &distributed {
                                for sub in &sub_children {
                                    let mut combined = existing.clone();
                                    combined.push(sub.clone());
                                    new_dist.push(combined);
                                }
                            }
                            distributed = new_dist;
                        }
                        other => {
                            for existing in &mut distributed {
                                existing.push(other.clone());
                            }
                        }
                    }
                }
                let flat: Vec<Formula> = distributed.into_iter().flatten().collect();
                Formula::Or(flat)
            }
            Formula::And(children) => Formula::And(children.into_iter().map(|c| c.distribute()).collect()),
            Formula::Imp(l, r) => Formula::Or(vec![l.negate_nnf(), r.distribute()]).distribute(),
            Formula::Iff(l, r) => Formula::Or(vec![
                Formula::And(vec![l.clone().nnf(), r.clone().nnf()]),
                Formula::And(vec![l.negate_nnf(), r.negate_nnf()]),
            ]).distribute(),
            other => other,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Literal {
    Pos(String, Vec<String>),
    Neg(String, Vec<String>),
}

fn push_and_clauses(f: Formula, clauses: &mut Vec<Vec<Literal>>) {
    match f {
        Formula::And(children) => {
            for c in children {
                push_and_clauses(c, clauses);
            }
        }
        other => {
            clauses.push(collect_literals(other));
        }
    }
}

fn collect_literals(f: Formula) -> Vec<Literal> {
    match f {
        Formula::Or(children) => {
            children.into_iter().map(|c| match c {
                Formula::Atom(p, args) => Literal::Pos(p, args),
                Formula::Not(inner) => match *inner {
                    Formula::Atom(p, args) => Literal::Neg(p, args),
                    _ => Literal::Pos("__CMP__".into(), vec![]),
                },
                _ => Literal::Pos("__CMP__".into(), vec![]),
            }).collect()
        }
        Formula::Atom(p, args) => vec![Literal::Pos(p, args)],
        Formula::Not(inner) => match *inner {
            Formula::Atom(p, args) => vec![Literal::Neg(p, args)],
            _ => vec![Literal::Pos("__CMP__".into(), vec![])],
        },
        _ => vec![Literal::Pos("__CMP__".into(), vec![])],
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ProverResult>()?;
    tableau::register_module(m)?;
    resolution::register_module(m)?;
    Ok(())
}
