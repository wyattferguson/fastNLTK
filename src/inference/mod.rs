//! Inference — Tableau + Resolution theorem provers.
//!
//! Proves first-order logic formulas using:
//! - Connected tableau calculus with free variables
//! - Resolution with CNF + saturation
//!
//! NLTK equivalent: nltk.inference.api + nltk.inference.tableau + nltk.inference.resolution

use pyo3::prelude::*;

pub mod discourse;
pub mod nonmonotonic;
pub mod resolution;
pub mod tableau;

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
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
    Imp(Box<Self>, Box<Self>),
    #[allow(dead_code)]
    Iff(Box<Self>, Box<Self>),
    Forall(String, Box<Self>),
    Exists(String, Box<Self>),
    Equal(String, String),
}

impl Formula {
    pub fn nnf(self) -> Self {
        match self {
            Self::Not(inner) => Self::negate_nnf(*inner),
            other => other,
        }
    }

    pub fn negate_nnf(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Atom(p, args) => Self::Not(Box::new(Self::Atom(p, args))),
            Self::Not(inner) => inner.nnf(),
            Self::And(children) => {
                Self::Or(children.into_iter().map(Self::negate_nnf).collect())
            }
            Self::Or(children) => {
                Self::And(children.into_iter().map(Self::negate_nnf).collect())
            }
            Self::Imp(l, r) => Self::And(vec![l.nnf(), r.negate_nnf()]),
            Self::Iff(l, r) => Self::Or(vec![
                Self::And(vec![l.clone().nnf(), r.clone().nnf()]),
                Self::And(vec![l.negate_nnf(), r.negate_nnf()]),
            ]),
            Self::Forall(v, body) => Self::Exists(v, Box::new(body.negate_nnf())),
            Self::Exists(v, body) => Self::Forall(v, Box::new(body.negate_nnf())),
            Self::Equal(a, b) => Self::Not(Box::new(Self::Equal(a, b))),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_cnf(self) -> Vec<Vec<Literal>> {
        let nnf = self.nnf();
        let dis = nnf.distribute();
        let mut clauses = Vec::new();
        push_and_clauses(dis, &mut clauses);
        clauses
    }

    fn distribute(self) -> Self {
        match self {
            Self::Or(children) => {
                let mut distributed: Vec<Vec<Self>> = vec![vec![]];
                for child in children {
                    let child = child.distribute();
                    match child {
                        Self::And(sub_children) => {
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
                let flat: Vec<Self> = distributed.into_iter().flatten().collect();
                Self::Or(flat)
            }
            Self::And(children) => {
                Self::And(children.into_iter().map(Self::distribute).collect())
            }
            Self::Imp(l, r) => Self::Or(vec![l.negate_nnf(), r.distribute()]).distribute(),
            Self::Iff(l, r) => Self::Or(vec![
                Self::And(vec![l.clone().nnf(), r.clone().nnf()]),
                Self::And(vec![l.negate_nnf(), r.negate_nnf()]),
            ])
            .distribute(),
            other => other,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Literal {
    Pos(String, Vec<String>),
    Neg(String, Vec<String>),
}

impl Ord for Literal {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Pos(p1, a1), Self::Pos(p2, a2)) | (Self::Neg(p1, a1), Self::Neg(p2, a2)) => {
                p1.cmp(p2).then_with(|| a1.cmp(a2))
            }
            (Self::Pos(..), Self::Neg(..)) => std::cmp::Ordering::Less,
            (Self::Neg(..), Self::Pos(..)) => std::cmp::Ordering::Greater,
        }
    }
}

impl PartialOrd for Literal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
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
        Formula::Or(children) => children
            .into_iter()
            .map(|c| match c {
                Formula::Atom(p, args) => Literal::Pos(p, args),
                Formula::Not(inner) => match *inner {
                    Formula::Atom(p, args) => Literal::Neg(p, args),
                    _ => Literal::Pos("__CMP__".into(), vec![]),
                },
                _ => Literal::Pos("__CMP__".into(), vec![]),
            })
            .collect(),
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
    discourse::register_module(m)?;
    nonmonotonic::register_module(m)?;
    Ok(())
}
