//! Model evaluation for logical formulas.
//!
//! Evaluates `Expression` values in a first-order model with
//! domain, valuation, and variable assignment.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sem::expression::{var_name, Assignment, Expression, Individual, Valuation};
use crate::sem::parse::parse_expression;

/// Core Rust evaluation function (no JSON).
pub fn model_evaluate(
    expr: &Expression,
    valuation: &Valuation,
    domain: &[Individual],
    assignment: &Assignment,
) -> Result<bool, String> {
    match expr {
        Expression::Constant(name, _) => {
            if name == "true" {
                Ok(true)
            } else if name == "false" {
                Ok(false)
            } else {
                Ok(domain.contains(name))
            }
        }
        Expression::Variable(name, _) => Ok(assignment.contains_key(name)),
        Expression::Application(func, arg) => {
            let pred_name = match func.as_ref() {
                Expression::Variable(n, _) | Expression::Constant(n, _) => n.clone(),
                _ => return Err(format!("Expected predicate, got {func}")),
            };
            let arg_val = match arg.as_ref() {
                Expression::Variable(n, _) => assignment.get(n).cloned().unwrap_or_default(),
                Expression::Constant(n, _) => n.clone(),
                e => return Err(format!("Expected argument, got {e}")),
            };
            if let Some(extensions) = valuation.get(&pred_name) {
                Ok(extensions.iter().any(|t| t.len() == 1 && t[0] == arg_val))
            } else {
                Ok(false)
            }
        }
        Expression::And(a, b) => {
            if !model_evaluate(a, valuation, domain, assignment)? {
                return Ok(false);
            }
            model_evaluate(b, valuation, domain, assignment)
        }
        Expression::Or(a, b) => {
            if model_evaluate(a, valuation, domain, assignment)? {
                return Ok(true);
            }
            model_evaluate(b, valuation, domain, assignment)
        }
        Expression::Not(e) => Ok(!model_evaluate(e, valuation, domain, assignment)?),
        Expression::If(a, b) => {
            if !model_evaluate(a, valuation, domain, assignment)? {
                return Ok(true);
            }
            model_evaluate(b, valuation, domain, assignment)
        }
        Expression::Iff(a, b) => {
            let l = model_evaluate(a, valuation, domain, assignment)?;
            let r = model_evaluate(b, valuation, domain, assignment)?;
            Ok(l == r)
        }
        Expression::Equality(a, b) => {
            let a_val: Option<String> = match a.as_ref() {
                Expression::Variable(n, _) => assignment.get(n).cloned(),
                Expression::Constant(n, _) => Some(n.clone()),
                _ => None,
            };
            let b_val: Option<String> = match b.as_ref() {
                Expression::Variable(n, _) => assignment.get(n).cloned(),
                Expression::Constant(n, _) => Some(n.clone()),
                _ => None,
            };
            Ok(a_val.is_some() && a_val == b_val)
        }
        Expression::Exists(var, body) => {
            let vn = var_name(var);
            for ind in domain {
                let mut new_assign = assignment.clone();
                new_assign.insert(vn.clone(), ind.clone());
                if model_evaluate(body, valuation, domain, &new_assign)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Expression::All(var, body) => {
            let vn = var_name(var);
            for ind in domain {
                let mut new_assign = assignment.clone();
                new_assign.insert(vn.clone(), ind.clone());
                if !model_evaluate(body, valuation, domain, &new_assign)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Expression::Lambda(_, _) => Err("Cannot evaluate lambda directly".to_string()),
    }
}

// PyO3 wrappers

#[pyfunction]
#[pyo3(signature = (formula))]
fn fromstring(formula: &str) -> PyResult<String> {
    let expr = parse_expression(formula).map_err(PyValueError::new_err)?;
    Ok(format!("{expr}"))
}

#[pyfunction]
#[pyo3(signature = (formula))]
fn simplify(formula: &str) -> PyResult<String> {
    let expr = parse_expression(formula).map_err(PyValueError::new_err)?;
    Ok(format!("{}", expr.simplify()))
}

#[pyfunction]
#[pyo3(signature = (formula, valuation_json, domain_json, assignment_json=""))]
fn evaluate_formula(
    formula: &str,
    valuation_json: &str,
    domain_json: &str,
    assignment_json: &str,
) -> PyResult<bool> {
    let expr = parse_expression(formula).map_err(PyValueError::new_err)?;
    let valuation: Valuation = serde_json::from_str(valuation_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid valuation JSON: {e}")))?;
    let domain: Vec<Individual> = serde_json::from_str(domain_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid domain JSON: {e}")))?;
    let assignment: Assignment = if assignment_json.is_empty() {
        Assignment::new()
    } else {
        serde_json::from_str(assignment_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid assignment JSON: {e}")))?
    };
    model_evaluate(&expr, &valuation, &domain, &assignment).map_err(PyValueError::new_err)
}

/// Register pyfunctions with Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fromstring, m)?)?;
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_formula, m)?)?;
    Ok(())
}
