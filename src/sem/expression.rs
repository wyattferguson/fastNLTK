//! Expression types for logical semantics.
//!
//! Defines the core `Expression` and `Type` enums, display formatting,
//! free variable extraction, substitution, and beta-reduction.

use std::collections::HashSet;
use std::fmt;

/// An individual in the model domain.
pub type Individual = String;

/// A valuation maps predicate names to sets of tuples.
pub type Valuation = HashMap<String, Vec<Vec<Individual>>>;

use std::collections::HashMap;

/// An assignment maps variable names to individuals.
pub type Assignment = HashMap<String, Individual>;

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Constant(String, Option<Type>),
    Variable(String, Option<Type>),
    Application(Box<Self>, Box<Self>),
    Lambda(Box<Self>, Box<Self>),
    Exists(Box<Self>, Box<Self>),
    All(Box<Self>, Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    If(Box<Self>, Box<Self>),
    Iff(Box<Self>, Box<Self>),
    Equality(Box<Self>, Box<Self>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Entity,
    TruthValue,
    Fun(Box<Self>, Box<Self>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entity => write!(f, "e"),
            Self::TruthValue => write!(f, "t"),
            Self::Fun(arg, result) => write!(f, "<{arg},{result}>"),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(name, _) => write!(f, "{name}"),
            Self::Variable(name, _) => write!(f, "{name}"),
            Self::Application(fn_expr, arg) => {
                let fn_str = format!("{fn_expr}");
                let arg_str = format!("{arg}");
                if is_atomic(fn_expr) {
                    if is_atomic(arg) {
                        write!(f, "{fn_str}({arg_str})")
                    } else {
                        write!(f, "{fn_str}({arg_str})")
                    }
                } else {
                    write!(f, "({fn_str})({arg_str})")
                }
            }
            Self::Lambda(var, body) => write!(f, "\\{var}.{body}"),
            Self::Exists(var, body) => write!(f, "exists {var}.{body}"),
            Self::All(var, body) => write!(f, "all {var}.{body}"),
            Self::And(a, b) => write!(f, "({a} & {b})"),
            Self::Or(a, b) => write!(f, "({a} | {b})"),
            Self::Not(expr) => write!(f, "-{expr}"),
            Self::If(a, b) => write!(f, "({a} -> {b})"),
            Self::Iff(a, b) => write!(f, "({a} <-> {b})"),
            Self::Equality(a, b) => write!(f, "({a} = {b})"),
        }
    }
}

#[inline]
pub fn is_atomic(expr: &Expression) -> bool {
    matches!(expr, Expression::Constant(_, _) | Expression::Variable(_, _))
}

// Free variables
impl Expression {
    #[must_use]
    pub fn free_variables(&self) -> Vec<String> {
        let mut free = Vec::new();
        let mut bound: HashSet<String> = HashSet::new();
        self.collect_free(&mut bound, &mut free);
        free
    }

    fn collect_free(&self, bound: &mut HashSet<String>, free: &mut Vec<String>) {
        match self {
            Self::Variable(name, _) => {
                if !bound.contains(name) && !free.contains(name) {
                    free.push(name.clone());
                }
            }
            Self::Constant(_, _) => {}
            Self::Application(f, a) => {
                f.collect_free(bound, free);
                a.collect_free(bound, free);
            }
            Self::Lambda(var, body) => {
                let vn = var_name(var);
                bound.insert(vn.clone());
                body.collect_free(bound, free);
                bound.remove(&vn);
            }
            Self::Exists(var, body) | Self::All(var, body) => {
                let vn = var_name(var);
                bound.insert(vn.clone());
                body.collect_free(bound, free);
                bound.remove(&vn);
            }
            Self::And(a, b)
            | Self::Or(a, b)
            | Self::If(a, b)
            | Self::Iff(a, b)
            | Self::Equality(a, b) => {
                a.collect_free(bound, free);
                b.collect_free(bound, free);
            }
            Self::Not(expr) => expr.collect_free(bound, free),
        }
    }
}

pub fn var_name(var: &Expression) -> String {
    match var {
        Expression::Variable(name, _) => name.clone(),
        _ => String::new(),
    }
}

pub fn binder_type(expr: &Expression) -> Option<Type> {
    match expr {
        Expression::Variable(_, t) => t.clone(),
        _ => None,
    }
}

pub fn fresh_var(base: &str, avoid: &[String]) -> String {
    let mut i = 0;
    loop {
        let candidate = format!("{base}{i}");
        if !avoid.contains(&candidate) {
            return candidate;
        }
        i += 1;
    }
}

// Substitution
impl Expression {
    pub fn substitute(&self, var: &str, replacement: &Self) -> Self {
        match self {
            Self::Variable(name, _typ) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Self::Constant(_, _) => self.clone(),
            Self::Application(f, a) => Self::Application(
                Box::new(f.substitute(var, replacement)),
                Box::new(a.substitute(var, replacement)),
            ),
            Self::Lambda(binder, body) => {
                let bn = var_name(binder);
                if bn == var {
                    self.clone()
                } else if replacement.free_variables().contains(&bn) {
                    let new_name = fresh_var(&bn, &replacement.free_variables());
                    let new_binder = Self::Variable(new_name, binder_type(binder));
                    let new_body = body.substitute(&bn, &new_binder);
                    Self::Lambda(
                        Box::new(new_binder),
                        Box::new(new_body.substitute(var, replacement)),
                    )
                } else {
                    Self::Lambda(
                        Box::new(binder.substitute(var, replacement)),
                        Box::new(body.substitute(var, replacement)),
                    )
                }
            }
            Self::Exists(binder, body) | Self::All(binder, body) => {
                let bn = var_name(binder);
                if bn == var {
                    self.clone()
                } else if replacement.free_variables().contains(&bn) {
                    let new_name = fresh_var(&bn, &replacement.free_variables());
                    let new_binder = Self::Variable(new_name, binder_type(binder));
                    let new_body = body.substitute(&bn, &new_binder);
                    let quant = if matches!(self, Self::Exists(_, _)) {
                        Expression::Exists
                    } else {
                        Expression::All
                    };
                    quant(Box::new(new_binder), Box::new(new_body.substitute(var, replacement)))
                } else {
                    let quant = if matches!(self, Self::Exists(_, _)) {
                        Expression::Exists
                    } else {
                        Expression::All
                    };
                    quant(
                        Box::new(binder.substitute(var, replacement)),
                        Box::new(body.substitute(var, replacement)),
                    )
                }
            }
            Self::And(a, b) => Self::And(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Or(a, b) => Self::Or(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Not(expr) => Self::Not(Box::new(expr.substitute(var, replacement))),
            Self::If(a, b) => Self::If(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Iff(a, b) => Self::Iff(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Self::Equality(a, b) => Self::Equality(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
        }
    }

    pub fn simplify(&self) -> Self {
        match self {
            Self::Application(f, arg) => {
                let f_simpl = f.simplify();
                let arg_simpl = arg.simplify();
                match &f_simpl {
                    Self::Lambda(binder, body) => {
                        let bn = var_name(binder);
                        body.substitute(&bn, &arg_simpl).simplify()
                    }
                    _ => Self::Application(Box::new(f_simpl), Box::new(arg_simpl)),
                }
            }
            Self::Lambda(binder, body) => {
                Self::Lambda(binder.clone(), Box::new(body.simplify()))
            }
            Self::Exists(binder, body) => {
                Self::Exists(binder.clone(), Box::new(body.simplify()))
            }
            Self::All(binder, body) => {
                Self::All(binder.clone(), Box::new(body.simplify()))
            }
            Self::And(a, b) => {
                Self::And(Box::new(a.simplify()), Box::new(b.simplify()))
            }
            Self::Or(a, b) => Self::Or(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Not(expr) => Self::Not(Box::new(expr.simplify())),
            Self::If(a, b) => Self::If(Box::new(a.simplify()), Box::new(b.simplify())),
            Self::Iff(a, b) => {
                Self::Iff(Box::new(a.simplify()), Box::new(b.simplify()))
            }
            Self::Equality(a, b) => {
                Self::Equality(Box::new(a.simplify()), Box::new(b.simplify()))
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sem::parse::parse_expression;

    #[test]
    fn test_free_variables() {
        let expr = parse_expression("walk(x) & man(y)").unwrap();
        let free = expr.free_variables();
        assert!(free.contains(&"x".to_string()));
        assert!(free.contains(&"y".to_string()));
    }

    #[test]
    fn test_free_variables_bound() {
        let expr = parse_expression("exists x.walk(x)").unwrap();
        let free = expr.free_variables();
        assert!(!free.contains(&"x".to_string()));
    }

    #[test]
    fn test_substitution() {
        let expr = parse_expression("walk(x)").unwrap();
        let john = Expression::Constant("john".to_string(), None);
        let result = expr.substitute("x", &john);
        assert_eq!(format!("{result}"), "walk(john)");
    }

    #[test]
    fn test_beta_reduction() {
        let expr = parse_expression(r"(\x.walk(x))(john)").unwrap();
        let simplified = expr.simplify();
        assert_eq!(format!("{simplified}"), "walk(john)");
    }
}
