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
    Application(Box<Expression>, Box<Expression>),
    Lambda(Box<Expression>, Box<Expression>),
    Exists(Box<Expression>, Box<Expression>),
    All(Box<Expression>, Box<Expression>),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
    If(Box<Expression>, Box<Expression>),
    Iff(Box<Expression>, Box<Expression>),
    Equality(Box<Expression>, Box<Expression>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Entity,
    TruthValue,
    Fun(Box<Type>, Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Entity => write!(f, "e"),
            Type::TruthValue => write!(f, "t"),
            Type::Fun(arg, result) => write!(f, "<{arg},{result}>"),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Constant(name, _) => write!(f, "{name}"),
            Expression::Variable(name, _) => write!(f, "{name}"),
            Expression::Application(fn_expr, arg) => {
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
            Expression::Lambda(var, body) => write!(f, "\\{var}.{body}"),
            Expression::Exists(var, body) => write!(f, "exists {var}.{body}"),
            Expression::All(var, body) => write!(f, "all {var}.{body}"),
            Expression::And(a, b) => write!(f, "({a} & {b})"),
            Expression::Or(a, b) => write!(f, "({a} | {b})"),
            Expression::Not(expr) => write!(f, "-{expr}"),
            Expression::If(a, b) => write!(f, "({a} -> {b})"),
            Expression::Iff(a, b) => write!(f, "({a} <-> {b})"),
            Expression::Equality(a, b) => write!(f, "({a} = {b})"),
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
            Expression::Variable(name, _) => {
                if !bound.contains(name) && !free.contains(name) {
                    free.push(name.clone());
                }
            }
            Expression::Constant(_, _) => {}
            Expression::Application(f, a) => {
                f.collect_free(bound, free);
                a.collect_free(bound, free);
            }
            Expression::Lambda(var, body) => {
                let vn = var_name(var);
                bound.insert(vn.clone());
                body.collect_free(bound, free);
                bound.remove(&vn);
            }
            Expression::Exists(var, body) | Expression::All(var, body) => {
                let vn = var_name(var);
                bound.insert(vn.clone());
                body.collect_free(bound, free);
                bound.remove(&vn);
            }
            Expression::And(a, b)
            | Expression::Or(a, b)
            | Expression::If(a, b)
            | Expression::Iff(a, b)
            | Expression::Equality(a, b) => {
                a.collect_free(bound, free);
                b.collect_free(bound, free);
            }
            Expression::Not(expr) => expr.collect_free(bound, free),
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
    pub fn substitute(&self, var: &str, replacement: &Expression) -> Expression {
        match self {
            Expression::Variable(name, _typ) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Expression::Constant(_, _) => self.clone(),
            Expression::Application(f, a) => Expression::Application(
                Box::new(f.substitute(var, replacement)),
                Box::new(a.substitute(var, replacement)),
            ),
            Expression::Lambda(binder, body) => {
                let bn = var_name(binder);
                if bn == var {
                    self.clone()
                } else if replacement.free_variables().contains(&bn) {
                    let new_name = fresh_var(&bn, &replacement.free_variables());
                    let new_binder = Expression::Variable(new_name.clone(), binder_type(binder));
                    let new_body = body.substitute(&bn, &new_binder);
                    Expression::Lambda(
                        Box::new(new_binder),
                        Box::new(new_body.substitute(var, replacement)),
                    )
                } else {
                    Expression::Lambda(
                        Box::new(binder.substitute(var, replacement)),
                        Box::new(body.substitute(var, replacement)),
                    )
                }
            }
            Expression::Exists(binder, body) | Expression::All(binder, body) => {
                let bn = var_name(binder);
                if bn == var {
                    self.clone()
                } else if replacement.free_variables().contains(&bn) {
                    let new_name = fresh_var(&bn, &replacement.free_variables());
                    let new_binder = Expression::Variable(new_name.clone(), binder_type(binder));
                    let new_body = body.substitute(&bn, &new_binder);
                    let quant = if matches!(self, Expression::Exists(_, _)) {
                        Expression::Exists
                    } else {
                        Expression::All
                    };
                    quant(Box::new(new_binder), Box::new(new_body.substitute(var, replacement)))
                } else {
                    let quant = if matches!(self, Expression::Exists(_, _)) {
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
            Expression::And(a, b) => Expression::And(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expression::Or(a, b) => Expression::Or(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expression::Not(expr) => Expression::Not(Box::new(expr.substitute(var, replacement))),
            Expression::If(a, b) => Expression::If(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expression::Iff(a, b) => Expression::Iff(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expression::Equality(a, b) => Expression::Equality(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
        }
    }

    pub fn simplify(&self) -> Expression {
        match self {
            Expression::Application(f, arg) => {
                let f_simpl = f.simplify();
                let arg_simpl = arg.simplify();
                match &f_simpl {
                    Expression::Lambda(binder, body) => {
                        let bn = var_name(binder);
                        body.substitute(&bn, &arg_simpl).simplify()
                    }
                    _ => Expression::Application(Box::new(f_simpl), Box::new(arg_simpl)),
                }
            }
            Expression::Lambda(binder, body) => {
                Expression::Lambda(binder.clone(), Box::new(body.simplify()))
            }
            Expression::Exists(binder, body) => {
                Expression::Exists(binder.clone(), Box::new(body.simplify()))
            }
            Expression::All(binder, body) => {
                Expression::All(binder.clone(), Box::new(body.simplify()))
            }
            Expression::And(a, b) => {
                Expression::And(Box::new(a.simplify()), Box::new(b.simplify()))
            }
            Expression::Or(a, b) => Expression::Or(Box::new(a.simplify()), Box::new(b.simplify())),
            Expression::Not(expr) => Expression::Not(Box::new(expr.simplify())),
            Expression::If(a, b) => Expression::If(Box::new(a.simplify()), Box::new(b.simplify())),
            Expression::Iff(a, b) => {
                Expression::Iff(Box::new(a.simplify()), Box::new(b.simplify()))
            }
            Expression::Equality(a, b) => {
                Expression::Equality(Box::new(a.simplify()), Box::new(b.simplify()))
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
