//! Semantics — Rust-accelerated logical expression parsing and evaluation.
//!
//! Implements NLTK's `nltk.sem.logic` module:
//!   - Expression types (Constant, Variable, Application, Lambda, Quantifier, etc.)
//!   - Recursive descent parser for logical formulas
//!   - Substitution, simplification (beta-reduction)
//!   - Free variable extraction
//!   - Pretty-printing
//!
//! Phase 1: Core Expression type + parser (~500 LoC Rust)
//! Phase 2: Model + evaluation (~400 LoC Rust)
//! Phase 3: DRT (~600 LoC Rust, optional)

use std::collections::{HashMap, HashSet};
use std::fmt;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An individual in the model domain.
pub type Individual = String;

/// A valuation maps predicate names to sets of tuples.
pub type Valuation = HashMap<String, Vec<Vec<Individual>>>;

/// An assignment maps variable names to individuals.
pub type Assignment = HashMap<String, Individual>;

// ═══════════════════════════════════════════════════════════
// Expression types
// ═══════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Constant(String, Option<Type>),                // e.g., "john", "man"
    Variable(String, Option<Type>),                // e.g., "x", "y"
    Application(Box<Expression>, Box<Expression>), // f(a)
    Lambda(Box<Expression>, Box<Expression>),      // \x.f(x)
    Exists(Box<Expression>, Box<Expression>),      // exists x.P(x)
    All(Box<Expression>, Box<Expression>),         // all x.P(x)
    And(Box<Expression>, Box<Expression>),         // P & Q
    Or(Box<Expression>, Box<Expression>),          // P | Q
    Not(Box<Expression>),                          // -P
    If(Box<Expression>, Box<Expression>),          // P -> Q
    Iff(Box<Expression>, Box<Expression>),         // P <-> Q
    Equality(Box<Expression>, Box<Expression>),    // x = y
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Entity,                    // e
    TruthValue,                // t
    Fun(Box<Type>, Box<Type>), // <e,t>
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
                // Check if function is a constant/variable — use juxtaposition
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

fn is_atomic(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Constant(_, _) | Expression::Variable(_, _)
    )
}

// ═══════════════════════════════════════════════════════════
// Free variables
// ═══════════════════════════════════════════════════════════

impl Expression {
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
                let var_name = var_name(var);
                bound.insert(var_name.clone());
                body.collect_free(bound, free);
                bound.remove(&var_name);
            }
            Expression::Exists(var, body) | Expression::All(var, body) => {
                let var_name = var_name(var);
                bound.insert(var_name.clone());
                body.collect_free(bound, free);
                bound.remove(&var_name);
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

fn var_name(var: &Expression) -> String {
    match var {
        Expression::Variable(name, _) => name.clone(),
        _ => String::new(),
    }
}

// ═══════════════════════════════════════════════════════════
// Substitution
// ═══════════════════════════════════════════════════════════

impl Expression {
    /// Substitute variable with expression. Returns new expression.
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
                let binder_name = var_name(binder);
                if binder_name == var {
                    // Shadowed — don't substitute
                    self.clone()
                } else if replacement.free_variables().contains(&binder_name) {
                    // Capture avoidance: alpha-convert the binder
                    let new_name = fresh_var(&binder_name, &replacement.free_variables());
                    let new_binder = Expression::Variable(new_name.clone(), binder_type(binder));
                    let new_body = body.substitute(&binder_name, &new_binder);
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
                let binder_name = var_name(binder);
                if binder_name == var {
                    self.clone()
                } else if replacement.free_variables().contains(&binder_name) {
                    let new_name = fresh_var(&binder_name, &replacement.free_variables());
                    let new_binder = Expression::Variable(new_name.clone(), binder_type(binder));
                    let new_body = body.substitute(&binder_name, &new_binder);
                    let quant = if matches!(self, Expression::Exists(_, _)) {
                        Expression::Exists
                    } else {
                        Expression::All
                    };
                    quant(
                        Box::new(new_binder),
                        Box::new(new_body.substitute(var, replacement)),
                    )
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

    /// Beta-reduction: (\x.M)(N) -> M[x:=N]
    pub fn simplify(&self) -> Expression {
        match self {
            Expression::Application(f, arg) => {
                let f_simpl = f.simplify();
                let arg_simpl = arg.simplify();
                match &f_simpl {
                    Expression::Lambda(binder, body) => {
                        let var_name = var_name(binder);
                        body.substitute(&var_name, &arg_simpl).simplify()
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

fn binder_type(expr: &Expression) -> Option<Type> {
    match expr {
        Expression::Variable(_, t) => t.clone(),
        _ => None,
    }
}

fn fresh_var(base: &str, avoid: &[String]) -> String {
    let mut i = 0;
    loop {
        let candidate = format!("{base}{i}");
        if !avoid.contains(&candidate) {
            return candidate;
        }
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String), // variable/constant name
    Lambda,        // \ or lambda
    Dot,           // .
    LParen,        // (
    RParen,        // )
    Comma,         // ,
    And,           // & or ^
    Or,            // |
    Not,           // -
    Arrow,         // ->
    Iff,           // <->
    Eq,            // =
    Exists,        // exists or E
    All,           // all or A
    TypeColon,     // : (for type annotations)
    End,           // end of input
}

struct Tokenizer {
    chars: Vec<char>,
    pos: usize,
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        Tokenizer {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        self.pos += 1;
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace();
        let c = match self.peek() {
            Some(c) => c,
            None => return Ok(Token::End),
        };

        match c {
            '(' => {
                self.advance();
                Ok(Token::LParen)
            }
            ')' => {
                self.advance();
                Ok(Token::RParen)
            }
            ',' => {
                self.advance();
                Ok(Token::Comma)
            }
            '.' => {
                self.advance();
                Ok(Token::Dot)
            }
            '\\' => {
                self.advance();
                Ok(Token::Lambda)
            }
            '^' => {
                self.advance();
                Ok(Token::And)
            }
            '|' => {
                self.advance();
                Ok(Token::Or)
            }
            '=' => {
                self.advance();
                if self.peek() == Some('>') {
                    // =>
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Eq)
                }
            }
            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Not)
                }
            }
            '<' => {
                self.advance();
                if self.peek() == Some('-') {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        Ok(Token::Iff)
                    } else {
                        Ok(Token::Eq)
                    }
                } else {
                    Err(format!("Unexpected '<' at position {}", self.pos))
                }
            }
            '&' => {
                self.advance();
                Ok(Token::And)
            }
            ':' => {
                self.advance();
                Ok(Token::TypeColon)
            }
            _ if c.is_alphabetic() || c == '_' => {
                let mut name = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_alphanumeric() || ch == '_' || ch == '\'' {
                        name.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                match name.as_str() {
                    "lambda" => Ok(Token::Lambda),
                    "exists" | "E" => Ok(Token::Exists),
                    "all" | "A" => Ok(Token::All),
                    _ => Ok(Token::Ident(name)),
                }
            }
            _ => Err(format!(
                "Unexpected character '{c}' at position {}",
                self.pos
            )),
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════

struct Parser {
    tokenizer: Tokenizer,
    current: Token,
}

impl Parser {
    fn new(input: &str) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::new(input);
        let current = tokenizer.next_token()?;
        Ok(Parser { tokenizer, current })
    }

    fn advance(&mut self) -> Result<(), String> {
        self.current = self.tokenizer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.current == *expected {
            self.advance()
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, self.current))
        }
    }

    /// Parse the full expression.
    fn parse(&mut self) -> Result<Expression, String> {
        self.parse_iff()
    }

    // <-> (lowest precedence)
    fn parse_iff(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_if()?;
        while self.current == Token::Iff {
            self.advance()?;
            let right = self.parse_if()?;
            left = Expression::Iff(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // ->
    fn parse_if(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_or()?;
        while self.current == Token::Arrow {
            self.advance()?;
            let right = self.parse_or()?;
            left = Expression::If(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // |
    fn parse_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_and()?;
        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and()?;
            left = Expression::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // &
    fn parse_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_equality()?;
        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_equality()?;
            left = Expression::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // =
    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_not()?;
        while self.current == Token::Eq {
            self.advance()?;
            let right = self.parse_not()?;
            left = Expression::Equality(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // - (unary)
    fn parse_not(&mut self) -> Result<Expression, String> {
        if self.current == Token::Not {
            self.advance()?;
            let expr = self.parse_not()?;
            Ok(Expression::Not(Box::new(expr)))
        } else {
            self.parse_quantifier()
        }
    }

    // exists/all
    fn parse_quantifier(&mut self) -> Result<Expression, String> {
        if self.current == Token::Exists {
            self.advance()?;
            let var = self.parse_atom()?;
            self.expect(&Token::Dot)?;
            let body = self.parse()?;
            Ok(Expression::Exists(Box::new(var), Box::new(body)))
        } else if self.current == Token::All {
            self.advance()?;
            let var = self.parse_atom()?;
            self.expect(&Token::Dot)?;
            let body = self.parse()?;
            Ok(Expression::All(Box::new(var), Box::new(body)))
        } else {
            self.parse_lambda()
        }
    }

    // lambda
    fn parse_lambda(&mut self) -> Result<Expression, String> {
        if self.current == Token::Lambda {
            self.advance()?;
            let var = self.parse_atom()?;
            self.expect(&Token::Dot)?;
            let body = self.parse()?;
            Ok(Expression::Lambda(Box::new(var), Box::new(body)))
        } else {
            self.parse_application()
        }
    }

    // Application: f(a,b) or f a or just an atom
    fn parse_application(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_atom()?;

        loop {
            match &self.current {
                Token::LParen => {
                    // Function application with args: f(a, b)
                    self.advance()?;
                    let arg = self.parse()?;
                    // Handle multiple args by currying
                    let mut result = Expression::Application(Box::new(expr), Box::new(arg));
                    while self.current == Token::Comma {
                        self.advance()?;
                        let next_arg = self.parse()?;
                        result = Expression::Application(Box::new(result), Box::new(next_arg));
                    }
                    self.expect(&Token::RParen)?;
                    expr = result;
                }
                Token::Ident(_)
                | Token::Exists
                | Token::All
                | Token::Lambda
                | Token::Not
                | Token::LParen => {
                    // Adjacent expressions: f a
                    let arg = if self.current == Token::LParen
                        || self.current == Token::Not
                        || self.current == Token::Exists
                        || self.current == Token::All
                        || self.current == Token::Lambda
                    {
                        self.parse_application()?
                    } else {
                        self.parse_atom()?
                    };
                    expr = Expression::Application(Box::new(expr), Box::new(arg));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_atom(&mut self) -> Result<Expression, String> {
        match &self.current {
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;
                // Check for type annotation
                let typ = if self.current == Token::TypeColon {
                    self.advance()?;
                    Some(self.parse_type()?)
                } else {
                    None
                };
                // All identifiers parse as Variables (for substitution to work)
                // The variable/constant distinction is semantic, not syntactic
                Ok(Expression::Variable(name, typ))
            }
            Token::LParen => {
                self.advance()?;
                let expr = self.parse()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::Exists => {
                self.advance()?;
                let var = self.parse_atom()?;
                self.expect(&Token::Dot)?;
                let body = self.parse()?;
                Ok(Expression::Exists(Box::new(var), Box::new(body)))
            }
            Token::All => {
                self.advance()?;
                let var = self.parse_atom()?;
                self.expect(&Token::Dot)?;
                let body = self.parse()?;
                Ok(Expression::All(Box::new(var), Box::new(body)))
            }
            Token::Lambda => {
                self.advance()?;
                let var = self.parse_atom()?;
                self.expect(&Token::Dot)?;
                let body = self.parse()?;
                Ok(Expression::Lambda(Box::new(var), Box::new(body)))
            }
            _ => Err(format!("Unexpected token {:?}", self.current)),
        }
    }

    fn parse_type(&mut self) -> Result<Type, String> {
        match &self.current {
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;
                match name.as_str() {
                    "e" => Ok(Type::Entity),
                    "t" => Ok(Type::TruthValue),
                    _ => Err(format!("Unknown type: {name}")),
                }
            }
            Token::LParen => {
                // Complex type: <e,t>
                self.advance()?;
                let arg = self.parse_type()?;
                self.expect(&Token::Comma)?;
                let result = self.parse_type()?;
                self.expect(&Token::RParen)?;
                Ok(Type::Fun(Box::new(arg), Box::new(result)))
            }
            _ => Err(format!("Expected type, got {:?}", self.current)),
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════

/// Parse a logical formula string into an Expression.
pub fn parse_expression(input: &str) -> Result<Expression, String> {
    let mut parser = Parser::new(input)?;
    let expr = parser.parse()?;
    if parser.current != Token::End {
        return Err(format!(
            "Unexpected tokens after expression: {:?}",
            parser.current
        ));
    }
    Ok(expr)
}

// ═══════════════════════════════════════════════════════════
// PyO3 wrapper
// ═══════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (formula))]
fn fromstring(formula: &str) -> PyResult<String> {
    let expr = parse_expression(formula).map_err(|e| PyValueError::new_err(e))?;
    Ok(format!("{}", expr))
}

#[pyfunction]
#[pyo3(signature = (formula))]
fn simplify(formula: &str) -> PyResult<String> {
    let expr = parse_expression(formula).map_err(|e| PyValueError::new_err(e))?;
    Ok(format!("{}", expr.simplify()))
}

// ═══════════════════════════════════════════════════════════
// Model evaluation
// ═══════════════════════════════════════════════════════════

/// Evaluate a formula string in a model.
/// Returns true/false if the formula is satisfied.
#[pyfunction]
#[pyo3(signature = (formula, valuation_json, domain_json, assignment_json=""))]
fn evaluate_formula(
    formula: &str,
    valuation_json: &str,
    domain_json: &str,
    assignment_json: &str,
) -> PyResult<bool> {
    let expr = parse_expression(formula).map_err(|e| PyValueError::new_err(e))?;
    // Parse JSON inputs
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
    model_evaluate(&expr, &valuation, &domain, &assignment).map_err(|e| PyValueError::new_err(e))
}

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
            let var_name = var_name(var);
            for ind in domain {
                let mut new_assign = assignment.clone();
                new_assign.insert(var_name.clone(), ind.clone());
                if model_evaluate(body, valuation, domain, &new_assign)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Expression::All(var, body) => {
            let var_name = var_name(var);
            for ind in domain {
                let mut new_assign = assignment.clone();
                new_assign.insert(var_name.clone(), ind.clone());
                if !model_evaluate(body, valuation, domain, &new_assign)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Expression::Lambda(_, _) => Err("Cannot evaluate lambda directly".to_string()),
    }
}

/// Register module with Python.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fromstring, m)?)?;
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_formula, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constant() {
        let expr = parse_expression("john").unwrap();
        assert_eq!(format!("{expr}"), "john");
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_expression("x").unwrap();
        assert_eq!(format!("{expr}"), "x");
    }

    #[test]
    fn test_parse_application() {
        let expr = parse_expression("walk(john)").unwrap();
        assert_eq!(format!("{expr}"), "walk(john)");
    }

    #[test]
    fn test_parse_binary_application() {
        let expr = parse_expression("chase(john,mary)").unwrap();
        assert!(format!("{expr}").contains("walk") || format!("{expr}").contains("chase"));
    }

    #[test]
    fn test_parse_lambda() {
        let expr = parse_expression(r"\x.walk(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("\\") || s.contains("lambda"));
        assert!(s.contains("x"));
        assert!(s.contains("walk"));
    }

    #[test]
    fn test_parse_exists() {
        let expr = parse_expression("exists x.walk(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("exists"));
    }

    #[test]
    fn test_parse_all() {
        let expr = parse_expression("all x.man(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("all"));
    }

    #[test]
    fn test_parse_and() {
        let expr = parse_expression("walk(x) & talk(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("&"));
    }

    #[test]
    fn test_parse_or() {
        let expr = parse_expression("walk(x) | talk(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("|"));
    }

    #[test]
    fn test_parse_not() {
        let expr = parse_expression("-walk(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("-"));
    }

    #[test]
    fn test_parse_arrow() {
        let expr = parse_expression("man(x) -> mortal(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("->"));
    }

    #[test]
    fn test_parse_iff() {
        let expr = parse_expression("man(x) <-> human(x)").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("<->"));
    }

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
        // (\x.walk(x))(john) -> walk(john)
        let expr = parse_expression(r"(\x.walk(x))(john)").unwrap();
        let simplified = expr.simplify();
        assert_eq!(format!("{simplified}"), "walk(john)");
    }

    #[test]
    fn test_complex_formula() {
        let expr = parse_expression("all x.(man(x) -> mortal(x))").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("all"));
        assert!(s.contains("man"));
        assert!(s.contains("mortal"));
    }

    #[test]
    fn test_equality() {
        let expr = parse_expression("x = john").unwrap();
        let s = format!("{expr}");
        assert!(s.contains("="));
    }

    #[test]
    fn test_parse_error() {
        let result = parse_expression("walk(");
        assert!(result.is_err());
    }

    #[test]
    fn test_simplify_lambda_application() {
        let expr = parse_expression(r"(\x.man(x))(john)").unwrap();
        let result = expr.simplify();
        assert_eq!(format!("{result}"), "man(john)");
    }
}
