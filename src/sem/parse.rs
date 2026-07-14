//! Recursive descent parser for logical formulas.
//!
//! Tokenizes and parses expressions in NLTK's logic syntax:
//! - Variables, constants, applications, lambda, quantifiers
//! - Connectives: &, |, -, ->, <->
//! - Equality: =

use crate::sem::expression::{Expression, Type};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Ident(String),
    Lambda,
    Dot,
    LParen,
    RParen,
    Comma,
    And,
    Or,
    Not,
    Arrow,
    Iff,
    Eq,
    Exists,
    All,
    TypeColon,
    End,
}

struct Tokenizer {
    chars: Vec<char>,
    pos: usize,
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        Self { chars: input.chars().collect(), pos: 0 }
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
            '(' => { self.advance(); Ok(Token::LParen) }
            ')' => { self.advance(); Ok(Token::RParen) }
            ',' => { self.advance(); Ok(Token::Comma) }
            '.' => { self.advance(); Ok(Token::Dot) }
            '\\' => { self.advance(); Ok(Token::Lambda) }
            '^' => { self.advance(); Ok(Token::And) }
            '|' => { self.advance(); Ok(Token::Or) }
            '=' => {
                self.advance();
                if self.peek() == Some('>') {
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
            '&' => { self.advance(); Ok(Token::And) }
            ':' => { self.advance(); Ok(Token::TypeColon) }
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
            _ => Err(format!("Unexpected character '{c}' at position {}", self.pos)),
        }
    }
}

struct Parser {
    tokenizer: Tokenizer,
    current: Token,
}

impl Parser {
    fn new(input: &str) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::new(input);
        let current = tokenizer.next_token()?;
        Ok(Self { tokenizer, current })
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

    fn parse(&mut self) -> Result<Expression, String> {
        self.parse_iff()
    }

    fn parse_iff(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_if()?;
        while self.current == Token::Iff {
            self.advance()?;
            let right = self.parse_if()?;
            left = Expression::Iff(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_if(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_or()?;
        while self.current == Token::Arrow {
            self.advance()?;
            let right = self.parse_or()?;
            left = Expression::If(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_and()?;
        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and()?;
            left = Expression::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_equality()?;
        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_equality()?;
            left = Expression::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_not()?;
        while self.current == Token::Eq {
            self.advance()?;
            let right = self.parse_not()?;
            left = Expression::Equality(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expression, String> {
        if self.current == Token::Not {
            self.advance()?;
            let expr = self.parse_not()?;
            Ok(Expression::Not(Box::new(expr)))
        } else {
            self.parse_quantifier()
        }
    }

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

    fn parse_application(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_atom()?;

        loop {
            match &self.current {
                Token::LParen => {
                    self.advance()?;
                    let arg = self.parse()?;
                    let mut result = Expression::Application(Box::new(expr), Box::new(arg));
                    while self.current == Token::Comma {
                        self.advance()?;
                        let next_arg = self.parse()?;
                        result = Expression::Application(Box::new(result), Box::new(next_arg));
                    }
                    self.expect(&Token::RParen)?;
                    expr = result;
                }
                Token::Ident(_) | Token::Exists | Token::All | Token::Lambda | Token::Not => {
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
                let typ = if self.current == Token::TypeColon {
                    self.advance()?;
                    Some(self.parse_type()?)
                } else {
                    None
                };
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

/// Parse a logical formula string into an Expression.
pub fn parse_expression(input: &str) -> Result<Expression, String> {
    let mut parser = Parser::new(input)?;
    let expr = parser.parse()?;
    if parser.current != Token::End {
        return Err(format!("Unexpected tokens after expression: {:?}", parser.current));
    }
    Ok(expr)
}

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
