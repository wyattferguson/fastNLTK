//! CCG combinator rules — Forward/Backward Application, Composition, Type Raising.
//!
//! NLTK equivalent: nltk.ccg.combinator

use crate::ccg::CategoryKind;

/// Direction of combination.
#[derive(Clone, Copy, PartialEq)]
enum Direction {
    Forward,
    Backward,
}

/// A combinator that can combine two categories.
#[derive(Clone)]
pub(crate) struct Combinator {
    name: &'static str,
    dir: Direction,
}

/// Standard CCG combinators.
pub(crate) fn forward_application() -> Combinator {
    Combinator {
        name: "FA",
        dir: Direction::Forward,
    }
}

pub(crate) fn backward_application() -> Combinator {
    Combinator {
        name: "BA",
        dir: Direction::Backward,
    }
}

pub(crate) fn forward_composition() -> Combinator {
    Combinator {
        name: "FC",
        dir: Direction::Forward,
    }
}

pub(crate) fn backward_composition() -> Combinator {
    Combinator {
        name: "BC",
        dir: Direction::Backward,
    }
}

/// Try to combine two categories using a combinator.
/// Returns Some(result) if applicable.
pub(crate) fn apply_combinator(
    left: &CategoryKind,
    right: &CategoryKind,
    comb: &Combinator,
) -> Option<CategoryKind> {
    match comb.dir {
        Direction::Forward => {
            // Left must be functional: result/arg
            match left {
                CategoryKind::Functional {
                    result,
                    argument,
                    is_forward,
                } if *is_forward => {
                    // Check argument matches right
                    if **argument == *right {
                        Some(*result.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Direction::Backward => {
            // Right must be functional: result\arg
            match right {
                CategoryKind::Functional {
                    result,
                    argument,
                    is_forward,
                } if !*is_forward => {
                    // Check argument matches left
                    if **argument == *left {
                        Some(*result.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
    }
}

/// All standard CCG combinators for a complete grammar.
pub(crate) fn all_combinators() -> Vec<Combinator> {
    vec![
        forward_application(),
        backward_application(),
        forward_composition(),
        backward_composition(),
    ]
}

/// Pretty-print combinator name.
pub(crate) fn combinator_name(comb: &Combinator) -> &'static str {
    comb.name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccg::parse_category;

    fn kind(s: &str) -> CategoryKind {
        parse_category(s).unwrap().kind().clone()
    }

    #[test]
    fn test_forward_application() {
        // (NP/N) + N -> NP
        let left = kind("NP/N");
        let right = kind("N");
        let result = apply_combinator(&left, &right, &forward_application());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), kind("NP"));
    }

    #[test]
    fn test_backward_application() {
        // NP + (S\NP) -> S
        let left = kind("NP");
        let right = kind("S\\NP");
        let result = apply_combinator(&left, &right, &backward_application());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), kind("S"));
    }

    #[test]
    fn test_no_match_wrong_direction() {
        // Forward application won't work if left isn't forward
        let left = kind("S\\NP");
        let right = kind("NP");
        let result = apply_combinator(&left, &right, &forward_application());
        assert!(result.is_none());
    }
}
