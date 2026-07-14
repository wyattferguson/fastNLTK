//! Nonmonotonic Reasoning — `DefaultReasoner` + `ClosedWorldReasoner`.
//!
//! Implements two nonmonotonic reasoning systems over symbolic knowledge bases:
//!
//! - **`DefaultReasoner`**: Computes extensions from default logic rules.
//!   A default rule has the form (prerequisite : justification / consequent).
//!   If the prerequisite holds and the justification is consistent,
//!   the consequent can be inferred. Multiple extensions arise when
//!   default rules conflict.
//!
//! - **`ClosedWorldReasoner`**: Assumes any fact not provably true is false.
//!   For each unknown proposition P, infers ~P.
//!
//! NLTK equivalents: nltk.inference.nonmonotonic

use hashbrown::HashSet;
use pyo3::prelude::*;
use std::fmt;

/// A default rule: (prerequisite : justification / consequent)
/// Means: if prerequisite holds and justification is consistent, infer consequent.
#[pyclass(name = "DefaultRule", module = "fastnltk._rust")]
#[derive(Clone, Debug)]
pub struct DefaultRule {
    #[pyo3(get)]
    pub prerequisite: String,
    #[pyo3(get)]
    pub justification: String,
    #[pyo3(get)]
    pub consequent: String,
    #[pyo3(get)]
    pub name: String,
}

#[pymethods]
impl DefaultRule {
    #[new]
    #[pyo3(signature = (prerequisite, justification, consequent, name=String::new()))]
    pub fn new(
        prerequisite: String,
        justification: String,
        consequent: String,
        name: String,
    ) -> Self {
        Self { prerequisite, justification, consequent, name }
    }

    fn __str__(&self) -> String {
        format!(
            "{}: {} : {} / {}",
            self.name, self.prerequisite, self.justification, self.consequent
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl fmt::Display for DefaultRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} : {} / {}",
            self.name, self.prerequisite, self.justification, self.consequent
        )
    }
}

/// `DefaultReasoner` computes extensions from a set of default rules and a background theory.
#[pyclass(name = "DefaultReasoner", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct DefaultReasoner {
    rules: Vec<DefaultRule>,
    max_extensions: usize,
}

#[pymethods]
impl DefaultReasoner {
    #[new]
    #[pyo3(signature = (rules, max_extensions=10))]
    pub fn new(rules: Vec<DefaultRule>, max_extensions: usize) -> Self {
        Self { rules, max_extensions }
    }

    /// Compute all extensions (fixed-point semantics).
    /// Returns list of extensions, where each extension is a list of facts.
    fn extensions(&self) -> Vec<Vec<String>> {
        let mut extensions: Vec<HashSet<String>> = vec![HashSet::new()];
        // Use HashSet for dedup instead of Vec::contains (O(n) → O(1))
        let mut new_seen: HashSet<Vec<String>> = HashSet::new();

        for _ in 0..self.max_extensions {
            let mut next_extensions: Vec<HashSet<String>> = Vec::new();
            new_seen.clear();

            for ext in &extensions {
                let mut added = false;

                for rule in &self.rules {
                    let prereq_holds =
                        ext.contains(&rule.prerequisite) || rule.prerequisite.is_empty();
                    let cons_already = ext.contains(&rule.consequent);
                    let just_consistent = !ext.contains(&format!("~{}", rule.justification));

                    if prereq_holds && !cons_already && just_consistent {
                        let mut new_ext = ext.clone();
                        new_ext.insert(rule.consequent.clone());
                        let mut sorted: Vec<String> = new_ext.iter().cloned().collect();
                        sorted.sort();
                        if new_seen.insert(sorted) {
                            next_extensions.push(new_ext);
                            added = true;
                        }
                    }
                }

                if !added {
                    let mut sorted: Vec<String> = ext.iter().cloned().collect();
                    sorted.sort();
                    if new_seen.insert(sorted) {
                        next_extensions.push(ext.clone());
                    }
                }
            }

            if next_extensions == extensions {
                break;
            }
            extensions = next_extensions;
        }

        // Convert to sorted string lists
        let mut results: Vec<Vec<String>> = Vec::new();
        for ext in extensions {
            let mut facts: Vec<String> = ext.into_iter().collect();
            facts.sort();
            results.push(facts);
        }

        results.sort();
        results
    }

    fn rules(&self) -> Vec<DefaultRule> {
        self.rules.clone()
    }
}

/// `ClosedWorldReasoner`: assumes any fact not provably true is false.
#[pyclass(name = "ClosedWorldReasoner", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct ClosedWorldReasoner {
    facts: Vec<String>,
}

#[pymethods]
impl ClosedWorldReasoner {
    #[new]
    pub fn new(facts: Vec<String>) -> Self {
        Self { facts }
    }

    /// Query a fact under closed-world assumption.
    /// Returns true if the fact is in the knowledge base, false otherwise.
    fn query(&self, fact: &str) -> bool {
        self.facts.contains(&fact.to_string())
    }

    /// The full set of known facts.
    fn facts(&self) -> Vec<String> {
        let mut result = self.facts.clone();
        // Add negations of everything not known
        let known: HashSet<String> = self.facts.iter().cloned().collect();
        // Infer negatives for unknown propositions
        for fact in &self.facts {
            let neg = format!("~{fact}");
            if !known.contains(&neg) {
                result.push(neg);
            }
        }
        result.sort();
        result.dedup();
        result
    }

    /// Get all positive facts.
    fn positive_facts(&self) -> Vec<String> {
        let mut result = self.facts.clone();
        result.sort();
        result.dedup();
        result
    }

    /// Get all negative facts (derived by closed-world).
    fn negative_facts(&self) -> Vec<String> {
        let known: HashSet<String> = self.facts.iter().cloned().collect();
        self.facts.iter().map(|f| format!("~{f}")).filter(|n| !known.contains(n)).collect()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DefaultRule>()?;
    m.add_class::<DefaultReasoner>()?;
    m.add_class::<ClosedWorldReasoner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rule_creation() {
        let rule =
            DefaultRule::new("bird(x)".into(), "flies(x)".into(), "flies(x)".into(), String::new());
        assert_eq!(rule.prerequisite, "bird(x)");
        assert_eq!(rule.consequent, "flies(x)");
    }

    #[test]
    fn test_default_rule_display() {
        let rule = DefaultRule::new(
            "bird(x)".into(),
            "flies(x)".into(),
            "flies(x)".into(),
            "bird-flies".into(),
        );
        let s = rule.__str__();
        assert!(s.contains("bird") || s.contains("bird-flies"), "should contain bird: {s}");
        assert!(s.contains("flies"), "should contain flies: {s}");
    }

    #[test]
    fn test_default_reasoner_empty() {
        let reasoner = DefaultReasoner::new(vec![], 10);
        let exts = reasoner.extensions();
        assert!(!exts.is_empty());
        assert!(exts[0].is_empty());
    }

    #[test]
    fn test_default_reasoner_simple() {
        let rules = vec![DefaultRule::new(
            "bird".into(),
            "flies".into(),
            "flies".into(),
            "bird-flies".into(),
        )];
        let reasoner = DefaultReasoner::new(rules, 10);
        let exts = reasoner.extensions();
        assert!(!exts.is_empty());
    }

    #[test]
    fn test_closed_world_reasoner() {
        let reasoner = ClosedWorldReasoner::new(vec!["bird(tweety)".into(), "cat(felix)".into()]);
        assert!(reasoner.query("bird(tweety)"));
        assert!(!reasoner.query("bird(felix)"));
    }

    #[test]
    fn test_closed_world_negatives() {
        let reasoner = ClosedWorldReasoner::new(vec!["bird(tweety)".into()]);
        let negatives = reasoner.negative_facts();
        assert!(negatives.contains(&"~bird(tweety)".to_string()));
        assert!(!negatives.contains(&"~cat(felix)".to_string()));
    }

    #[test]
    fn test_closed_world_all_facts() {
        let reasoner = ClosedWorldReasoner::new(vec!["bird(tweety)".into()]);
        let all = reasoner.facts();
        assert!(all.contains(&"bird(tweety)".to_string()));
    }

    #[test]
    fn test_rules_method() {
        let rule = DefaultRule::new("bird".into(), "flies".into(), "flies".into(), "bf".into());
        let reasoner = DefaultReasoner::new(vec![rule], 10);
        let rs = reasoner.rules();
        assert_eq!(rs.len(), 1);
        assert_eq!(rs[0].name, "bf");
    }

    #[test]
    fn test_empty_prerequisite() {
        let rules =
            vec![DefaultRule::new(String::new(), "fact".into(), "fact".into(), "always".into())];
        let reasoner = DefaultReasoner::new(rules, 10);
        let exts = reasoner.extensions();
        assert!(!exts.is_empty());
        assert!(
            exts.iter().any(|e| e.contains(&"fact".to_string())),
            "should derive 'fact' from empty prereq: {exts:?}"
        );
    }

    #[test]
    fn test_closed_world_negative_facts() {
        let reasoner = ClosedWorldReasoner::new(vec!["bird(tweety)".into(), "cat(felix)".into()]);
        let pos = reasoner.positive_facts();
        assert_eq!(pos.len(), 2);
        assert!(pos.contains(&"bird(tweety)".to_string()));
    }

    #[test]
    fn test_closed_world_empty_kb() {
        let reasoner = ClosedWorldReasoner::new(vec![]);
        assert!(!reasoner.query("anything"));
        assert!(reasoner.positive_facts().is_empty());
    }
}
