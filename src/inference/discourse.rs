//! Inference Discourse — DRT discourse thread processing.
//!
//! Builds on `src/drt.rs` for DRS representation and provides
//! discourse-level reading comprehension over threaded DRSs.
//!
//! Core functionality:
//!   - `DiscourseThread`: manages a sequence of DRSs
//!   - DRS merging across the thread (union of referents + conditions)
//!   - FOL conversion of the merged discourse
//!   - Yes/no question answering over a model
//!
//! NLTK equivalent: nltk.inference.discourse

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::drt::DRS;
use crate::sem::{self, model_evaluate, Assignment, Expression, Individual, Valuation};

/// A discourse representation thread: sequence of DRSs with referent tracking.
#[pyclass(name = "DiscourseThread", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct DiscourseThread {
    drss: Vec<DRS>,
    /// All referents mentioned across the discourse
    all_referents: Vec<String>,
}

#[pymethods]
impl DiscourseThread {
    #[new]
    fn new() -> Self {
        DiscourseThread {
            drss: Vec::new(),
            all_referents: Vec::new(),
        }
    }

    /// Add a DRS from bracket notation.
    fn add_drs(&mut self, drs_string: &str) -> PyResult<()> {
        let drs = DRS::from_string(drs_string).map_err(|e| PyValueError::new_err(e))?;
        // Track new referents
        for ref_ in &drs.universe {
            if !self.all_referents.contains(ref_) {
                self.all_referents.push(ref_.clone());
            }
        }
        self.drss.push(drs);
        Ok(())
    }

    /// Add a DRS from the existing DRS type.
    fn add(&mut self, drs_string: &str) -> PyResult<()> {
        self.add_drs(drs_string)
    }

    /// Get all DRS strings.
    fn get_drss(&self) -> Vec<String> {
        self.drss.iter().map(|d| format!("{d}")).collect()
    }

    /// Number of DRSs in the thread.
    fn __len__(&self) -> usize {
        self.drss.len()
    }

    /// Merge all DRSs in the thread into a single DRS.
    fn merge(&self) -> String {
        let mut universe = Vec::new();
        let mut conditions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for drs in &self.drss {
            for ref_ in &drs.universe {
                if seen.insert(ref_.clone()) {
                    universe.push(ref_.clone());
                }
            }
            for cond in &drs.conditions {
                conditions.push(cond.clone());
            }
        }

        let merged = DRS {
            universe,
            conditions,
        };
        format!("{merged}")
    }

    /// Convert entire discourse to FOL.
    fn to_fol(&self) -> String {
        let merged_drs = self.merge_drs();
        let fol = merged_drs.to_fol();
        format!("{fol}")
    }

    /// Answer a yes/no question about the discourse.
    /// Returns "true", "false", or "unknown".
    fn answer_question(
        &self,
        question_drs: &str,
        valuation_json: &str,
        domain_json: &str,
    ) -> PyResult<String> {
        let q = DRS::from_string(question_drs).map_err(|e| PyValueError::new_err(e))?;

        // Merge discourse into single DRS
        let discourse = self.merge_drs();

        // Combine discourse & question: discourse & question
        let mut combined_universe = discourse.universe.clone();
        let mut seen: std::collections::HashSet<String> =
            discourse.universe.iter().cloned().collect();
        for ref_ in &q.universe {
            if seen.insert(ref_.clone()) {
                combined_universe.push(ref_.clone());
            }
        }

        let mut conditions = discourse.conditions.clone();
        conditions.extend(q.conditions.clone());

        let combined = DRS {
            universe: combined_universe,
            conditions,
        };

        let valuation: Valuation = serde_json::from_str(valuation_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid valuation JSON: {e}")))?;
        let domain: Vec<Individual> = serde_json::from_str(domain_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid domain JSON: {e}")))?;

        let assignment = Assignment::new();
        match combined.evaluate(&valuation, &domain, &assignment) {
            Ok(true) => Ok("true".to_string()),
            Ok(false) => Ok("false".to_string()),
            Err(_) => Ok("unknown".to_string()),
        }
    }
}

impl DiscourseThread {
    fn merge_drs(&self) -> DRS {
        let mut universe = Vec::new();
        let mut conditions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for drs in &self.drss {
            for ref_ in &drs.universe {
                if seen.insert(ref_.clone()) {
                    universe.push(ref_.clone());
                }
            }
            for cond in &drs.conditions {
                conditions.push(cond.clone());
            }
        }

        DRS {
            universe,
            conditions,
        }
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiscourseThread>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drt::DRS;

    #[test]
    fn test_empty_thread() {
        let thread = DiscourseThread::new();
        assert_eq!(thread.__len__(), 0);
        assert!(thread.get_drss().is_empty());
    }

    #[test]
    fn test_add_drs() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x)])").unwrap();
        assert_eq!(thread.__len__(), 1);
    }

    #[test]
    fn test_merge() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x)])").unwrap();
        thread.add_drs("([y],[cat(y)])").unwrap();
        let merged = thread.merge();
        assert!(merged.contains("dog") && merged.contains("cat"));
    }

    #[test]
    fn test_to_fol() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x)])").unwrap();
        let fol = thread.to_fol();
        assert!(fol.contains("exists") || fol.contains("dog"));
    }

    #[test]
    fn test_answer_question_true() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x)])").unwrap();
        thread.add_drs("([y],[cat(y)])").unwrap();

        let mut valuation: Valuation = HashMap::new();
        valuation.insert("dog".to_string(), vec![vec!["fido".to_string()]]);
        valuation.insert("cat".to_string(), vec![vec!["felix".to_string()]]);
        let domain = vec!["fido".to_string(), "felix".to_string()];
        let val_json = serde_json::to_string(&valuation).unwrap();
        let dom_json = serde_json::to_string(&domain).unwrap();

        let answer = thread
            .answer_question("([x],[dog(x)])", &val_json, &dom_json)
            .unwrap();
        assert_eq!(answer, "true");
    }

    #[test]
    fn test_answer_question_false() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[cat(x)])").unwrap();
        let mut valuation: Valuation = HashMap::new();
        valuation.insert("cat".to_string(), vec![vec!["felix".to_string()]]);
        let domain = vec!["felix".to_string()];
        let val_json = serde_json::to_string(&valuation).unwrap();
        let dom_json = serde_json::to_string(&domain).unwrap();

        let answer = thread
            .answer_question("([x],[dog(x)])", &val_json, &dom_json)
            .unwrap();
        assert_eq!(answer, "false");
    }

    #[test]
    fn test_multiple_referents() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x)])").unwrap();
        thread.add_drs("([x, y],[dog(x), bone(y)])").unwrap();
        let merged = thread.merge();
        // The DRS should merge referents
        assert!(merged.contains("dog") && merged.contains("bone"));
    }

    #[test]
    fn test_add_method_alias() {
        let mut thread = DiscourseThread::new();
        thread.add("([x],[dog(x)])").unwrap();
        assert_eq!(thread.__len__(), 1);
    }

    #[test]
    fn test_question_unknown() {
        let thread = DiscourseThread::new();
        let mut valuation: Valuation = HashMap::new();
        valuation.insert("dog".to_string(), vec![vec!["fido".to_string()]]);
        let domain = vec!["fido".to_string()];
        let val_json = serde_json::to_string(&valuation).unwrap();
        let dom_json = serde_json::to_string(&domain).unwrap();
        // Empty discourse + invalid question DRS -> should error
        let result = thread.answer_question("([x],[dog(x)])", &val_json, &dom_json);
        assert!(result.is_ok(), "Should handle empty discourse");
    }

    #[test]
    fn test_negation_in_drs() {
        let mut thread = DiscourseThread::new();
        thread.add_drs("([x],[dog(x), -([y],[cat(y)])])").unwrap();
        assert_eq!(thread.__len__(), 1);
        let fol = thread.to_fol();
        // FOL should contain both dog and negation
        assert!(fol.contains("dog"), "should contain dog: {fol}");
        assert!(
            fol.contains("exists") || fol.contains("all"),
            "should have quantifier: {fol}"
        );
    }
}
