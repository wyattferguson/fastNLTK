//! Chunking — Rust-accelerated `RegexpChunkParser`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

// Tag pattern: compiles a <tag_pattern> to a Regex

/// Compile a single tag pattern like `<DT>`, `<JJ?>`, `<NN.*>` into a regex.
/// The pattern is applied to just the tag string.
fn compile_tag_pattern(pattern: &str) -> Result<Regex, String> {
    // Strip < and >, convert NLTK-style patterns to regex
    let inner = pattern.trim_start_matches('<').trim_end_matches('>').trim();

    // Convert NLTK pattern syntax to Rust regex:
    // - `?` → make preceding char optional in the regex sense
    // - `*` → zero or more of preceding char
    // - `.` → match any char (already works in regex)
    // Escape special regex chars except . * ?
    let mut re_str = String::with_capacity(inner.len() + 4);
    re_str.push('^');
    for ch in inner.chars() {
        match ch {
            '.' => re_str.push('.'),
            '*' => re_str.push('*'),
            '?' => re_str.push('?'),
            '|' => re_str.push('|'),
            // Escape all other special regex chars
            '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '\\' => {
                re_str.push('\\');
                re_str.push(ch);
            }
            _ => {
                // Check if we need to escape
                if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                    re_str.push(ch);
                } else {
                    re_str.push('\\');
                    re_str.push(ch);
                }
            }
        }
    }
    re_str.push('$');

    Regex::new(&re_str).map_err(|e| format!("Invalid tag pattern {inner}: {e}"))
}

/// Parse a rule pattern like `<DT><JJ?><NN.*>` into a sequence of tag regexes.
fn parse_tag_sequence(pattern: &str) -> Result<Vec<Regex>, String> {
    let mut regexes = Vec::new();
    let mut remaining = pattern.trim();

    while !remaining.is_empty() {
        let start = remaining
            .find('<')
            .ok_or_else(|| format!("Expected '<' in pattern, got: {remaining}"))?;
        let end = remaining
            .find('>')
            .ok_or_else(|| format!("Expected '>' closing tag in pattern: {remaining}"))?;
        let tag_pattern = &remaining[start..=end];
        let re = compile_tag_pattern(tag_pattern)?;
        regexes.push(re);
        remaining = &remaining[end + 1..];
    }

    Ok(regexes)
}

// ChunkRule: find tag sequences matching a pattern, mark as chunk

/// Apply a chunk rule to a sequence of tags. Modifies IOB tags in-place.
#[allow(clippy::needless_pass_by_ref_mut)]
fn apply_chunk_rule(tag_patterns: &[Regex], tags: &mut [&str], iob: &mut [&str]) {
    let num_tags = tags.len();
    let mut i = 0;
    while i < num_tags {
        // Try to match pattern starting at position i
        let mut matched = true;
        let mut j = 0;
        while j < tag_patterns.len() && i + j < num_tags {
            if !tag_patterns[j].is_match(tags[i + j]) {
                matched = false;
                break;
            }
            j += 1;
        }
        if matched && j == tag_patterns.len() {
            // Mark this span as IOB chunk
            iob[i] = "B-NP";
            for slot in &mut iob[i + 1..i + j] {
                *slot = "I-NP";
            }
            i += j;
        } else {
            i += 1;
        }
    }
}

// RegexpParser: parse grammar string, apply rules

#[pyclass(name = "RegexpParser", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct RegexpParser {
    /// Compiled chunk rules: each has a label + sequence of tag pattern regexes
    rules: Vec<(String, Vec<Regex>)>,
}

#[pymethods]
impl RegexpParser {
    #[new]
    #[pyo3(signature = (grammar))]
    fn new(grammar: &str) -> PyResult<Self> {
        let rules = Self::parse_grammar(grammar)?;
        Ok(Self { rules })
    }

    /// Parse a tagged sentence and return (word, `pos_tag`, `iob_tag`) triples.
    #[pyo3(signature = (tokens))]
    fn parse(&self, tokens: Vec<(String, String)>) -> Vec<(String, String, String)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Extract words and POS tags
        let pos_tags: Vec<&str> = tokens.iter().map(|(_, t)| t.as_str()).collect();
        let mut iob: Vec<&str> = vec!["O"; tokens.len()];

        // Apply each rule
        for (_label, tag_patterns) in &self.rules {
            apply_chunk_rule(tag_patterns, &mut pos_tags.clone(), &mut iob);
        }

        // Return (word, pos_tag, iob_tag) triples
        tokens.into_iter().zip(iob).map(|((w, t), i)| (w, t, i.to_string())).collect()
    }
}

impl RegexpParser {
    fn parse_grammar(grammar: &str) -> PyResult<Vec<(String, Vec<Regex>)>> {
        let mut rules = Vec::new();

        for line in grammar.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: LABEL: {pattern}
            let colon_pos = line.find(':').ok_or_else(|| {
                PyValueError::new_err(format!("Expected 'LABEL: pattern' format, got: {line}"))
            })?;

            let label = line[..colon_pos].trim().to_string();
            let rest = line[colon_pos + 1..].trim();

            // Find {pattern}
            let brace_start = rest
                .find('{')
                .ok_or_else(|| PyValueError::new_err(format!("Expected '{{' in rule: {line}")))?;
            let brace_end = rest
                .find('}')
                .ok_or_else(|| PyValueError::new_err(format!("Expected '}}' in rule: {line}")))?;

            let pattern_str = &rest[brace_start + 1..brace_end];
            let tag_patterns = parse_tag_sequence(pattern_str)
                .map_err(|e| PyValueError::new_err(format!("{e} in rule: {line}")))?;

            rules.push((label, tag_patterns));
        }

        Ok(rules)
    }
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RegexpParser>()?;
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_tag_pattern_dt() {
        let re = compile_tag_pattern("<DT>").unwrap();
        assert!(re.is_match("DT"));
        assert!(!re.is_match("NN"));
    }

    #[test]
    fn test_compile_tag_pattern_wildcard() {
        let re = compile_tag_pattern("<NN.*>").unwrap();
        assert!(re.is_match("NN"));
        assert!(re.is_match("NNS"));
        assert!(re.is_match("NNP"));
        assert!(re.is_match("NNPS"));
    }

    #[test]
    fn test_compile_tag_pattern_optional() {
        let re = compile_tag_pattern("<JJ?>").unwrap();
        assert!(re.is_match("JJ"));
        assert!(re.is_match("J"));
        assert!(!re.is_match("NN"));
    }

    #[test]
    fn test_compile_tag_pattern_dot_star() {
        let re = compile_tag_pattern("<.*>").unwrap();
        assert!(re.is_match("DT"));
        assert!(re.is_match("NN"));
        assert!(re.is_match("VBZ"));
    }

    #[test]
    fn test_parse_tag_sequence_single() {
        let regexes = parse_tag_sequence("<DT>").unwrap();
        assert_eq!(regexes.len(), 1);
    }

    #[test]
    fn test_parse_tag_sequence_multi() {
        let regexes = parse_tag_sequence("<DT><JJ?><NN.*>").unwrap();
        assert_eq!(regexes.len(), 3);
    }

    #[test]
    fn test_grammar_parsing() {
        let parser = RegexpParser::new("NP: {<DT><NN>}").unwrap();
        assert_eq!(parser.rules.len(), 1);
        assert_eq!(parser.rules[0].0, "NP");
    }

    #[test]
    fn test_grammar_multiline() {
        let parser = RegexpParser::new("NP: {<DT><NN>}\nVP: {<VB.*>}").unwrap();
        assert_eq!(parser.rules.len(), 2);
    }

    #[test]
    fn test_parse_simple() {
        let parser = RegexpParser::new("NP: {<DT><NN>}").unwrap();
        let tokens = vec![
            ("The".to_string(), "DT".to_string()),
            ("cat".to_string(), "NN".to_string()),
            ("sat".to_string(), "VBD".to_string()),
        ];
        let result = parser.parse(tokens);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].2, "B-NP");
        assert_eq!(result[1].2, "I-NP");
        assert_eq!(result[2].2, "O");
    }

    #[test]
    fn test_parse_no_match() {
        let parser = RegexpParser::new("NP: {<DT><NN>}").unwrap();
        let tokens = vec![("cat".to_string(), "NN".to_string())];
        let result = parser.parse(tokens);
        assert_eq!(result[0].2, "O");
    }

    #[test]
    fn test_empty_tokens() {
        let parser = RegexpParser::new("NP: {<DT><NN>}").unwrap();
        let result = parser.parse(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_invalid_grammar() {
        let result = RegexpParser::new("invalid grammar");
        assert!(result.is_err());
    }
}
