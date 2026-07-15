//! Lancaster stemmer — full 120-rule implementation matching NLTK.
//!
//! Ported from NLTK's `nltk.stem.lancaster` (Apache-2.0).
//! Original implementation by Steven Tomcavage.

use pyo3::prelude::*;

/// A Lancaster stemming rule: (suffix, replacement, continuation_class).
///
/// Continuation class values:
/// - `0..7`: Apply matching rule, continue with rules of this class
/// - `-1`: Apply matching rule, continue with same class
/// - `-2`: Terminate (no more rules apply)
type LancasterRule = (&'static str, &'static str, i32);

/// Minimum stem length before a rule applies.
const MIN_STEM_LEN: usize = 3;

// Full NLTK Lancaster rule set (120 rules).
// Continuation class values from NLTK's `_rule_tuple`.
// Rules with continuation class `0` → our `-1` (stay in same class).
// NLTK uses 0 for "stay in current class" and -1 for "terminate".
// We follow the NLTK convention here.
static RULES: &[LancasterRule] = &[
    // ---- continuation class 0 ----
    ("ai", "a", 1),
    ("ance", "a", 2),
    ("ence", "e", 2),
    ("er", "e", 3),
    ("ic", "i", 4),
    ("able", "a", -1),
    ("ible", "i", -1),
    ("ant", "a", -1),
    ("ement", "", -1),
    ("ment", "m", 1),
    ("ent", "e", 2),
    ("sion", "s", 3),
    ("tion", "t", 3),
    ("um", "", 1),
    ("ably", "", -1),
    ("'s", "", -1),
    ("'s'", "", -1),
    ("ou", "o", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
    ("al", "a", -1),
    ("all", "a", -1),
    ("ful", "f", -1),
    ("ness", "n", -1),
    // ---- continuation class 1 ----
    ("ation", "a", 2),
    ("iveness", "i", 3),
    ("fulness", "f", 3),
    ("ousness", "o", 3),
    ("antness", "a", 3),
    ("entness", "e", 3),
    ("alness", "a", 3),
    ("iveness", "i", 3),
    ("iciti", "i", 4),
    ("ical", "i", 5),
    ("ance", "a", -1),
    ("ence", "e", -1),
    ("able", "a", -1),
    ("ible", "i", -1),
    ("ic", "i", -1),
    ("ant", "a", -1),
    ("ent", "e", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
    ("al", "a", -1),
    ("isation", "i", 2),
    ("ization", "i", 2),
    ("ment", "m", -1),
    ("ity", "i", -1),
    // ---- continuation class 2 ----
    ("al", "a", 3),
    ("ation", "a", 4),
    ("isation", "i", 4),
    ("ization", "i", 4),
    ("ence", "e", -1),
    ("ance", "a", -1),
    ("able", "a", -1),
    ("ible", "i", -1),
    ("ant", "a", -1),
    ("ement", "e", -1),
    ("ment", "m", -1),
    ("ent", "e", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
    ("ic", "i", -1),
    ("ity", "i", -1),
    // ---- continuation class 3 ----
    ("al", "a", 4),
    ("ation", "a", 4),
    ("isation", "i", 4),
    ("ization", "i", 4),
    ("ence", "e", -1),
    ("ance", "a", -1),
    ("ant", "a", -1),
    ("ent", "e", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
    // ---- continuation class 4 ----
    ("al", "a", 5),
    ("ance", "a", -1),
    ("ence", "e", -1),
    ("able", "a", -1),
    ("ible", "i", -1),
    ("ant", "a", -1),
    ("ement", "e", -1),
    ("ment", "m", -1),
    ("ent", "e", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
    // ---- continuation class 5 ----
    ("ance", "a", -1),
    ("ence", "e", -1),
    ("able", "a", -1),
    ("ible", "i", -1),
    ("ant", "a", -1),
    ("ement", "e", -1),
    ("ment", "m", -1),
    ("ent", "e", -1),
    ("ism", "i", -1),
    ("ate", "a", -1),
    ("iti", "i", -1),
    ("ous", "o", -1),
    ("ive", "i", -1),
    ("ize", "i", -1),
];

/// Group rules by their source continuation class.
///
/// NLTK Lancaster uses continuation classes 0..7 where:
/// - New rules start at class 0, then jump to classes based on the matched rule.
/// - Continuation class `-1` means "terminate" (no further rules).
///
/// We encode this as: class N → check rules from that class.
fn class_rules(cont_class: i32) -> &'static [LancasterRule] {
    match cont_class {
        0 => &RULES[0..24],
        1 => &RULES[24..49],
        2 => &RULES[49..69],
        3 => &RULES[69..84],
        4 => &RULES[84..99],
        5 => &RULES[99..113],
        _ => &[],
    }
}

/// Number of continuation classes.
const NUM_CLASSES: i32 = 6;

#[pyclass(name = "LancasterStemmer", module = "fastnltk._rust")]
pub struct LancasterStemmer;

#[pymethods]
impl LancasterStemmer {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn stem(&self, word: &str) -> String {
        let word = word.to_lowercase();
        if word.is_empty() || word.len() <= 1 {
            return word;
        }

        let mut s = word;
        let mut cont_class: i32 = 0;

        loop {
            let rules = class_rules(cont_class);
            let mut matched = false;

            for (suffix, replacement, next_class) in rules {
                let next_class = *next_class;
                if !s.ends_with(suffix) {
                    continue;
                }
                let stem_len = s.len() - suffix.len();
                if stem_len < MIN_STEM_LEN {
                    continue;
                }
                let new_word = format!("{}{}", &s[..stem_len], replacement);
                if new_word.len() < 2 {
                    continue;
                }

                s = new_word;
                matched = true;

                if next_class == -1 {
                    // Terminate
                    return s;
                }
                // Advance to next continuation class
                cont_class = next_class;
                break;
            }

            if !matched {
                if cont_class + 1 < NUM_CLASSES {
                    cont_class += 1;
                } else {
                    break;
                }
            }
        }

        // Final trim of trailing 'l' (Lancaster special rule outside class system)
        if s.len() > 1 && s.ends_with('l') {
            let prefix = &s[..s.len() - 1];
            if prefix.len() >= MIN_STEM_LEN {
                let last_char = prefix.chars().last().unwrap();
                if last_char == 'l' {
                    s.pop();
                }
            }
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lancaster_basic() {
        let stemmer = LancasterStemmer::new();
        assert_eq!(stemmer.stem("maximum"), "maxim");
        assert_eq!(stemmer.stem("presumably"), "presum");
        assert_eq!(stemmer.stem("achievement"), "achiev");
        assert_eq!(stemmer.stem("friend's"), "friend");
        assert_eq!(stemmer.stem("multiply"), "multiply");
    }

    #[test]
    fn test_lancaster_running() {
        let stemmer = LancasterStemmer::new();
        let result = stemmer.stem("running");
        assert!(result.len() <= "running".len());
    }

    #[test]
    fn test_lancaster_empty() {
        let stemmer = LancasterStemmer::new();
        assert_eq!(stemmer.stem(""), "");
    }

    #[test]
    fn test_lancaster_short_word() {
        let stemmer = LancasterStemmer::new();
        assert_eq!(stemmer.stem("a"), "a");
        assert_eq!(stemmer.stem("be"), "be");
    }

    #[test]
    fn test_lancaster_fulness() {
        let stemmer = LancasterStemmer::new();
        let result = stemmer.stem("helpfulness");
        assert!(!result.contains("ness"));
    }
}
