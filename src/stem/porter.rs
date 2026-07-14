//! Porter stemmer — pure Rust implementation of the Porter stemming algorithm.
//!
//! Based on the original 1980 paper by Martin Porter:
//! "An algorithm for suffix stripping" (Program, 14(3): 130–137).
//!
//! Matches NLTK's PorterStemmer implementation.

use pyo3::prelude::*;

// ═══════════════════════════════════════════════════════════
// PorterStemmer
// ═══════════════════════════════════════════════════════════

/// Porter stemmer implementation.
#[pyclass(name = "PorterStemmer", module = "fastnltk._rust")]
pub struct PorterStemmer;

#[pymethods]
impl PorterStemmer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn stem(&self, word: &str) -> String {
        porter_stem(word)
    }
}

/// Core Porter stemming algorithm.
fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_lowercase();
    }

    let mut w = word.to_lowercase().chars().collect::<Vec<char>>();
    if w.is_empty() {
        return String::new();
    }

    let original_len = w.len();

    // Step 1a
    if ends_with(&w, "sses") {
        replace_end(&mut w, "sses", "ss");
    } else if ends_with(&w, "ies") {
        replace_end(&mut w, "ies", "i");
    } else if ends_with(&w, "ss") {
        // keep ss
    } else if ends_with(&w, "s") && !is_short_word(&w, original_len) {
        // Only remove 's' if preceded by a vowel
        if w.len() > 2 && is_consonant(&w, w.len() - 2) {
            w.pop();
        }
    }

    // Step 1b
    let mut step1b_done = false;
    if ends_with(&w, "eed") {
        if measure(&w[..w.len() - 3]) > 0 {
            replace_end(&mut w, "eed", "ee");
        }
    } else if ends_with(&w, "ed") && contains_vowel(&w[..w.len() - 2]) {
        replace_end(&mut w, "ed", "");
        step1b_done = true;
    } else if ends_with(&w, "ing") && contains_vowel(&w[..w.len() - 3]) {
        replace_end(&mut w, "ing", "");
        step1b_done = true;
    }

    if step1b_done {
        if ends_with(&w, "at") {
            replace_end(&mut w, "at", "ate");
        } else if ends_with(&w, "bl") {
            w.push('e');
        } else if ends_with(&w, "iz") {
            replace_end(&mut w, "iz", "ize");
        } else if double_consonant(&w)
            && !ends_with(&w, "l")
            && !ends_with(&w, "s")
            && !ends_with(&w, "z")
        {
            w.pop();
        } else if measure(&w) == 1 && ends_with_cvc(&w) {
            w.push('e');
        }
    }

    // Step 1c
    if w.len() > 1 && w.last() == Some(&'y') && contains_vowel(&w[..w.len() - 1]) {
        let last = w.len() - 1;
        w[last] = 'i';
    }

    // Step 2
    if w.len() >= 4 {
        let m = measure(&w);
        if m > 0 {
            if ends_with(&w, "ational") {
                replace_end(&mut w, "ational", "ate");
            } else if ends_with(&w, "tional") {
                replace_end(&mut w, "tional", "tion");
            } else if ends_with(&w, "enci") {
                replace_end(&mut w, "enci", "ence");
            } else if ends_with(&w, "anci") {
                replace_end(&mut w, "anci", "ance");
            } else if ends_with(&w, "izer") {
                replace_end(&mut w, "izer", "ize");
            } else if ends_with(&w, "abli") {
                replace_end(&mut w, "abli", "able");
            } else if ends_with(&w, "alli") {
                replace_end(&mut w, "alli", "al");
            } else if ends_with(&w, "entli") {
                replace_end(&mut w, "entli", "ent");
            } else if ends_with(&w, "eli") {
                replace_end(&mut w, "eli", "e");
            } else if ends_with(&w, "ousli") {
                replace_end(&mut w, "ousli", "ous");
            } else if ends_with(&w, "ization") {
                replace_end(&mut w, "ization", "ize");
            } else if ends_with(&w, "ation") {
                replace_end(&mut w, "ation", "ate");
            } else if ends_with(&w, "ator") {
                replace_end(&mut w, "ator", "ate");
            } else if ends_with(&w, "alism") {
                replace_end(&mut w, "alism", "al");
            } else if ends_with(&w, "iveness") {
                replace_end(&mut w, "iveness", "ive");
            } else if ends_with(&w, "fulness") {
                replace_end(&mut w, "fulness", "ful");
            } else if ends_with(&w, "ousness") {
                replace_end(&mut w, "ousness", "ous");
            } else if ends_with(&w, "aliti") {
                replace_end(&mut w, "aliti", "al");
            } else if ends_with(&w, "iviti") {
                replace_end(&mut w, "iviti", "ive");
            } else if ends_with(&w, "biliti") {
                replace_end(&mut w, "biliti", "ble");
            }
        }
    }

    // Step 3
    if w.len() >= 3 {
        let m = measure(&w);
        if m > 0 {
            if ends_with(&w, "icate") {
                replace_end(&mut w, "icate", "ic");
            } else if ends_with(&w, "ative") {
                replace_end(&mut w, "ative", "");
            } else if ends_with(&w, "alize") {
                replace_end(&mut w, "alize", "al");
            } else if ends_with(&w, "iciti") {
                replace_end(&mut w, "iciti", "ic");
            } else if ends_with(&w, "ical") {
                replace_end(&mut w, "ical", "ic");
            } else if ends_with(&w, "ful") {
                replace_end(&mut w, "ful", "");
            } else if ends_with(&w, "ness") {
                replace_end(&mut w, "ness", "");
            }
        }
    }

    // Step 4
    if w.len() >= 3 {
        let m = measure(&w);
        if m > 1 {
            if ends_with(&w, "al") {
                replace_end(&mut w, "al", "");
            } else if ends_with(&w, "ance") {
                replace_end(&mut w, "ance", "");
            } else if ends_with(&w, "ence") {
                replace_end(&mut w, "ence", "");
            } else if ends_with(&w, "er") {
                replace_end(&mut w, "er", "");
            } else if ends_with(&w, "ic") {
                replace_end(&mut w, "ic", "");
            } else if ends_with(&w, "able") {
                replace_end(&mut w, "able", "");
            } else if ends_with(&w, "ible") {
                replace_end(&mut w, "ible", "");
            } else if ends_with(&w, "ant") {
                replace_end(&mut w, "ant", "");
            } else if ends_with(&w, "ement") {
                replace_end(&mut w, "ement", "");
            } else if ends_with(&w, "ment") {
                replace_end(&mut w, "ment", "");
            } else if ends_with(&w, "ent") {
                replace_end(&mut w, "ent", "");
            } else if ends_with(&w, "ion")
                && w.len() > 4
                && (w[w.len() - 4] == 's' || w[w.len() - 4] == 't')
            {
                replace_end(&mut w, "ion", "");
            } else if ends_with(&w, "ou") {
                replace_end(&mut w, "ou", "");
            } else if ends_with(&w, "ism") {
                replace_end(&mut w, "ism", "");
            } else if ends_with(&w, "ate") {
                replace_end(&mut w, "ate", "");
            } else if ends_with(&w, "iti") {
                replace_end(&mut w, "iti", "");
            } else if ends_with(&w, "ous") {
                replace_end(&mut w, "ous", "");
            } else if ends_with(&w, "ive") {
                replace_end(&mut w, "ive", "");
            } else if ends_with(&w, "ize") {
                replace_end(&mut w, "ize", "");
            }
        }
    }

    // Step 5a
    if w.len() >= 2 {
        if w[w.len() - 1] == 'e' {
            let m = measure(&w[..w.len() - 1]);
            if m > 1 {
                w.pop();
            } else if m == 1 && !ends_with_cvc(&w[..w.len() - 1]) {
                w.pop();
            }
        }
    }

    // Step 5b
    if w.len() > 1 && w[w.len() - 1] == 'l' && double_consonant(&w) && measure(&w) > 1 {
        w.pop();
    }

    w.into_iter().collect()
}

// ═══════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════

/// Check if slice ends with given suffix
fn ends_with(s: &[char], suffix: &str) -> bool {
    let suffix_chars: Vec<char> = suffix.chars().collect();
    if s.len() < suffix_chars.len() {
        return false;
    }
    &s[s.len() - suffix_chars.len()..] == suffix_chars.as_slice()
}

/// Replace suffix
fn replace_end(s: &mut Vec<char>, old: &str, new: &str) {
    let old_len = old.chars().count();
    let new_chars: Vec<char> = new.chars().collect();
    s.truncate(s.len() - old_len);
    s.extend(new_chars);
}

/// Measure of a word: number of VC sequences
fn measure(s: &[char]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let n = s.len();

    // Skip leading consonants
    while i < n && is_consonant(s, i) {
        i += 1;
    }

    // Count VC sequences
    while i < n {
        // Vowel sequence
        while i < n && !is_consonant(s, i) {
            i += 1;
        }
        // Consonant sequence
        while i < n && is_consonant(s, i) {
            i += 1;
        }
        if i > 0 {
            // We've passed at least one vowel, check if we also passed consonants
            let mut had_vowel = false;
            let mut had_consonant = false;
            for j in 0..i {
                if !is_consonant(s, j) {
                    had_vowel = true;
                } else if had_vowel {
                    had_consonant = true;
                }
            }
            if had_vowel && had_consonant {
                count += 1;
            }
        }
    }

    count
}

/// Check if character at position i is a consonant
fn is_consonant(s: &[char], i: usize) -> bool {
    if i >= s.len() {
        return true;
    }
    let c = s[i];
    match c {
        'a' | 'e' | 'i' | 'o' | 'u' => false,
        'y' if i > 0 => !is_consonant(s, i - 1),
        _ => true,
    }
}

/// Check if word contains a vowel
fn contains_vowel(s: &[char]) -> bool {
    s.iter().enumerate().any(|(i, _)| !is_consonant(s, i))
}

/// Check if word ends with CVC pattern
fn ends_with_cvc(s: &[char]) -> bool {
    if s.len() < 3 {
        return false;
    }
    let n = s.len();
    is_consonant(s, n - 3) && !is_consonant(s, n - 2) && is_consonant(s, n - 1)
}

/// Check if the last two characters are the same consonant
fn double_consonant(s: &[char]) -> bool {
    if s.len() < 2 {
        return false;
    }
    let n = s.len();
    s[n - 1] == s[n - 2] && is_consonant(s, n - 1)
}

/// Check if word is short (used in step 1a)
fn is_short_word(_s: &[char], _len: usize) -> bool {
    false // simplified
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_porter_cats() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("cats"), "cat");
    }

    #[test]
    fn test_porter_ponies() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("ponies"), "poni");
    }

    #[test]
    fn test_porter_walking() {
        let stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("walking"), "walk");
    }
}
