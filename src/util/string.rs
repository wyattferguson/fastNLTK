//! String utility functions.

/// Count occurrences of a character in a string.
#[must_use]
pub fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

/// Check if string contains only whitespace.
pub fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// Collapse multiple whitespace characters into a single space.
#[must_use]
pub fn collapse_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}
