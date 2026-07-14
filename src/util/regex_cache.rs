//! Simple regex compilation cache.
//!
//! Compiled regexes are cached by pattern+flags to avoid recompilation
//! overhead when the same pattern is used repeatedly.

use std::collections::HashMap;

use std::sync::LazyLock;
use parking_lot::Mutex;
use regex::Regex;

static REGEX_CACHE: LazyLock<Mutex<HashMap<(String, u32), Regex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get or compile a regex, caching it by pattern+flags.
pub fn get_or_compile(pattern: &str, flags: u32) -> Result<Regex, regex::Error> {
    let key = (pattern.to_string(), flags);
    let cache = REGEX_CACHE.lock();

    if let Some(re) = cache.get(&key) {
        return Ok(re.clone());
    }

    // Compile (drop lock during compilation to avoid blocking other threads)
    drop(cache);
    let re = compile_regex(pattern, flags)?;

    // Re-acquire lock to store
    REGEX_CACHE.lock().insert(key, re.clone());

    Ok(re)
}

/// Compile a regex with the given flags.
fn compile_regex(pattern: &str, flags: u32) -> Result<Regex, regex::Error> {
    let mut builder = regex::RegexBuilder::new(pattern);

    if flags & 0x02 != 0 {
        builder.case_insensitive(true);
    }
    if flags & 0x08 != 0 {
        builder.multi_line(true);
    }
    if flags & 0x10 != 0 {
        builder.dot_matches_new_line(true);
    }
    if flags & 0x100 != 0 {
        builder.unicode(false);
    } else {
        builder.unicode(true);
    }
    if flags & 0x40 != 0 {
        builder.ignore_whitespace(true);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_or_compile() {
        let re = get_or_compile(r"\w+", 0).unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("   "));
    }

    #[test]
    fn test_cache_hit() {
        let re1 = get_or_compile(r"\d+", 0).unwrap();
        let re2 = get_or_compile(r"\d+", 0).unwrap();
        assert!(re1.is_match("123"));
        assert!(re2.is_match("456"));
    }

    #[test]
    fn test_invalid_pattern() {
        let result = get_or_compile(r"[invalid", 0);
        assert!(result.is_err());
    }
}
