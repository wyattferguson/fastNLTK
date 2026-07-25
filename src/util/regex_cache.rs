//! Regex compilation cache.
//!
//! Uses `parking_lot::RwLock` over `HashMap` to allow concurrent reads.
//! Write lock only taken for cache-miss compilation (rare after warmup).
//! For rayon-based parallel tokenization, consider `dashmap`.

use std::collections::HashMap;

use parking_lot::RwLock;
use regex::Regex;
use std::sync::LazyLock;

static REGEX_CACHE: LazyLock<RwLock<HashMap<(String, u32), Regex>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Get or compile a regex, caching it by pattern+flags.
pub fn get_or_compile(pattern: &str, flags: u32) -> Result<Regex, regex::Error> {
    let key = (pattern.to_string(), flags);

    // Fast path: check cache with read lock
    {
        let cache = REGEX_CACHE.read();
        if let Some(re) = cache.get(&key) {
            return Ok(re.clone());
        }
    }

    // Compile outside any lock (regex compilation is expensive)
    let re = compile_regex(pattern, flags)?;

    // Write lock to insert
    {
        let mut cache = REGEX_CACHE.write();
        // Check again — another thread may have inserted while we compiled
        if let Some(existing) = cache.get(&key) {
            return Ok(existing.clone());
        }
        cache.insert(key, re.clone());
    }

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
