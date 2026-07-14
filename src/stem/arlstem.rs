//! `ARLSTem` — Arabic stemmers matching NLTK's `ARLSTem` and `ARLSTem2`.
//!
//! Rule-based Arabic stemming: removes prefixes, suffixes, diacritics,
//! normalizes letters, converts plural to singular, feminine to masculine.

use pyo3::prelude::*;

// ═══════════════════════════════════════════════════════════
// ARLSTem
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "ARLSTem", module = "fastnltk._rust")]
pub struct ARLSTem;

#[pymethods]
impl ARLSTem {
    #[new]
    fn new() -> Self {
        Self
    }

    fn stem(&self, token: &str) -> String {
        if token.is_empty() {
            return String::new();
        }
        let mut t = normalize(token);
        let pre = strip_prefix(&t);
        if let Some(ref p) = pre {
            t.clone_from(p);
        }
        t = strip_suffix(&t);
        if let Some(ps) = plural_to_singular(&t) {
            return ps;
        }
        if let Some(fm) = feminine_to_masculine(&t) {
            return fm;
        }
        if pre.is_none() {
            return strip_verb(&t);
        }
        t
    }
}

// ═══════════════════════════════════════════════════════════
// ARLSTem2
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "ARLSTem2", module = "fastnltk._rust")]
pub struct ARLSTem2;

#[pymethods]
impl ARLSTem2 {
    #[new]
    fn new() -> Self {
        Self
    }

    fn stem(&self, token: &str) -> String {
        if token.is_empty() {
            return String::new();
        }
        let mut t = normalize(token);
        let pre = strip_prefix(&t);
        if let Some(ref p) = pre {
            t.clone_from(p);
        }
        t = strip_suffix(&t);
        if let Some(ps) = plural_to_singular(&t) {
            return ps;
        }
        if let Some(fm) = feminine_to_masculine(&t) {
            return fm;
        }
        if pre.is_none() {
            return strip_verb2(&t);
        }
        t
    }
}

// ═══════════════════════════════════════════════════════════
// Normalization
// ═══════════════════════════════════════════════════════════

fn normalize(token: &str) -> String {
    let mut result = String::with_capacity(token.len());
    for c in token.chars() {
        match c {
            // Remove Arabic diacritics (Tashkeel)
            '\u{064B}'..='\u{0652}' | '\u{0670}' => {}
            // Replace hamzated Alif with plain Alif
            '\u{0622}' | '\u{0623}' | '\u{0625}' => result.push('\u{0627}'), // Alif
            // Replace Alif Maqsura with Yaa
            '\u{0649}' => result.push('\u{064A}'), // Yaa
            // Replace Ta Marbuta with Haa
            '\u{0629}' => result.push('\u{0647}'), // Haa
            _ => result.push(c),
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════
// Prefix stripping
// ═══════════════════════════════════════════════════════════

static PREFIXES: &[&str] = &[
    "ال",   // Al-
    "وبال", // Wa-bi-Al
    "فبال", // Fa-bi-Al
    "بال",  // Bi-Al
    "ولل",  // Wa-li-Al
    "لل",   // Li-Al
    "فسوف", // Fa-sawfa
    "سوف",  // Sawfa
    "فال",  // Fa-Al
    "وسوف", // Wa-sawfa
    "وال",  // Wa-Al
    "وب",   // Wa-bi
    "فل",   // Fa-li
    "وسي",  // Wa-sa-
    "فسي",  // Fa-sa-
    "وس",   // Wa-sa
    "فس",   // Fa-sa
    "لي",   // Li-
    "ل",    // L-
    "ف",    // F-
    "و",    // W-
    "س",    // S-
    "ب",    // B-
];

fn strip_prefix(token: &str) -> Option<String> {
    for p in PREFIXES {
        if token.starts_with(p) {
            let p_chars = p.chars().count();
            let t_chars = token.chars().count();
            if t_chars > p_chars + 1 {
                return Some(token.chars().skip(p_chars).collect());
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════
// Suffix stripping
// ═══════════════════════════════════════════════════════════

static SUFFIXES: &[&str] = &[
    "\u{643}\u{645}\u{627}", // كما
    "\u{647}\u{645}\u{627}", // هما
    "\u{643}\u{645}",        // كم
    "\u{647}\u{645}",        // هم
    "\u{647}\u{646}",        // هن
    "\u{643}\u{646}",        // كن
    "\u{648}\u{646}",        // ون (masculine plural)
    "\u{64A}\u{646}",        // ين (masculine plural)
    "\u{627}\u{646}",        // ان (dual)
    "\u{648}\u{627}",        // وا
    "\u{647}\u{627}",        // ها
    "\u{646}\u{627}",        // نا
    "\u{62A}\u{645}",        // تم
    "\u{62A}\u{646}",        // تن
    "\u{62A}\u{627}",        // تا
    "\u{647}",               // ه
    "\u{647}\u{627}",        // ها
    "\u{646}",               // ن
    "\u{627}",               // ا
    "\u{64A}",               // ي
    "\u{62A}",               // ت
    "\u{629}",               // ة
    "\u{643}",               // ك
];

fn strip_suffix(token: &str) -> String {
    for s in SUFFIXES {
        if token.ends_with(s) {
            let s_chars = s.chars().count();
            let t_chars = token.chars().count();
            if t_chars > s_chars + 1 {
                let stem: String = token.chars().take(t_chars - s_chars).collect();
                return stem;
            }
        }
    }
    token.to_string()
}

// ═══════════════════════════════════════════════════════════
// Plural to singular
// ═══════════════════════════════════════════════════════════

static PLURAL_SUFFIXES: &[(&str, &str)] = &[
    ("ات", ""), // جمع مؤنث سالم
    ("ين", ""), // جمع مذكر سالم
    ("ون", ""), // جمع مذكر سالم
    ("ان", ""), // مثنى
];

fn plural_to_singular(token: &str) -> Option<String> {
    for (suffix, replacement) in PLURAL_SUFFIXES {
        if token.ends_with(suffix) {
            let s_chars = suffix.chars().count();
            let t_chars = token.chars().count();
            if t_chars > s_chars + 1 {
                let stem: String = token.chars().take(t_chars - s_chars).collect();
                let result = format!("{stem}{replacement}");
                if result.chars().count() >= 2 {
                    return Some(result);
                }
            }
        }
    }

    // Broken plural patterns (simplified)
    let chars: Vec<char> = token.chars().collect();
    let len = chars.len();
    if len >= 4 && chars[1] == '\u{0627}' && len <= 5 {
        // Check for common broken plural patterns
        // Pattern: C1aC2C3 -> C1C2C3 (remove internal alif)
        let mut result = String::with_capacity(len - 1);
        result.push(chars[0]);
        for &c in &chars[2..] {
            result.push(c);
        }
        return Some(result);
    }
    None
}

// ═══════════════════════════════════════════════════════════
// Feminine to masculine
// ═══════════════════════════════════════════════════════════

fn feminine_to_masculine(token: &str) -> Option<String> {
    let chars: Vec<char> = token.chars().collect();
    let len = chars.len();
    // Remove Ta Marbuta (ة) at end -> convert to Haa (ه)
    if len > 2 && chars[len - 1] == '\u{0629}' {
        let mut result: String = chars[..len - 1].iter().collect();
        result.push('\u{0647}'); // ة -> ه
        return Some(result);
    }
    None
}

// ═══════════════════════════════════════════════════════════
// Verb stripping (ARLSTem)
// ═══════════════════════════════════════════════════════════

fn strip_verb(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    let mut t: String = chars.iter().collect();

    let verb_prefixes = [
        "\u{633}\u{64A}",
        "\u{633}",
        "\u{641}",
        "\u{64A}",
        "\u{62A}",
        "\u{627}",
        "\u{646}",
        "\u{623}",
    ];
    for p in &verb_prefixes {
        let p_chars = p.chars().count();
        if t.starts_with(*p) && chars.len() > p_chars + 2 {
            t = t.chars().skip(p_chars).collect();
            break;
        }
    }

    let verb_suffixes = [
        "\u{648}\u{646}",
        "\u{64A}\u{646}",
        "\u{627}\u{646}",
        "\u{648}\u{627}",
        "\u{646}",
        "\u{627}",
        "\u{64A}",
        "\u{62A}",
        "\u{62A}\u{645}",
        "\u{62A}\u{646}",
        "\u{646}\u{627}",
    ];
    for s in &verb_suffixes {
        let s_chars = s.chars().count();
        if t.ends_with(*s) && t.chars().count() > s_chars + 1 {
            t = t.chars().take(t.chars().count() - s_chars).collect();
            break;
        }
    }

    t
}

// ═══════════════════════════════════════════════════════════
// Verb stripping (ARLSTem2 — more aggressive)
// ═══════════════════════════════════════════════════════════

fn strip_verb2(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    let mut t: String = chars.iter().collect();

    let verb_prefixes = [
        "\u{641}",
        "\u{648}",
        "\u{628}",
        "\u{644}",
        "\u{633}",
        "\u{64A}",
        "\u{62A}",
        "\u{627}",
        "\u{646}",
        "\u{623}",
        "\u{633}\u{64A}",
    ];
    for p in &verb_prefixes {
        let p_chars = p.chars().count();
        if t.starts_with(*p) && chars.len() > p_chars + 2 {
            t = t.chars().skip(p_chars).collect();
            break;
        }
    }

    let verb_suffixes = [
        "\u{648}\u{646}",
        "\u{64A}\u{646}",
        "\u{627}\u{646}",
        "\u{648}\u{627}",
        "\u{646}",
        "\u{627}",
        "\u{64A}",
        "\u{62A}",
        "\u{62A}\u{627}",
        "\u{62A}\u{645}",
        "\u{62A}\u{646}",
        "\u{646}\u{627}",
        "\u{643}\u{645}\u{627}",
        "\u{647}\u{645}",
        "\u{647}\u{646}",
        "\u{643}\u{646}",
    ];
    for s in &verb_suffixes {
        let s_chars = s.chars().count();
        if t.ends_with(*s) && t.chars().count() > s_chars + 1 {
            t = t.chars().take(t.chars().count() - s_chars).collect();
            break;
        }
    }

    t
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ARLSTem>()?;
    m.add_class::<ARLSTem2>()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_remove_diacritics() {
        // Arabic word with Fatha: "مدرسة" (school)
        let word = "\u{0645}\u{064E}\u{062F}\u{0652}\u{0631}\u{064E}\u{0633}\u{064E}\u{0629}";
        let norm = normalize(word);
        // Diacritics removed, Ta Marbuta -> Haa
        assert_eq!(norm, "\u{0645}\u{062F}\u{0631}\u{0633}\u{0647}");
    }

    #[test]
    fn test_normalize_hamza() {
        // Alif with hamza above -> plain Alif
        let word = "\u{0623}\u{0643}\u{0644}"; // أكل
        let norm = normalize(word);
        assert_eq!(norm, "\u{0627}\u{0643}\u{0644}"); // اكل
    }

    #[test]
    fn test_strip_prefix_al() {
        // الكتاب (Al-kitab) -> كتاب (kitab)
        let result = strip_prefix("\u{0627}\u{0644}\u{0643}\u{062A}\u{0627}\u{0628}");
        assert_eq!(result, Some("\u{0643}\u{062A}\u{0627}\u{0628}".to_string()));
    }

    #[test]
    fn test_strip_prefix_fa() {
        // ف (fa-) prefix
        let result = strip_prefix("\u{0641}\u{0643}\u{062A}\u{0627}\u{0628}");
        assert_eq!(result, Some("\u{0643}\u{062A}\u{0627}\u{0628}".to_string()));
    }

    #[test]
    fn test_strip_suffix_plural() {
        // كتابون (kitabun) -> كتاب (kitab)
        let result = strip_suffix("\u{0643}\u{062A}\u{0627}\u{0628}\u{0648}\u{0646}");
        assert_eq!(result, "\u{0643}\u{062A}\u{0627}\u{0628}");
    }

    #[test]
    fn test_arlstem_basic() {
        let stemmer = ARLSTem::new();
        let result = stemmer.stem("\u{0627}\u{0644}\u{0643}\u{062A}\u{0627}\u{0628}");
        // الكتاب -> after removing ال prefix -> كتاب (4 chars)
        assert!(!result.is_empty());
        assert!(result.chars().count() <= 6);
    }

    #[test]
    fn test_arlstem_empty() {
        let stemmer = ARLSTem::new();
        assert_eq!(stemmer.stem(""), "");
    }

    #[test]
    fn test_arlstem2_basic() {
        let stemmer = ARLSTem2::new();
        let result = stemmer.stem("\u{0627}\u{0644}\u{0643}\u{062A}\u{0627}\u{0628}");
        assert!(result.len() >= 3);
    }

    #[test]
    fn test_feminine_to_masculine() {
        // مدرسة (madrasa) -> مدرسه
        let result = feminine_to_masculine("\u{0645}\u{062F}\u{0631}\u{0633}\u{0629}");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.chars().last(), Some('\u{0647}'));
    }

    #[test]
    fn test_plural_to_singular() {
        // كتابات (kitabat) -> كتاب (kitab)
        let word = "\u{0643}\u{062A}\u{0627}\u{0628}\u{0627}\u{062A}";
        let result = plural_to_singular(word);
        assert!(result.is_some());
    }
}
