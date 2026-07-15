use std::sync::LazyLock;
use pyo3::prelude::*;
use regex::Regex;
use smol_str::SmolStr;
use std::borrow::Cow;

#[pyclass(name = "ToktokTokenizer", module = "fastnltk._rust")]
pub struct ToktokTokenizer;

static TOKTOK_SUBS: LazyLock<Vec<(Regex, String)>> = LazyLock::new(build_subs);

fn mk_re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|e| panic!("bad regex '{p}': {e}"))
}

fn sub(p: &str, r: &str) -> (Regex, String) {
    (mk_re(p), r.to_string())
}

#[pymethods]
impl ToktokTokenizer {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (text, return_str=false))]
    fn tokenize(&self, text: &str, return_str: bool) -> Vec<String> {
        let subs = &*TOKTOK_SUBS;
        let mut s: Cow<str> = Cow::Borrowed(text);
        for (re, replacement) in subs {
            if let Cow::Borrowed(inner) = s {
                if re.is_match(inner) {
                    s = Cow::Owned(re.replace_all(inner, replacement.as_str()).into_owned());
                }
            } else if let Cow::Owned(ref inner) = s {
                if re.is_match(inner) {
                    s = Cow::Owned(re.replace_all(inner, replacement.as_str()).into_owned());
                }
            }
        }
        let t = s.trim().to_string();
        if return_str {
            return vec![t];
        }
        if t.is_empty() {
            return Vec::new();
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        let mut out = Vec::with_capacity(parts.len());
        for p in parts {
            out.push(SmolStr::new(p).to_string());
        }
        out
    }
}

fn build_subs() -> Vec<(Regex, String)> {
    vec![
        sub("\u{00a0}", " "),
        sub(r"([،;؛¿!\x22\]\)}»›\u{201d}؟¡%\u{066a}°±©®।॥\u{2026}])", " $1 "),
        sub(r"([({\[\u2018\u201c\u201e\u201a\u2019\xab\u2039\u300c\u300e])", " $1 "),
        sub(r"[\u{2013}\u{2014}]", " $0 "),
        sub(r"& ", "&amp; "),
        sub("\t", " &#9; "),
        sub(r"\|", " &#124; "),
        sub(r"([\x27\u2019`])", " $1 "),
        sub(r" ` ` ", " `` "),
        sub(r" ' ' ", " '' "),
        sub(r"(?m)\.$", " ."),
        sub(r"(?m)\.\s*([\x22\x27\u2019\xbb\u203a\u201d])\s*$", " . $1"),
        sub(r"(?m)\?$", " ?"),
        sub(r"(?m)!$", " !"),
        sub(r"(,{2,})", " $1 "),
        sub(r"(-{2,})", " $1 "),
        sub(r"(\.{2,})", " $1 "),
        sub(
            r"([\[({⸨\u201a\u201e\u2045\u207d\u208d\u2329\u27e6\u27e8\u27ea\u27ec\u27ee\u2983\u2985\u2987\u2989\u298b\u298d\u298f\u2991\u2993\u2995\u2997\u29d8\u29da\u29fc\u2e22\u2e24\u2e26\u2e28\u3008\u300a\u300c\u300e\u3010\u3014\u3016\u3018\u301a\u301d\ufe17\ufe59\ufe5b\ufe5d\uff08\uff3b\uff5b\uff5f\uff62])",
            "$1 ",
        ),
        sub(
            r"([)\]}⨀\u0f3b\u0f3d\u2046\u207e\u208e\u232a\u27e7\u27e9\u27eb\u27ed\u27ef\u2984\u2986\u2988\u298a\u298c\u298e\u2990\u2992\u2994\u2996\u2998\u29d9\u29db\u29fd\u2e23\u2e25\u2e27\u2e29\u3009\u300b\u300d\u300f\u3011\u3015\u3017\u3019\u301b\u301e\ufe18\ufe36\ufe38\ufe3a\ufe3c\ufe3e\ufe40\ufe42\ufe44\ufe48\ufe5a\ufe5c\ufe5e\uff09\uff3d\uff5d\uff60\uff63])",
            " $1",
        ),
        sub(
            r"([$\xa2\xa3\xa4\xa5\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\u20a0\u20a1\u20a2\u20a3\u20a4\u20a5\u20a6\u20a7\u20a8\u20a9\u20aa\u20ab\u20ac\u20ad\u20ae\u20af\u20b0\u20b1\u20b2\u20b3\u20b4\u20b5\u20b6\u20b7\u20b8\u20b9\u20ba\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6])",
            "$1 ",
        ),
        sub(r" {2,}", " "),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> ToktokTokenizer {
        ToktokTokenizer
    }

    #[test]
    fn test_basic() {
        let r = tok().tokenize("Is 9.5 or 525,600 my favorite number?", false);
        assert_eq!(r, vec!["Is", "9.5", "or", "525,600", "my", "favorite", "number", "?"]);
    }

    #[test]
    fn test_empty() {
        assert!(tok().tokenize("", false).is_empty());
    }

    #[test]
    fn test_period() {
        let r = tok().tokenize("Hello world.", false);
        assert_eq!(r, vec!["Hello", "world", "."]);
    }

    #[test]
    fn test_exclam() {
        let r = tok().tokenize("Wow!", false);
        assert_eq!(r, vec!["Wow", "!"]);
    }

    #[test]
    fn test_return_str() {
        let r = tok().tokenize("Hello world.", true);
        assert_eq!(r.len(), 1);
        assert!(r[0].contains("Hello"));
    }

    #[test]
    fn test_simple() {
        assert_eq!(tok().tokenize("Hello world", false), vec!["Hello", "world"]);
    }

    #[test]
    fn test_punct() {
        let r = tok().tokenize("Hi! How are you?", false);
        assert_eq!(r, vec!["Hi", "!", "How", "are", "you", "?"]);
    }
}
