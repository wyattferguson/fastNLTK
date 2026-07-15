**OVERVIEW**: fastNLTK v0.4.0 — Rust/Python NLP library, 75 Rust source files (~20K LoC) + 18 Python shim files, 234 tests passing. Overall healthy: no `unsafe`, no `todo!`, 0 C dependencies (vendor/ excluded from review). Main concerns: PyO3 error handling patterns, 32 remaining ruff warnings, and several files >300 lines.

---

## ISSUES

### CRITICAL

- **`src/data.rs` uses `LazyLock<Vec<PathBuf>>` with `.unwrap()` if path setup fails** — if `nltk_data` directories can't be resolved at import time, the program panics instead of propagating a Python exception. Same pattern in `tweet.rs` line 9-22 (compiled regexes). Not immediately exploitable, but means certain import-time failures are uncatchable from Python. Fix: wrap in `std::sync::OnceLock<Result<...>>` and return fallible accessor.

### HIGH

- **32 remaining ruff `UP006` warnings** (`fastnltk/*.py`): files without `from __future__ import annotations` still use `List[str]` / `Tuple[int,int]` instead of `list[str]` / `tuple[int,int]`. Fix: add the `__future__` import to the 13 files missing it. Quick autofix (`ruff check --fix --unsafe-fixes` handles most).

- **`src/tag/sequential/taggers.rs` AffixTagger:** `_affix_len` and `_backoff` parameters accepted but ignored (prefix `_`). The constructor accepts `_affix_len: usize` but hardcodes `3` for slicing. If a user passes a different length it silently has no effect. Either respect the parameter or remove from signature.

- **`src/util/regex_cache.rs` uses `Mutex<HashMap>` for regex cache**: the lock is held across potential regex compilation (which can take milliseconds for complex patterns). `get_or_compile` drops the lock during compilation, but re-acquires it to store — double-lock pattern. For the single-threaded PyO3 use case this is fine, but the lock granularity is suboptimal if rayon is used. Consider `once_cell::sync::LazyLock` + `dashmap` for concurrent access.

### MEDIUM

- **`src/tag/perceptron.rs` Viterbi hot path**: `em_probs[j] <= 0.0` check is inside the `j` loop but recomputed for every `k` in the inner loop. The `em_ln` is computed before the `k` loop but `em_probs[j]` is checked both inside and outside. Minor redundancy.

- **`src/tag/tnt.rs` train() counts in two passes**: first pass collects tag names and word counts, second pass uses the built `tag_id` mapping. Could be done in one pass with `entry()` pattern (like the old code did). Two-pass doubles training time for large corpora.

- **`src/tokenize/simple.rs` `tokenize_space` returns `Vec<String>` for `""`**: empty string → `[""]` which matches NLTK but requires a special case branch. The `else` in the fast path adds 3 lines for a corner case.

- **`src/tokenize/mwe.rs` uses `HashMap<String, ...>` for trie nodes**: `String` keys instead of `SmolStr` or `Box<str>`. MWE tokens are typically short strings that would benefit from SmolStr inlining.

- **`vendor/rustling/` is 71 files of vendored source**: the actual diff is only ~5 files (Cargo.toml + cfg gates). Consider publishing the fork as a git dependency instead of vendoring the full source. Eliminates 51K lines from the working tree.

### LOW

- `src/drt.rs` (456 lines), `src/parse.rs` (422 lines), `src/tree.rs` (380 lines) — exceed 300-line guideline. Could extract sub-modules.
- `src/classify/textcat.rs` uses `HashMap<String, f64>` — SmolStr would inline language codes.
- `src/metrics/` directory has 8 files for 4 metric categories — consider consolidating small files.
- 118 clippy `must_use_candidate` warnings — easy `#[must_use]` annotations on pure functions.
- `Cargo.toml` has `unwrap_used = "allow"` in clippy lints — would prefer `#[allow()]` on individual sites.
- `src/chat.rs` imports many NLTK pieces for Eliza — could be pure Rust.

---

## RECOMMENDATIONS

1. **Fix UP006 warnings** (32 remaining) — add `from __future__ import annotations` to 13 Python shim files. 10 minutes, eliminates all `List`/`Tuple`/`Optional` legacy type annotations.
2. **Respect or remove `_affix_len` parameter** in `AffixTagger::new()` — silent ignored parameter is a UX bug.
3. **Publish rustling fork as git dep** — eliminates 71 vendored files from the tree, cleaner dependency management.
4. **Add `#[must_use]` to pure functions** — fix 118 clippy warnings with `cargo clippy --fix`.
5. **Handle import-time `LazyLock` fallibility in `data.rs`** — wrap in `OnceLock<Result>` so Python can catch configuration errors.
