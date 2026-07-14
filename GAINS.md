# Performance Optimization Plan — fastNLTK

> **Baseline:** v0.2.0 — 272 Rust tests, 236/236 Python tests, 23.3x avg speedup  
> **Allocation audit:** 719 alloc sites (244 clone, 337 to_string, 137 format!, 110 Box::new)  
> **Status:** Port complete. Optimization phase begins.

---

## Tier 1: Architecture (high effort, 2-5x gain)

### A1. Custom allocator — mimalloc
**Gain:** 10-20% across all benchmarks (allocation-heavy workloads see more)  
**Effort:** Trivial (1 line in lib.rs)

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

Add `mimalloc = "0.1"` to Cargo.toml. Microsoft's mimalloc outperforms system allocator for small, frequent allocations (tags, tokens, edges). Zero code changes required.  
**Why:** 719 allocation sites. Every `format!()`, `clone()`, `to_string()` goes through the allocator. mimalloc's free-list sharding reduces contention.

---

### A2. `smol_str` — small-string optimization for tokens/tags
**Gain:** 30-60% fewer heap allocs in tokenizer/tagger hot paths  
**Effort:** Medium (replace `String` with `SmolStr` in structs)

Tags like `"NN"`, `"VBZ"`, `"DT"` are 2-3 bytes. Tokens average 5-8 chars. `smol_str::SmolStr` stores ≤22 bytes inline on the stack, zero heap allocation.

**Files to change (7):**
- `tag/sequential.rs` — `word_to_tag: HashMap<SmolStr, SmolStr>` (48 clones eliminated)
- `tag/perceptron.rs` — feature strings (21 to_strings eliminated)
- `tokenize/mwe.rs` — TrieNode children keys
- `parse.rs` — `Production { lhs: SmolStr, rhs: Vec<SmolStr> }` (22 clones eliminated)
- `ccg/lexicon.rs` — `entries: HashMap<SmolStr, Vec<Category>>`
- `chunk.rs` — tag patterns, IOB tags
- `tokenize/punkt.rs` — abbrev_types, collocations

**Caveat:** `SmolStr` is immutable — if you need mutable strings, use `compact_str::CompactString` (aggressive re-inlining) instead.

---

### A3. `bumpalo` arena — eliminate 110 `Box::new()` calls
**Gain:** 2-4x on CCG parse, DRS operations, inference proofs  
**Effort:** Medium (arena parameter threading)

Every `Box::new(Expression)` and `Box<CCGEdge>` is a malloc. `bumpalo::Bump` allocates all objects in a contiguous slab, freed together.

**Files:**
- `ccg/chart.rs` — build CCEdge tree in arena, not Box
- `drt.rs` — `DRSCondition` variants use `&'bump DRS` instead of `Box<DRS>`
- `inference/mod.rs` — `Formula` uses arena refs
- `sem.rs` — `Expression` uses arena refs

```rust
// Before: Box<Expression> per node
// After: &'arena Expression — arena-backed reference
let expr: &Expression = bump.alloc(Expression::And(l, r));
```

**Caveat:** Requires lifetime threading through all recursive functions. Largest refactor by scope.

---

### A4. `Logos` DFA lexer — replace regex tokenization
**Gain:** 2-4x on treebank/toktok/tweet tokenizers  
**Effort:** High (reimplement tokenizer logic as Logos derive)

Regex-based tokenizers compile regex → search text → collect matches. A DFA-based lexer compiles to a jump table at build time, executing in O(n) with no backtracking.

```rust
#[derive(Logos)]
enum TreebankToken {
    #[regex(r"[A-Za-z]+")]
    Word,
    #[token("n't")]
    Contraction,
    #[regex(r"[.,!?;:]")]
    Punct,
    // ... etc
}
```

**Files affected:**
- `tokenize/treebank.rs` — multiple regex passes → single Logos pass
- `tokenize/toktok.rs` — regex-based replacements → Logos lexer
- `tokenize/tweet.rs` — URL/mention/hashtag detection via Logos
- `chunk.rs` — tag pattern regex → compile to Logos at construction

**Caveat:** Requires `logos = "0.15"` dep. Logos only handles known patterns; user-supplied regex in `RegexpTokenizer` stays as-is.

---

## Tier 2: Algorithmic (medium effort, 1.3-3x gain)

### B1. Binary search on sorted `Vec` for small maps
**Gain:** 2-5x faster lookup for < 15 entries (no hashing overhead)  
**Effort:** Low (replace HashMap with sorted Vec + binary_search)

Many HashMaps have < 15 entries (chunk rules, CCG combinators, tag patterns). Hashing a 3-char string costs more than linear scan or binary search.

```rust
// Before: HashMap<String, Vec<usize>> 
// After:  Vec<(String, Vec<usize>)> sorted, use binary_search_by_key
let idx = map.binary_search_by_key(&key, |(k,_)| k).ok();
```

**Candidates (12 files):**
- `ccg/combinator.rs` — 6 combinators → use array + linear scan
- `chunk.rs` — grammar rules (typically 1-5)
- `classify/maxent.rs` — label_counts (10-50 labels)
- `probability.rs` — ConditionalFreqDist conditions
- `tag/perceptron.rs` — feature index

---

### B2. `SmallVec<[T; 8]>` for fixed-capacity collections
**Gain:** Eliminates 74 `Vec::new()` heap allocations  
**Effort:** Low (replace `Vec` with `SmallVec` in known-small collections)

CCG chart edges per cell: 1-5 typically. Parse productions per LHS: 1-10. Chunk rules per grammar: 1-5. Using `SmallVec<[T; 8]>` stores up to 8 elements inline.

```rust
use smallvec::{SmallVec, smallvec};
// Before: Vec<CCGEdge>  // heap allocates for first element
// After:  SmallVec<[CCGEdge; 8]>  // inline for ≤8, heap only beyond
```

**Candidates:** ccg/chart.rs, parse.rs, chunk.rs, tokenize/regexp.rs, classify/*.rs

---

### B3. `Cow<'_, str>` in PyO3 FFI methods
**Gain:** 20-40% fewer string allocations in Python FFI hot paths  
**Effort:** Medium (change method signatures to accept `Cow<str>` or `&str`)

PyO3 can give you `Cow<'_, str>` from Python strings via `PyStringMethods::to_cow()`. This avoids copying when the Python string is ASCII (zero-copy) and only allocates for non-ASCII.

```rust
// Before: fn tag(&self, tokens: Vec<String>)
// After:  fn tag<'a>(&self, py: Python<'a>, tokens: &Bound<'a, PyList>)
//         -> use py.allow_threads + to_cow for zero-copy
```

**Files:** `tag/sequential.rs` (21 String params), `tag/tnt.rs` (8), `tag/hmm.rs` (4)

---

### B4. Pre-compute `String` return values (memoization)
**Gain:** 30-50% on `__str__`, `__repr__`, `Display` hot paths  
**Effort:** Low (add cached String field to struct, update on mutation)

Many structs call `format!()` or `to_string()` on every `__str__()` call. Cache the result.

```rust
struct Tree {
    label: String,
    children: Vec<TreeNode>,
    #[pyo3(skip)] cached_str: OnceCell<String>,
}
```

**Candidates (6 files):** `tree.rs` (9 format!), `parse.rs` (5 format!), `ccg/mod.rs`, `drt.rs` (13 format!), `sem.rs` (35 format!)

---

## Tier 3: Rust-specific micro-optimizations (low effort, 1.1-1.5x gain)

### C1. `#[inline]` on hot one-liners
**Effort:** Trivial

```rust
#[inline]
fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 { ... }
#[inline]
fn is_atomic(expr: &Expression) -> bool { ... }
#[inline]
fn format_kind(k: &CategoryKind) -> String { ... }
```

---

### C2. `Vec::with_capacity` on every known-size collection
**Effort:** Trivial (search-replace pattern)  
**Remaining:** ~60 sites still use `Vec::new()` where size is statically known

---

### C3. `#![forbid(unsafe_code)]` — for production confidence
**Effort:** Trivial (add to lib.rs)

No `unsafe` anywhere. This is a lint guarantee, not a perf gain. But it prevents accidentally introducing unsound code.

---

### C4. Eliminate `format!(...)` for string concatenation in Display
**Effort:** Low (use `write!` macro or `push_str`)

`format!("{} {}", a, b)` allocates a new String. `write!(f, "{} {}", a, b)` writes directly to formatter.

**Sites:** 137 format!() calls. 35 in sem.rs, 17 in perceptron.rs, 13 in drt.rs.

---

### C5. LTO profile tuning
**Effort:** Trivial

```toml
[profile.release]
lto = "fat"        # was "thin" — fat LTO enables cross-crate inlining
codegen-units = 1
opt-level = 3
panic = "abort"    # smaller binaries, no unwind tables (Python wraps panics)
```

---

### C6. SIMD byte classification for tokenizer character checks
**Effort:** Medium (use `memchr` or `memspan` crate)

The tokok/treebank tokenizers iterate byte-by-byte checking character classes. `memspan` uses runtime-detected SIMD (AVX-512, AVX2, SSE4.2) to skip/classify bytes in chunks.

```rust
// Replace: for (i, ch) in text.char_indices() { if ch == '.' || ... }
// With:    memspan::find_any(text.as_bytes(), b".!?")
```

---

## Tier 4: Code Quality — Professional Rust Standards

Inspired by production crates (tokio, serde, ripgrep, clap, rayon, regex, pyo3 itself).
Every public library should follow these. Some are mechanical, some are design-level.

---

### Q1. Structured error types with `thiserror` → replace 64 raw `PyErr` sites
**Effort:** Medium  
**Pattern from:** `pyo3`, `serde`, `regex`

Currently every error is `PyValueError::new_err(format!(...))` — no structure, no
programmatic handling for callers, no consistent messages.

```rust
// Before (64 identical patterns):
PyValueError::new_err(format!("Expected '->' in grammar line: {line}"))

// After (1 error type, reused everywhere):
#[derive(Debug, thiserror::Error)]
pub enum FastNltkError {
    #[error("invalid grammar: expected '->' in line '{0}'")]
    GrammarParse(String),
    #[error("empty input")]
    EmptyInput,
    #[error("input too long ({0} words, max {1})")]
    InputTooLong(usize, usize),
    #[error("model not trained")]
    NotTrained,
    #[error("invalid category: {0}")]
    InvalidCategory(String),
    #[error("no parse found")]
    NoParse,
}

impl From<FastNltkError> for PyErr {
    fn from(e: FastNltkError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}
```

**Crates to study:** `thiserror` (used by 40,000+ crates), `miette` (fancy diagnostics)

---

### Q2. Clippy `pedantic` + `nursery` lint enablement
**Effort:** Trivial (add to Cargo.toml)  
**Pattern from:** `ripgrep`, `clap`, `serde`

```toml
[lints.clippy]
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
# Allow specific pedantic lints that don't apply
missing_errors_doc = "allow"
module_name_repetitions = "allow"
```

Fixes ~80 warnings, surfaces real issues (needless borrows, implicit hashers, missing safeties).

---

### Q3. `#![warn(missing_docs)]` — enforce doc coverage
**Effort:** Low  
**Pattern from:** `tokio`, `regex`, `rayon`

Every `pub` item must have a doc comment. Currently ~40 undocumented pub items.
Prevents API surface decay as the codebase grows.

```rust
// lib.rs
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
```

---

### Q4. Newtype wrappers for domain concepts
**Effort:** Medium  
**Pattern from:** `serde` (Value, Number), `regex` (Regex, Captures)

Currently raw `String` and `Vec<String>` are used everywhere. Strong typing
prevents bugs and enables future optimization:

```rust
// Before: word_or_tag: String
// After:
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Tag(String);
impl Tag { pub fn as_str(&self) -> &str { &self.0 } }

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Word(String);

// Compiler prevents: fn score(word: Tag, context: &[Word]) — caught at compile time
```

Not required for v0.2 (API compatibility concern with Python FFI), but should plan for v0.3.

---

### Q5. `const`-ify all static data
**Effort:** Low  
**Pattern from:** `regex`, `serde_json`

Many static arrays should be `const` for compile-time evaluation:

```rust
// Before: static BOOSTERS: &[&str] = &[...];
// After:  const BOOSTERS: &[&str] = &[...];
// Before: let mut lex = HashMap::new(); lex.insert("love", 3.2); ...
// After:  use phf::phf_map; static LEXICON: phf::Map<&str, f64> = phf_map! { ... };
```

`phf` crate creates compile-time perfect hash maps — zero runtime init cost,
zero allocation, O(1) lookup for static data like VADER lexicon.

---

### Q6. `unsafe` audit: zero `unsafe` today, lock it in
**Effort:** Trivial  
**Pattern from:** `ripgrep`, `cargo`

```rust
// lib.rs — top of file
#![forbid(unsafe_code)]
```

Currently 0 unsafe blocks anywhere. `forbid` makes this a compiler-enforced
invariant. Critical for a library that processes untrusted text input.

---

### Q7. Test organization: unit tests inline, integration tests in `tests/`
**Effort:** Low  
**Pattern from:** `tokio`, `serde`

Current: ~20 Rust test modules scattered across files. Good pattern already.
Improvements:
- Add `doc = "include_str!(...)"` for README examples that are tested
- Add property-based tests with `proptest` for tokenizer invariants
- Add fuzz targets for `from_string` parsers (CCG, DRS, Tree, Formula)

```rust
// Example: property-based test for tokenizer
proptest! {
    #[test]
    fn tokenizer_never_panics(s in "\\PC*") {
        let _ = word_tokenize(&s);  // must not panic on any input
    }
}
```

---

### Q8. `#[must_use]` on pure functions
**Effort:** Low  
**Pattern from:** `std`, `serde`, `itertools`

Functions that compute a value without side effects should be annotated:

```rust
#[must_use]
pub fn free_variables(&self) -> Vec<String> { ... }

#[must_use]
pub fn productions(&self) -> Vec<String> { ... }
```

Prevents callers from accidentally discarding computed results.

---

### Q9. Eliminate clone-in-loop patterns
**Effort:** Medium  
**Pattern from:** `rustc` performance guidelines

Identified sites cloning in hot loops:

| File | Pattern | Fix |
|---|---|---|
| `ccg/chart.rs:106` | `cat.clone()` per word in lex init | Store `Arc<Category>` |
| `ccg/chart.rs:140-141` | `l.clone(), r.clone()` per combinator match | Borrow from chart, clone only on success |
| `chat.rs:41,53` | `responses[idx].clone()` per call | Return `&str` or `Arc<str>` |
| `classify/maxent.rs:98-100` | `name.clone()`, `label.clone()` per feature | Use `Cow<str>` or entry API better |
| `classify/naivebayes.rs:88,91` | Same pattern | Same fix |

---

### Q10. Consistent module structure: `mod.rs` → module-name file
**Effort:** Low  
**Pattern from:** Rust 2018+ convention, `clap`, `regex`

Currently some modules use `mod.rs` style (tokenize, stem, metrics, classify,
tag, ccg, inference) while others use top-level files. Rust 2018 edition prefers
non-`mod.rs` style:

```
src/tokenize/mod.rs      → src/tokenize.rs
src/tokenize/treebank.rs → src/tokenize/treebank.rs  (child stays)
```

Not urgent but follows modern Rust convention.

---

### Q11. `cargo doc` — ensure docs build without warnings
**Effort:** Low  
**Pattern from:** All top-tier crates

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

Currently would fail on ~40 undocumented public items. Fix by adding docs or
making internal items `pub(crate)`.

---

### Q12. CI-ready: `cargo clippy -- -D warnings`
**Effort:** Low  
**Pattern from:** `tokio`, `serde` CI pipelines

Treat all clippy warnings as errors in CI. Currently 66 warnings pass silently.
Fixing them all makes the build self-documenting about code quality.

---

### Q13. `Cargo.toml` metadata completeness
**Effort:** Trivial  
**Pattern from:** `cargo` publish requirements

```toml
[package]
name = "fastnltk"
version = "0.2.0"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/..."
documentation = "https://docs.rs/fastnltk"
readme = "README.md"
keywords = ["nlp", "nltk", "tokenization", "tagging", "stemming"]  # MISSING
categories = ["text-processing", "science"]                           # MISSING
```

---

### Q14. Fuzz testing for parsers
**Effort:** Medium (add `cargo-fuzz` target)  
**Pattern from:** `regex`, `serde_json`, `url`

Every `from_string` parser (CCG, DRS, Tree, CFG, Formula) should have a fuzz
target that asserts it never panics on arbitrary input.

```rust
// fuzz/fuzz_targets/ccg_parse.rs
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = fastnltk::ccg::parse_category(s);
    }
});
```

---

### Q15. `rkyv` zero-copy deserialization for model loading
**Effort:** Medium  
**Pattern from:** `rkyv` crate (used in gamedev, databases)

Currently Punkt/Perceptron models are loaded via Python pickle → bincode.
`rkyv` supports zero-copy deserialization — the on-disk bytes ARE the in-memory
representation. No allocation during model load.

```rust
// bincode: deserialize(allocates) → HashMap<String, ...>
// rkyv:    access_bytes(zero-copy) → &ArchivedHashMap<String, ...>
```

---

## Implementation Priority Matrix

| # | Optimization | Tier | Est. Gain | Effort |
|---|---|---|---|---|
| C1 | `#[inline]` hot functions | C | 1.05x | 5 min |
| C2 | `Vec::with_capacity` sweep | C | 1.05x | 10 min |
| C5 | LTO=fat, panic=abort | C | 1.1x | 5 min |
| Q2 | Clippy pedantic+nursery | Q | quality | 5 min |
| Q3 | `#![warn(missing_docs)]` | Q | docs | 5 min |
| Q6 | `#![forbid(unsafe_code)]` | Q | safety | 2 min |
| Q11 | `cargo doc` warning-free | Q | docs | 15 min |
| Q12 | CI clippy deny-warnings | Q | quality | 10 min |
| Q13 | Cargo.toml metadata | Q | publish | 5 min |
| Q5 | `const`/`phf` static data | Q | init time | 1 hr |
| Q8 | `#[must_use]` pure fns | Q | safety | 30 min |
| A1 | mimalloc allocator | A | 1.15x | 5 min |
| C4 | `format!` cleanup in Display | C | 1.1x | 30 min |
| B2 | SmallVec for fixed collections | B | 1.2x | 1 hr |
| Q9 | Eliminate clone-in-loop | Q | 1.2x | 2 hr |
| Q1 | `thiserror` error types | Q | quality | 3 hr |
| B1 | Binary search on sorted Vec | B | 1.3x | 2 hr |
| A2 | smol_str for tokens/tags | A | 1.5x | 3 hr |
| B3 | `Cow<str>` in PyO3 FFI | B | 1.3x | 2 hr |
| B4 | Cached Display strings | B | 1.2x | 1 hr |
| A3 | bumpalo arena | A | 2.0x | 5 hr |
| Q4 | Newtype wrappers | Q | quality | 4 hr |
| Q10 | mod.rs → file convention | Q | style | 1 hr |
| A4 | Logos DFA lexer | A | 2.5x | 8 hr |
| C6 | SIMD byte classification | C | 1.3x | 3 hr |
| Q14 | Fuzz targets | Q | safety | 3 hr |
| Q15 | rkyv zero-copy models | Q | load time | 4 hr |
| Q7 | Property-based tests | Q | safety | 3 hr |

---

## Execution Order (recommended)

**Day 1 (immediate):** Q2+Q3+Q6+Q11+Q12+Q13+C1+C2+C5+A1 — quality baseline + 20% perf
**Day 2:** C4+Q5+Q8+Q9+B2 — format cleanup, const data, clone elimination
**Day 3-4:** Q1+A2+B1+B3 — error types, smol_str, sorted Vec, Cow FFI
**Day 5:** B4+Q10 — Display cache, module convention
**Day 6-8:** A3+Q4 — arena allocator + newtype wrappers
**Day 9-12:** A4+C6 — Logos DFA + SIMD (tokenizer rewrite)
**Day 13-15:** Q7+Q14+Q15 — property tests, fuzz, rkyv models

**Projected cumulative speedup:** 3-5x on microbenchmarks.
**Code quality target:** zero clippy warnings, 100% doc coverage, zero unwrap in production paths.
