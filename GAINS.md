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

## Implementation Priority Matrix

| # | Optimization | Tier | Est. Gain | Effort | Benchmark Impact |
|---|---|---|---|---|---|
| C1 | `#[inline]` hot functions | C | 1.05x | 5 min | All modules |
| C2 | `Vec::with_capacity` sweep | C | 1.05x | 10 min | All modules |
| C4 | `format!` cleanup in Display | C | 1.1x | 30 min | sem, drt, perceptron |
| C5 | LTO=fat, panic=abort | C | 1.1x | 5 min | All modules |
| A1 | mimalloc allocator | A | 1.15x | 5 min | All modules |
| B2 | SmallVec for fixed collections | B | 1.2x | 1 hr | ccg, parse, chunk |
| B1 | Binary search on sorted Vec | B | 1.3x | 2 hr | classify, probability |
| A2 | smol_str for tokens/tags | A | 1.5x | 3 hr | tag, tokenize, parse |
| B3 | `Cow<str>` in PyO3 FFI | B | 1.3x | 2 hr | tag (21 String params) |
| B4 | Cached Display strings | B | 1.2x | 1 hr | tree, parse, ccg, drt |
| A3 | bumpalo arena | A | 2.0x | 5 hr | ccg, drt, inference, sem |
| A4 | Logos DFA lexer | A | 2.5x | 8 hr | treebank, toktok, tweet |
| C6 | SIMD byte classification | C | 1.3x | 3 hr | toktok, treebank |
| C3 | forbid(unsafe_code) | C | safety | 2 min | lib.rs |

---

## Execution Order (recommended)

**Hour 1:** C1 + C2 + C3 + C5 + A1 — immediate wins, no refactor  
**Hour 2:** C4 + B2 — format cleanup + SmallVec  
**Hours 3-4:** B1 + B4 — sorted Vec maps + Display cache  
**Hours 5-7:** A2 — smol_str rollout (touches 7 files)  
**Hours 8-9:** B3 — Cow<str> PyO3 FFI  
**Days 3-4:** A3 — bumpalo arena (largest refactor)  
**Days 5-7:** A4 + C6 — Logos DFA + SIMD  

**Projected cumulative speedup:** 3-5x on microbenchmarks (especially tokenizers and taggers).
