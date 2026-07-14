# Code Quality Review — fastNLTK v0.2.0

Comprehensive code quality sweep against major Rust project standards (tokio, serde, ripgrep).
Generated 2026-07-13.

---

## Summary

| Metric | Current | Target | Gap |
|---|---|---|---|
| clippy (deny-all) | 612 errors | 0 | 612 issues across 18 categories |
| `panic!` / `unwrap()` (prod) | 2 / ~15 | 0 | 2 panics, ~15 unwraps in non-test paths |
| `unsafe` | 0 | 0 | ✅ |
| Doc coverage (pub items) | 219 pub, ~7 under-documented files | 100% | 7 files need docs |
| `#[allow(...)]` suppressor count | 6 | 0 | 6 lints suppressed |
| Clone calls (non-test) | 252 | <50 | High alloc surface |
| `as` lossy casts | 7 | 0 | Use `.into()` / `f64::from()` |
| `Box<dyn Error>` / erased errors | 0 | 0 | ✅ (FastNltkError) |
| TODO/FIXME/HACK markers | 0 | 0 | ✅ |
| Max file size | sem.rs 1027 lines | <500 | 5 files >500 lines |

---

## Critical Issues

### C1. `panic!()` in production paths (2 sites)

Panics in library code are unacceptable — they crash the Python interpreter without recovery.

| File | Line | Code | Fix |
|---|---|---|---|
| `src/drt.rs` | 419 | `_ => panic!("Expected negation")` | Return `Err(FastNltkError::Parse(...))` |
| `src/tokenize/toktok.rs` | 8 | `unwrap_or_else(\|_\| panic!("bad regex: {p}"))` | Return `Result` from constructor or use `expect` with message |

**Fix cost**: 30 min. Both in constructor/parse paths that should propagate errors.

### C2. Production `.unwrap()` without fallback (15+ sites)

Unwraps in non-test code paths that should return `Result`:

| Category | Count | Example | File |
|---|---|---|---|
| PyO3 `.extract()` | 6 | `key.extract().unwrap_or_default()` | naivebayes.rs, maxent.rs |
| Regex compile | 3 | `Regex::new(p).unwrap()` | chunk.rs, toktok.rs |
| Parser constructors | 4 | `CCGLexicon::new(...).unwrap()` | ccg/lexicon.rs |
| Array indexing | 1 | `labels.iter().position(...).unwrap()` | maxent.rs:130 |
| General | 2 | `parse_expression(...).unwrap()` | sem.rs tests (true test code, ok) |

**Fix cost**: 2-3 hours. Most are in `#[pymethods]` constructors — add `PyResult<T>` return types.

### C3. `#[allow(...)]` suppressing real issues (6 sites)

Suppressed lints hide code quality problems:

| File | Line | Suppression | Issue |
|---|---|---|---|
| `src/collocations.rs` | 45 | `#[allow(dead_code)]` | Dead code not removed |
| `src/tokenize/punkt.rs` | 26 | `#[allow(dead_code)]` | Dead field |
| `src/tokenize/punkt.rs` | 47 | `#[allow(dead_code)]` | Dead field |
| `src/tokenize/punkt.rs` | 71 | `#[allow(unused_variables)]` | Unused parameter |
| `src/probability.rs` | 51, 57, 232 | `#[allow(non_snake_case)]` | Python-named structs — use `#[pyclass(name = "...")]` instead |
| `src/sentiment.rs` | 170 | `#[allow(unused_mut)]` | Remove `mut` binding |

**Fix cost**: 1 hour. `dead_code` removals may be breaking changes — audit first.

---

## High Priority

### H1. Clippy warnings breakdown (612 total)

| Warning | Count (est) | Category | Auto-fix? |
|---|---|---|---|
| `doc_markdown` (missing backticks) | ~100 | pedantic | ✅ `--fix` |
| `use_self` (struct name repetition) | ~100 | nursery | ✅ `--fix` |
| `unused_self` (method never uses &self) | ~15 | nursery | Manual |
| `if_same_then_else` (identical branches) | ~5 | style | ✅ `--fix` |
| `collapsible_match / _if` | ~8 | style | ✅ `--fix` |
| `needless_range_loop` | ~5 | style | Manual |
| `cast_sign_loss` / `cast_possible_wrap` | ~8 | pedantic | Manual |
| `uninlined_format_args` | ~3 | style | ✅ `--fix` |
| `suboptimal_flops` (log2, mul_add) | ~3 | pedantic | ✅ `--fix` |
| `clone_on_copy` / `clone_from` | ~3 | perf | ✅ `--fix` |
| `redundant_closure_for_method_calls` | ~5 | style | ✅ `--fix` |
| `single_char_pattern` | ~3 | style | ✅ `--fix` |
| `unnecessary_wraps` (unused Result) | ~5 | style | ✅ `--fix` |
| Other (cast_lossless, manual_let_else, etc.) | ~150 | various | Mixed |

**Auto-fixable**: ~250 warnings (`cargo clippy --fix`). Remaining 362 need manual triage.

**Fix cost**: 4-6 hours for manual fixes, 5 min for auto-fix passes.

### H2. Documentation gap (7 files)

Files with pub API but near-zero doc coverage:

| File | Pub Items | Doc Lines | Severity |
|---|---|---|---|
| `src/lib.rs` | 23 | 1 | **Critical** — crate entry point |
| `src/stem/mod.rs` | 10 | 0 | High — 9 stemmer re-exports |
| `src/inference/mod.rs` | 6 | 1 | High |
| `src/tag/mod.rs` | 5 | 1 | Medium |
| `src/classify/mod.rs` | 4 | 0 | Medium |
| `src/collocations.rs` | 4 | 1 | Medium |
| `src/stem/arlstem.rs` | 3 | 0 | Low |

**Fix cost**: 3-4 hours. Add module-level `//!` docs + `///` on pub items.

### H3. Lossy `as` casts (7 production sites)

Numeric casts that silently truncate:

| File | Line | Code | Fix |
|---|---|---|---|
| `src/probability.rs` | 68 | `count as f64 / self.total as f64` | `f64::from(count) / f64::from(self.total)` |
| `src/probability.rs` | 284 | `as f64 / as f64` | `f64::from()` |
| `src/tag/sequential.rs` | 141 | `correct as f64 / total as f64` | `f64::from()` |
| `src/tag/tnt.rs` | 235 | `tri_count as f64` | `f64::from()` |
| `src/tag/tnt.rs` | 240 | `bi_count as f64` | `f64::from()` |
| `src/tag/tnt.rs` | 258 | `count as f64` | `f64::from()` |
| `src/tokenize/treebank.rs` | 91, 93 | `as isize / as usize` | `try_into().unwrap()` with error context |

**Fix cost**: 30 min. Replace with `From`/`TryFrom` impls.

### H4. Clone-heavy code paths (252 sites)

252 `.clone()` calls in non-test code. Hotspots:

| File | Appx. Sites | Concern |
|---|---|---|
| `src/probability.rs` | ~20 | FreqDist cloning labels in loops |
| `src/tree.rs` | ~30 | Tree node cloning for subtrees |
| `src/tag/sequential.rs` | ~25 | Tag/token cloning in backoff chain |
| `src/sem.rs` | ~20 | Expression cloning in substitution |
| `src/lm.rs` | ~15 | Context cloning in n-gram lookups |

**Fix cost**: 1-2 days. Focus on:
- Replace `String::clone()` with `&str` borrowing where possible
- Replace `Vec::clone()` with `Cow<[T]>` or `Arc<[T]>`
- Replace `Tree::clone()` with `Rc<Tree>` (partial: already done in CCG chart)

### H5. Large files need splitting (>500 lines)

| File | Lines | Fn Count | Action |
|---|---|---|---|
| `src/sem.rs` | 1027 | 43 | Split into `parse.rs`, `evaluate.rs`, `model.rs` |
| `src/tag/sequential.rs` | 502 | 24 | Extract backoff chain into `backoff.rs` |
| `src/lm.rs` | 501 | 49 | Split into `mle.rs`, `lidstone.rs`, `witten_bell.rs` |
| `src/drt.rs` | 471 | - | Split `syntax`/`semantics` |
| `src/probability.rs` | 463 | 52 | Split by distribution type |

**Fix cost**: 2-3 days. Public API must remain stable.

---

## Medium Priority

### M1. `unused_self` — 15+ stateless methods (nursery)

Methods that take `&self` but never use it. Should be associated functions:

```rust
// Before
impl Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> { ... }  // never uses self
}
// After  
impl Tokenizer {
    fn tokenize(text: &str) -> Vec<String> { ... }
}
```

**Affected**: treebank.rs, toktok.rs, regexp.rs, simple.rs, punkt.rs, tnt.rs.

**Fix cost**: 1 hour. Python FFI methods (`#[pymethods]`) may require `&self` for PyO3 — those are exceptions.

### M2. `needless_range_loop` (style)

5 index-based loops when `enumerate()` would work:

| File | Approx count |
|---|---|
| `src/tag/tnt.rs` | 3 |
| `src/tokenize/mwe.rs` | 1 |
| `src/tokenize/punkt.rs` | 1 |

**Fix cost**: 15 min.

### M3. `if_same_then_else` — dead code branches (style)

| File | Line | Issue |
|---|---|---|
| `src/tokenize/punkt.rs` | 162 | `if bytes[i] == b' ' { i } else { i }` — both branches identical |
| `src/tokenize/treebank.rs` | 124-136 | 3 if branches push same token |

**Fix cost**: 10 min.

### M4. `cast_sign_loss` / `cast_possible_wrap` (pedantic)

isize↔usize conversions in TLS path (tnt.rs, tree.rs):

```rust
// Before
let tag_prev = &self.tags[back[i - 1][k] as usize];
let best_k = k as isize;

// After
let tag_prev = &self.tags[usize::try_from(back[i - 1][k]).expect("valid index")];
```

**Fix cost**: 45 min. Add bounds checks or `try_into().expect("...")` with context.

### M5. Uninspected PyO3 `.extract()` (correctness risk)

naivebayes.rs and maxent.rs use `key.extract().unwrap_or_default()` — silently returns empty string/String on type mismatch:

```rust
// Before: silently ignores type errors
let k: String = key.extract().unwrap_or_default();

// After: propagate error
let k: String = key.extract()?;
```

**Fix cost**: 1 hour. Change method signatures from `-> Vec<...>` to `-> PyResult<Vec<...>>`.

---

## Low Priority

### L1. `suboptimal_flops` (pedantic)

| File | Fix |
|---|---|
| `src/tokenize/texttiling.rs:129` | `lf.log(2.0)` → `lf.log2()` |
| `src/tokenize/texttiling.rs:130` | `rf.log(2.0)` → `rf.log2()` |
| `src/tag/tnt.rs:216` | `a + b * c` → `b.mul_add(c, a)` |

**Fix cost**: 5 min.

### L2. `uninlined_format_args` (style)

| File | Fix |
|---|---|
| `src/tokenize/punkt.rs:374` | `format!("{:?}", x)` → `format!("{x:?}")` |
| `src/tree.rs:214` | `format!("'{}'", s)` → `format!("'{s}'")` |

**Fix cost**: 2 min.

### L3. `clone_from` missed optimization (perf)

`src/tag/sequential.rs:269,300`: Use `clone_from()` instead of `= x.clone()`:
```rust
// Before
prev1 = tag.clone();
// After
prev1.clone_from(tag);
```

**Fix cost**: 5 min.

### L4. Magic numbers (nursery)

Constants without named bindings:

| File | Example |
|---|---|
| `src/lm.rs:45` | `unwrap_or(0.0)` — repeated magic defaults |
| `src/cluster.rs:84` | `vec![0.0; dim]` — zero-initialization |
| `src/corpus.rs` | Hard-coded resource paths |

**Fix cost**: 1 hour. Extract named constants.

---

## Architecture Issues

### A1. Tightly-coupled `sem.rs` (1027 lines, 43 functions)

sem.rs handles parsing + evaluation + model checking + substitution + unification in one file. Should decompose:

```
src/sem/
  mod.rs        — re-exports, register_module
  parse.rs      — Expression parser, Token enum, parser combinators
  evaluate.rs   — model_check, evaluate, satisfy
  transform.rs  — substitute, alpha_convert, free_variables, replace
  model.rs      — Model, Assignment, Valuation types
```

### A2. Backoff chain in `tag/sequential.rs`

Sequential backoff tagger stores taggers in a `Vec`, but selector logic is inline. Extract into a `BackoffTagger` trait + chain struct:

```rust
trait BackoffTagger {
    fn choose_tag(&self, tokens: &[String], i: usize, history: &[String]) -> Option<String>;
}

struct BackoffChain {
    taggers: Vec<Box<dyn BackoffTagger>>,
}
```

### A3. Probability distributions as giant enum

`src/probability.rs` (463 lines, 52 functions): All distributions in one file. Each could be its own module under `src/probability/`:

```
src/probability/
  mod.rs
  freqdist.rs    — FreqDist
  conditional.rs — ConditionalFreqDist
  lidstone.rs
  laplace.rs
  wittenbell.rs  — (move from lm.rs)
  mle.rs
  elephant.rs
  eager.rs
```

### A4. Box/Rc/Arc usage (28 sites)

28 pointer types. Most justified (Py<T>, recursive Tree, Rc<CCGEdge>). Audit for unnecessary boxing:

| Type | Where | Verdict |
|---|---|---|
| `Rc<CCGEdge>` | ccg/chart.rs | ✅ justified (cloning-heavy) |
| `Box<Formula>` | drt.rs | ✅ justified (recursive enum) |
| `Py<T>` | tree.rs, various | ✅ required by PyO3 |
| `LazyLock<Regex>` | regex_cache.rs, treebank.rs | ✅ compile-once |

---

## Test Coverage Gaps

### G1. Untested public API

Functions with zero dedicated tests:

| Function | File | Reason |
|---|---|---|
| `find_resource` | data.rs | I/O-dependent, hard to test |
| `find_resource_dir` | data.rs | I/O-dependent |
| `model_evaluate` | sem.rs | Complex setup |
| `register_module` | 15+ files | PyO3 boilerplate (acceptable) |
| `tokenize_simple_sentences` | punkt.rs | Private helper, tested via public API |

### G2. Missing error-path coverage

- `CCGLexicon::new()` — no test for invalid category strings
- `RegexpParser::new()` — no test for malformed grammar
- `parse_category("")` — no test for empty string edge case
- `DRS::from_string` — no test for malformed DRS syntax

---

## Fix Plan (Estimated Effort)

### Quick Wins (< 1 day total)
1. **C1**: Fix 2 `panic!()` → `Err(...)` — 30 min
2. **C3**: Remove `#[allow(...)]` suppressors — 1 hour
3. **H3**: Replace `as` casts with `From` — 30 min
4. **L1-L3**: Flops, format_args, clone_from — 10 min
5. **M3**: Dead if branches — 10 min

### Significant Effort (2-3 days)
6. **H1**: Fix 350+ manual clippy warnings — 4-6 hours
7. **H2**: Doc all pub API — 3-4 hours
8. **C2**: Replace unwrap with Result in constructors — 2-3 hours
9. **M1**: Convert unused_self methods — 1 hour
10. **M4**: Safe cast conversions — 45 min
11. **M5**: PyO3 extract error propagation — 1 hour
12. **L4**: Extract magic numbers — 1 hour

### Deep Refactors (5-7 days)
13. **H4**: Reduce clone surface — 1-2 days
14. **H5**: Split large files — 2-3 days
15. **A1-A4**: Architecture decomposition — 2-3 days

### Test Improvements (1-2 days)
16. **G1**: Expand test coverage — 1 day
17. **G2**: Error-path tests — 1 day

---

## Summary

v0.2.0 is functional and correct (275 Rust + 254 Python tests pass). For production-readiness matching major Rust crates:

1. **Fix 2 panics** — crashes are unacceptable
2. **Fix 612 clippy warnings** — 250 auto-fixable, 362 manual
3. **Document 7 under-documented files** — especially lib.rs (23 pub items)
4. **Replace 15 `unwrap()` in constructors** — return Result instead
5. **Remove 6 `#[allow]` suppressors** — fix the underlying issue
6. **Split 5 files >500 lines** — sem.rs (1027) first
7. **Reduce 252 clone calls** — profile-guided
8. **Add error-path tests** — 100% coverage on unwrap paths
