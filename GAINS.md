# Performance Gains — fastNLTK

> **Generated:** 2026-07-14  
> **Current baseline:** 272 Rust tests, 236/236 Python tests, 23.3x avg speedup (54 benchmarks)  
> **Completed:** P0 (lto, FxHashMap, Vec capacity), P1 (CCG flat-array, tag hashbrown), P3 (TextTiling LazyLock, tokenizer Lazy, sentiment Vec cap)

---

## Global Optimizations (apply to all modules)

### G1. `Cargo.toml` — Release profile hardening
**Gain:** 10–30% across all benchmarks
```toml
[profile.release]
lto = "thin"           # Link-time optimization for PyO3 (significant gains)
codegen-units = 1       # Single codegen unit = better inlining
opt-level = 3           # Aggressive optimizations (default for release)
panic = "abort"         # Smaller binaries, no unwind tables
```
**Cost:** Longer compile time (~2-3x).  
**Status:** ✅ Already configured (lto="thin", codegen-units=1).
**Bench:** maturin issue #1529 confirms `lto="thin"` is critical for PyO3 extensions.

### G2. HashMap → `FxHashMap` (rustc-hash) everywhere except security-critical
**Gain:** 2–5x faster lookups for string-keyed maps  
**Files:** 21 files use `std::collections::HashMap`, only 2 use hashbrown  
**Status:** ✅ Done for hot-path modules (ccg, texttiling, mwe, sequential tagger). Remaining 15 files use HashMap for IntoPy returns (can't convert).
**Already done:** `collocations.rs` uses `hashbrown::HashMap`  
**Priority files:** `ccg/lexicon.rs`, `classify/maxent.rs`, `classify/naivebayes.rs`, `tag/sequential.rs`, `parse.rs`, `inference/`  
**Caveat:** FxHash can produce collisions on adversarial input. Safe for NLP data.

### G3. `Vec::new()` → `Vec::with_capacity(n)` where size is known
**Gain:** Eliminates reallocation chains in hot loops  
**Files:** 27 files use `Vec::new()` without `with_capacity` in non-test code  
**Status:** ✅ Done in hot paths (sentiment, HMM, TextTiling). Remaining files are cold paths.
**Pattern:** Every `collect()` chain, every `push`-in-loop, every JSON/serde deserialize.

### G4. `String.clone()` → `&str` or `Arc<str>` or `Cow<'_, str>`
**Gain:** Up to 94% allocation reduction (real-world Rust services)  
**Files:** 234 `.clone()` calls across 57 files  
**Worst offenders:** `tag/sequential.rs` (48 clones), `sem.rs` (28), `parse.rs` (22), `ccg/chart.rs` (12), `inference/discourse.rs` (12)  
**Technique:** Use `Into<String>` or `AsRef<str>` in internal APIs, only clone at FFI boundary.

### G5. `format!()` → `write!()` / `fmt::Write` / string builder
**Gain:** Avoids intermediate `String` allocation per `format!` call  
**Files:** 132 `format!()` calls across 57 files  
**Worst offenders:** `sem.rs` (35 formats), `tag/perceptron.rs` (17), `drt.rs` (13)

### G6. `lazy_static` / `LazyLock` for all Regex
**Gain:** Regex compilation is 50–200µs; compiling in hot path kills tokenizer speed  
**Already done:** `texttiling.rs` uses `LazyLock`  
**Status:** ✅ Done (treebank.rs confirmed already Lazy, texttiling uses LazyLock).

### G7. `rayon` parallel iterators for batch operations
**Gain:** Near-linear scaling to core count for independent work  
**Candidates:** `word_tokenize`, `tag_sents`, `stem` on word lists, `sent_tokenize` on paragraph lists  
**Files:** `tokenize/mod.rs`, `tag/mod.rs`, `stem/*.rs`, `parse.rs`

### G8. String interning with `lasso` or `string_cache`
**Gain:** Tags ("NN", "VBZ", "DT") and frequent words ("the", "a") cloned thousands of times  
**Technique:** Intern all tag strings and frequent words → compare via `usize` IDs instead of `String`  
**Candidates:** `tag/sequential.rs`, `tag/perceptron.rs`, `chunk.rs`, `classify/`

### G9. Small-string optimization with `compact_str` or `smol_str`
**Gain:** 24 bytes inline stack storage → zero heap allocation for short strings  
**Candidates:** Tags (≤5 chars), word tokens (avg 5–8 chars English), category names

---

## Per-Module Optimizations

### `src/sem.rs` (1041 LoC, 28 clones, 35 formats, 19 unwraps)
**Current:** `Box<Expression>` enum tree with lots of cloning in substitution/display/free_vars.

| # | Technique | Gain | Effort |
|---|---|---|---|
| S1 | Replace `Box<Expression>` with `Rc<Expression>` | Eliminates deep clones in substitution, NNF conversion, free_vars collection | Medium |
| S2 | `Display` use `fmt::Write` instead of format! chaining | 35 fewer allocations per `to_string()` | Low |
| S3 | Intern variable/constant names via `lasso` | `Variable("x")` and `Constant("john")` cloned repeatedly | Medium |
| S4 | `free_variables` return `Vec<&str>` instead of `Vec<String>` | Avoids cloning every variable name | Low |
| S5 | Cache parsed `Expression` with `LruCache<String, Rc<Expression>>` | Parser is slow; most inputs repeat | Low |

**Estimated gain:** 2–4x on `from_string`, 3–5x on `substitute`/`simplify`.

---

### `src/tag/sequential.rs` (527 LoC, 48 clones) — CLONE HOTSPOT
**Current:** Ngram taggers clone entire training corpora into HashMap during `.train()`.

| # | Technique | Gain | Effort |
|---|---|---|---|
| T1 | Training: use `&[(String, String)]` slices instead of cloning each sentence | Cuts 48 clones during train; major allocation reduction | Medium |
| T2 | `HashMap<String, HashMap<String, u64>>` → `FxHashMap<&str, FxHashMap<&str, u64>>` | Faster hash + string interning | Medium |
| T3 | Backoff tagger chains: `Arc<TaggerI>` instead of `Box<dyn TaggerI>` | Shared ownership, cheap clone | Low |

**Estimated gain:** 30–50% faster `.train()`, 10–20% faster `.tag()`.

---

### `src/lm.rs` (514 LoC, 6 clones, 10 unwraps)
**Current:** Mostly delegates to `rustling`. Lightweight wrapper.

| # | Technique | Gain | Effort |
|---|---|---|---|
| L1 | `generate()` pre-allocate output `Vec` with `with_capacity` | One fewer reallocation per call | Low |
| L2 | `score()` avoid `unwrap_or(0.0)` → proper error types | No perf gain; correctness | Low |

**Estimated gain:** Negligible (wrapper is thin). Main perf is in `rustling`.

---

### `src/drt.rs` (486 LoC, 5 clones, 13 formats)
**Current:** DRS with `Box<DRS>` recursion, JSON serde for Python interop.

| # | Technique | Gain | Effort |
|---|---|---|---|
| D1 | `Rc<DRS>` instead of `Box<DRS>` | Sharing in merge/resolve, less cloning | Medium |
| D2 | `serde_json` → manual `to_string` for DRS | `serde_json::to_string` is expensive; custom serializer faster | Low |
| D3 | Intern predicate strings (dog, cat, bone, runs, etc.) | Repeated predicates in discourse | Low |

**Estimated gain:** 2–3x on `merge()`, 1.5x on `answer_question()`.

---

### `src/parse.rs` (453 LoC, 22 clones)
**Current:** CFG + Earley chart parser. Clones productions and nonterminal strings heavily.

| # | Technique | Gain | Effort |
|---|---|---|---|
| P1 | Intern nonterminal symbols as `usize` IDs | `HashMap<String, Vec<usize>>` → `HashMap<usize, Vec<usize>>` | Medium |
| P2 | Earley chart: use `ArrayVec` for small spans | Avoid heap alloc for 1–3 state entries | Low |
| P3 | `from_string` pre-allocate `productions` with known line count | Fewer reallocations | Low |

**Estimated gain:** 1.5–2x on parse.

---

### `src/ccg/chart.rs` (439 LoC, 12 clones, 10 unwraps)
**Current:** CKY chart as `HashMap<(usize, usize), Vec<CCGEdge>>`.

| # | Technique | Gain | Effort |
|---|---|---|---|
| C1 | Replace `HashMap<(usize, usize), Vec<CCGEdge>>` with `Vec<Vec<Vec<CCGEdge>>>` (flat 3D array) | Hash + tuple key overhead eliminated; O(1) index-based access | Medium |
| C2 | `CCGEdge.cat` clone → shared `Rc<Category>` | Each edge clones Category during combination | Low |
| C3 | Pre-allocate chart capacity: `n * n` cells max | Avoids HashMap growth during parse | Low |

**Estimated gain:** 30–50% faster parse for medium sentences.

---

### `src/ccg/combinator.rs` (186 LoC)
**Current:** Returns `Option<CategoryKind>` with `*result.clone()`.

| # | Technique | Gain | Effort |
|---|---|---|---|
| CB1 | Return `&CategoryKind` instead of `CategoryKind` | Avoids clone in apply; caller decides to clone or not | Low |

**Estimated gain:** Fewer allocations per combination attempt.

---

### `src/classify/maxent.rs` (359 LoC) + `naivebayes.rs` (351 LoC)
**Current:** GIS training loop, frequent HashMap lookups for feature weights.

| # | Technique | Gain | Effort |
|---|---|---|---|
| CL1 | `HashMap<String, u64>` → `FxHashMap<&str, u64>` or interned keys | 2–5x faster lookups in training loop | Medium |
| CL2 | GIS iteration: pre-compute feature indices as `Vec<usize>` | Avoids string-based lookup per feature per iteration | Medium |
| CL3 | `prob_classify`: pre-allocate scores HashMap with `with_capacity(labels.len())` | Fewer reallocations | Low |

**Estimated gain:** 1.5–2x faster `.train()`, 20% faster `.classify()`.

---

### `src/tree.rs` (402 LoC, 8 clones, 9 formats)
**Current:** Already partially optimized. Remaining issues:

| # | Technique | Gain | Effort |
|---|---|---|---|
| TR1 | `parse_brackets` use `&str` slices instead of `String` for labels/words | Avoid clone of every label | Medium |
| TR2 | `collect_productions` use `Vec<&str>` intermediate | Avoid String join allocation | Low |

**Estimated gain:** 1.2–1.5x on `from_string`.

---

### `src/tokenize/punkt.rs` (407 LoC) — BIGGEST TOKENIZER
**Current:** Punkt sentence tokenizer with annotation/orthography cache.

| # | Technique | Gain | Effort |
|---|---|---|---|
| PT1 | `HashMap<String, OrthoContext>` → `FxHashMap<InternedStr, OrthoContext>` | Faster lookup for every word during training/predict | Medium |
| PT2 | Candidate boundary: use slices instead of String for context windows | Less allocation during boundary detection | Low |
| PT3 | Parallel training: `rayon` over training sentences | Multi-core speedup for `.train()` | Low |

**Estimated gain:** 1.3–2x on `sent_tokenize`.

---

### `src/tokenize/treebank.rs` (220 LoC)
**Current:** Regex-based Treebank tokenizer. Multiple regex passes.

| # | Technique | Gain | Effort |
|---|---|---|---|
| TK1 | Compile all regexes as `LazyLock` statics (currently recompiled per invocation) | Eliminates ~50µs regex compile per call | Low |
| TK2 | Combine regex passes into single pass where possible | Fewer iterations over text | Medium |

**Estimated gain:** 1.5–2x for small inputs, 1.1x for large (compile cost amortized).

---

### `src/tokenize/regexp.rs` (227 LoC)
**Current:** Wraps user-provided regex. Can't optimize user patterns but can optimize wrapper.

| # | Technique | Gain | Effort |
|---|---|---|---|
| RX1 | Parse regex once at construction, not per `.tokenize()` | User's regex cloned into wrapper | Low |

**Estimated gain:** Minor (Python user provides regex; overhead is in regex engine, not wrapper).

---

### `src/inference/tableau.rs` (361 LoC) + `resolution.rs` (261 LoC)
**Current:** Theorem provers with formula cloning. Resolution prover already partially optimized.

| # | Technique | Gain | Effort |
|---|---|---|---|
| I1 | Arena allocation (`bumpalo`) for temporary Formula/Clause objects during proof search | Eliminates `Box`/`Vec` alloc overhead in search loop | Medium |
| I2 | Literal dedup: use `BTreeSet<Literal>` instead of manual sort+dedup | Ord already implemented, cleaner API | Low |
| I3 | `nonmonotonic.rs`: use `bit-set` for extension enumeration | Avoids cloning rule sets for each extension | Medium |

**Estimated gain:** 2–5x on `DefaultReasoner.extensions()` (currently 55ms for 10 rules).

---

### `src/chunk.rs` (304 LoC, 7 formats, 11 unwraps)
**Current:** Regexp chunk parser with IOB tagging.

| # | Technique | Gain | Effort |
|---|---|---|---|
| CH1 | Compile tag regexes at construction, not per sentence | `compile_tag_pattern` called every sentence | Low |
| CH2 | Use `Vec<(String, String)>` pre-allocation | Known number of tokens per sentence | Low |

**Estimated gain:** 1.5–2x on `parse()`.

---

### `src/cluster.rs` (238 LoC, 5 clones, 6 unwraps)
**Current:** K-means with Euclidean distance.

| # | Technique | Gain | Effort |
|---|---|---|---|
| KU1 | Use `nalgebra` or `ndarray` for vector math | SIMD-accelerated distance computations | Medium |
| KU2 | `classify()` pre-compute `norm_sq` for each centroid | Avoids recomputing in each classification | Low |

**Estimated gain:** 2–3x for large dimensional clusters.

---

### `src/sentiment.rs` (225 LoC)
**Current:** VADER sentiment analyzer with built-in lexicon.

| # | Technique | Gain | Effort |
|---|---|---|---|
| SN1 | Build lexicon as `FxHashMap<&'static str, f64>` instead of `HashMap<String, f64>` | Avoid String → lookup cost for every word | Low |

**Estimated gain:** 1.5x on `polarity_scores()`.

---

### `src/collocations.rs` (276 LoC)
**Current:** Already uses `hashbrown::HashMap`. Good.

| # | Technique | Gain | Effort |
|---|---|---|---|
| CO1 | Pre-allocate ngram_fd with estimated capacity (vocab size ^ n) | Fewer reallocations during construction | Low |
| CO2 | Use `rayon` for scoring multiple measures in parallel | Multi-core score computation | Low |

**Estimated gain:** 20% on `.score_ngrams()`.

---

### `src/probability.rs` (489 LoC) — CLONE HOTSPOT
**Current:** FreqDist, ConditionalFreqDist. 11 clones.

| # | Technique | Gain | Effort |
|---|---|---|---|
| PR1 | `update()` incrementally instead of replacing HashMap | Fewer allocs when adding to existing distribution | Low |
| PR2 | `FreqDist` use `FxHashMap` with compound keys | Faster lookup | Low |

**Estimated gain:** 20% on `update()`.

---

### `src/metrics/` (493 LoC total, 7 files)
**Current:** Already fast (104x windowdiff, 50x edit_distance). Near optimal.

| # | Technique | Gain | Effort |
|---|---|---|---|
| M1 | `jaro.rs`: eliminate temporary Vec in `find_matching_chars` | Single-pass matching | Low |

**Estimated gain:** Negligible (already sub-millisecond).

---

## Priority Matrix

| Priority | Module | Technique | Est. Gain | Effort | File |
|---|---|---|---|---|---|
| 🔴 P0 | Global | `lto="thin"` + `codegen-units=1` | **15–30%** | Trivial | Cargo.toml |
| 🔴 P0 | All HashMap | `std::HashMap` → `FxHashMap` | **2–5x** lookup | Low | 19 files |
| 🔴 P0 | All `Vec::new()` | → `Vec::with_capacity(n)` | **Cut reallocs** | Low | 27 files |
| 🟠 P1 | `tag/sequential.rs` | Training: pass by ref, not clone | **30–50%** train | Medium | tag/sequential.rs |
| 🟠 P1 | `sem.rs` | `Rc<Expression>` + intern vars | **2–4x** | Medium | sem.rs |
| 🟠 P1 | `ccg/chart.rs` | Flat 3D array chart | **30–50%** parse | Medium | ccg/chart.rs |
| 🟠 P1 | `parse.rs` | Intern nonterminals | **1.5–2x** | Medium | parse.rs |
| 🟡 P2 | `inference/nonmonotonic.rs` | Arena + bit-set extensions | **2–5x** | Medium | inference/ |
| 🟡 P2 | `classify/` | FxHashMap + pre-index features | **1.5–2x** train | Medium | classify/ |
| 🟡 P2 | `drt.rs` | `Rc<DRS>` + custom serializer | **2–3x** | Medium | drt.rs |
| 🟡 P2 | `tokenize/punkt.rs` | FxHashMap for OrthoContext | **1.3–2x** | Medium | tokenize/punkt.rs |
| 🟢 P3 | All `format!()` | → `write!()` / fmt::Write | **Fewer allocs** | Low | 132 sites |
| 🟢 P3 | `tokenize/treebank.rs` | LazyLock regex | **1.5–2x** small | Low | tokenize/treebank.rs |
| 🟢 P3 | `cluster.rs` | SIMD distance with ndarray | **2–3x** | High | cluster.rs |
| 🟢 P3 | `rayon` parallel batch ops | `par_iter` for tokenize/tag | **2–8x** batch | Low | tokenize/, tag/ |
| 🟢 P3 | String interning | `lasso` for tags + frequent words | **Memory 50%↓** | High | tag/, chunk/, tokenize/ |

---

## Implementation Order (recommended)

**Week 1:** P0 items — Cargo.toml, FxHashMap, Vec::with_capacity  
**Week 2:** P1 items — tag clone reduction, sem Rc, CCG flat chart, parse intern  
**Week 3:** P2 items — inference arena, classify FxHashMap, DRT Rc, Punkt  
**Week 4:** P3 items — format! cleanup, lazy regex, rayon, string interning  

**Projected cumulative speedup:** 2–5x on microbenchmarks, 1.5–3x on real-world workloads.
