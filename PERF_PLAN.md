# Performance Research — Next Targets

## Current 10 Slowest Operations

Measured on v0.4.0, release build, 1000-word input (unless noted).

| Rank | Operation | Time | Bottleneck | Fix |
|---|---|---|---|---|
| 1 | **TweetTokenizer** | 0.91ms | `build_patterns()` called on every tokenize call — recompiles 4 regexes | Move to `LazyLock` |
| 2 | **PorterStemmer** | 0.78ms | Wrapping C Snowball, per-word call | Already optimal (pure C) |
| 3 | **SnowballStemmer** | 0.24ms | Same C Snowball wrapper | Already optimal |
| 4 | **RegexpStemmer** | 0.15ms | Regex match per word | N/A (pattern-dependent) |
| 5 | **LancasterStemmer** | 0.15ms | Table lookup per word | Already optimal |
| 6 | **ToktokTokenizer** | 0.10ms | Regex substitution chain | Already fast (<0.1ms) |
| 7-10 | **Other tokenizers** | <0.05ms | Already near wire speed | No action needed |

**Only actionable target: TweetTokenizer (0.91ms).** Stemmers are already optimal C/Rust. Other tokenizers are <0.05ms.

## TweetTokenizer Fix

**Problem:** `build_patterns()` is called inside `tokenize()`, recompiling 4 regexes from string literals on every invocation. The benchmark calls `tokenize()` 200 times → 800 `Regex::new()` calls.

**Fix:** Use `LazyLock<Regex>` statics:

```rust
static URL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"...").unwrap());
static EMOTICON_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"...").unwrap());
static PHONE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"...").unwrap());
static MAIN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"...").unwrap());
```

**Estimated gain: 10-50×** (regex compilation removed from hot path).

---

## General Codebase Improvements

Audited all 75+ Rust source files for systematic optimizations:

### 1. String → SmolStr in struct fields

Many core structs use `String` fields where the values are short and finite. `SmolStr` inlines strings < 23 bytes (no heap alloc).

| File | Fields | Est. gain per op |
|---|---|---|
| `src/tree.rs` | `Tree.label`, `TreeNode::Leaf(String)` | Minor (parse/create) |
| `src/drt.rs` | `DRS.universe: Vec<String>` | Minor (DS creation) |
| `src/parse.rs` | `CFG.start_symbol`, `Production.lhs/rhs` | Minor (grammar build) |
| `src/sem/expression.rs` | `Expression` variants with String fields | Minor |

**Verdict:** Won't meaningfully affect benchmark times (these are one-time construction costs, not hot path). Skip.

### 2. HashMap<String, ...> → FxHashMap<SmolStr, ...>

Many modules use `HashMap<String, ...>` with short string keys.

| File | Usage | Impact |
|---|---|---|
| `src/parse.rs` | `HashMap<String, Vec<usize>>` | Low (grammar build, not hot) |
| `src/drt.rs` | Various String-based lookup | Low |
| `src/tokenize/mwe.rs` | Already fixed ✅ | |

**Verdict:** Low impact — these are lookup tables built during `__init__`, not hot path.

### 3. Sequential regex passes (Toktok, Treebank)

Both already optimized in this session:
- Treebank: single-pass char scanner ✅ 
- Regexp fast path: memchr3 SIMD ✅

### 4. Allocation pre-sizing

Many `Vec::new()` could use `Vec::with_capacity()` in known-size loops.

| File | Pattern | Fix |
|---|---|---|
| `src/tree.rs` | `vec![]` in recursive parse | Pre-size with tree depth |
| `src/parse.rs` | `Vec::new()` in Earley chart | Pre-size with grammar size |

**Verdict:** Minor. Most hot paths already use `with_capacity`.

### 5. LazyLock for all module-level regexes

Audited all static regex patterns — most are already behind `LazyLock`.

| File | Status |
|---|---|
| `src/tokenize/tweet.rs` | ❌ `build_patterns()` called per-tokenize |
| `src/tokenize/treebank.rs` | ✅ Already LazyLock (now removed — char scanner) |
| `src/tokenize/toktok.rs` | ✅ Already LazyLock |
| `src/stem/regexp.rs` | ✅ Already LazyLock |
| `src/tokenize/texttiling.rs` | ✅ Already LazyLock |

**Only fix needed: tweet.rs.**

## Summary

| Item | Effort | Gain | Type |
|---|---|---|---|
| **TweetTokenizer LazyLock** | 10 min | 10-50× | Bugfix |
| String→SmolStr in structs | 2h | <5% | Polish |
| HashMap→FxHashMap | 1h | <5% | Polish |
| Allocation pre-sizing | 1h | <5% | Polish |

**Recommendation:** Fix TweetTokenizer (10 min, 10-50× gain). The rest are polish items that won't meaningfully move benchmarks.
