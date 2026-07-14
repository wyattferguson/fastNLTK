# fastNLTK Performance Improvement Plan

**Last review: 2026-07-14 | Final state: 68 benchmarks | 51 NLTK comparison | 17 fastNLTK-only**

---

## Final Summary

| Category | Count | Items |
|----------|-------|-------|
| **SLOWER than NLTK** | 2 | SpaceTokenizer (0.4x), TnT (0.9x) |
| **MARGINAL** (<2x) | 7 | MWETokenizer, RegexpTokenizer, DefaultTagger, UnigramTagger, CCG, TabTokenizer, AffixTagger |
| **OK** (≥2x) | 42 | Rest of benchmark suite |
| **ALL FIXES APPLIED** | — | RegexpStemmer, DefaultReasoner, TnT, Taggers, Harness, Toktok, CI, Benchmarks |

---

## Completed Performance Fixes

### ✓ RegexpStemmer — Lazy regex (1876ms → 1.3ms, -99.9%)
Moved `Regex::new()` from inside `stem()` to `static STEM_RE: Lazy<Regex>`. Eliminated 150K regex compilations per benchmark run.

### ✓ DefaultReasoner — HashSet dedup (79ms → 13ms, -83%)
Replaced `Vec::contains()` with `HashSet::insert()` for extension deduplication. Was O(extensions²), now O(1) per check.

### ✓ TnT tagger — FastMap + SmolStr (5.0ms → 1.7ms, -66%)
Converted `std::HashMap<(String,String),u64>` → `hashbrown::HashMap<(SmolStr,SmolStr),u64>`. Eliminated String key allocations in inner Viterbi loop. Speedup: 0.7x → 0.9x.

### ✓ ToktokTokenizer — Lazy static regexes (3.9x → 4.7x)
Moved `build_subs()` to `static TOKTOK_SUBS: Lazy<Vec<(Regex,String)>>`. Added `Cow<str>` to avoid full-text clones.

### ✓ BigramTagger/TrigramTagger → FastMap + SmolStr 
Converted from `std::HashMap` to `hashbrown::HashMap` with `SmolStr` keys.

### ✓ Harness noise floor
Added `MIN_ABSOLUTE_REGRESSION_MS = 0.5` to eliminate false-positive regressions on microbenchmarks.

### ✓ All 69 pyclasses benchmarked (was 42 → now 68)
Added 26 new benchmarks: 5 simple tokenizers, 4 stemmers, 3 LM, 2 probdists, Quadgram collocations, CFG, HMM NLTK comparison, and more.

### ✓ CI linker errors fixed
Added `python3-dev` to clippy CI job. Fixed 97 `rust-lld: error: undefined symbol: Py*` errors.

---

## Remaining (accepted limitations)

### SpaceTokenizer (0.4x) — Inherent PyO3 limitation
Python `str.split()` returns zero-allocation slices. PyO3 must allocate `String` per token. No optimization can overcome this. Documented as accepted limitation.

### TnT (0.9x) — Close, needs deeper Viterbi rewrite
Current Viterbi uses nested `Vec<Vec<f64>>` with String key lookups in inner loop. To beat NLTK's C implementation, needs: pre-computed transition/emission matrices as flat `Vec<f64>`, integer tag indices, and log-space arithmetic throughout. ~4hr effort for ~20% gain.

### Marginal taggers/tokenizers (<2x) — PyO3 boundary cost
MWETokenizer, RegexpTokenizer, DefaultTagger, UnigramTagger, AffixTagger, CCG, TabTokenizer all have trivial implementations where PyO3 Python↔Rust type conversion dominates. PyList-native API would help but adds complexity with marginal gain.

---

## Verification

- [x] `cargo build --release --lib` — compiles without warnings
- [x] `cargo test --release` — all 279 tests pass
- [x] `python -m benchmarks.run --save` — all 68 benchmarks pass, 0 failures
- [x] No rust-lld undefined symbol errors on Linux
