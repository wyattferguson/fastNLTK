# fastNLTK Performance Improvement Plan

Benchmarks run 2026-07-14, debug build. 42 benchmarks, 33 with NLTK comparison.

---

## Summary

| Category | Count | Items |
|----------|-------|-------|
| **SLOWER** than NLTK | 1 | SpaceTokenizer (0.5x) |
| **MARGINAL** (1.0x-2.0x) | 7 | MWETokenizer, DefaultTagger, RegexpTokenizer, UnigramTagger, BigramTagger, AffixTagger, TrigramTagger |
| **OK** (≥2x) | 25 | — |
| **fastNLTK-only** | 7 | HMM, LM, inference etc. |
| **FAILED** | 2 | TextTiling, KMeans (missing numpy) |

---

## Root Cause Analysis & Fixes

### 1. SpaceTokenizer — 0.5x (2x SLOWER than NLTK)

```
NLTK: 0.25ms   fastNLTK: 0.47ms
```

**Root cause:** `text.split(' ').map(String::from).collect()` allocates a heap `String` for every token. NLTK Python `str.split()` returns zero-allocation slices into the original string.

For 53KB "medium" fixture: ~8000 tokens = 8000 heap allocations.

**Fix:**

```rust
// Old
fn tokenize_space(text: &str) -> Vec<String> {
    text.split(' ').map(String::from).collect()
}

// New — use smol_str::SmolStr (already a dependency)
fn tokenize_space(text: &str) -> Vec<String> {
    let n = text.split(' ').count();
    let mut tokens = Vec::with_capacity(n);
    for s in text.split(' ') {
        tokens.push(
            SmolStr::new(s).to_string()  // stack-alloc for short tokens (<23 bytes)
        );
    }
    tokens
}
```

**Expected improvement:** 1.5-3x faster. SmolStr stack-allocates tokens ≤22 bytes (most English words). Pre-allocating `Vec` capacity avoids reallocations.

**Alternative (bigger win):** Use Python's split directly from Rust via `pyo3::types::PyString::call_method1("split", (" ",))`. Eliminates all Rust-side allocation. However this couples to CPython internals — not recommended.

---

### 2. MWETokenizer — 1.1x (barely faster)

```
NLTK: 1.02ms   fastNLTK: 0.90ms
```

**Root cause:** 

1. `tokenize(&self, text: Vec<String>)` — PyO3 must convert Python `List[str]` → `Vec<String>`, cloning every string on the boundary. NLTK operates on Python list directly with zero conversion.

2. Internal `text[i].clone()` for non-matched tokens adds second allocation.

**Fix:**

```rust
// Change signature to accept &[String] to avoid move/clone on boundary
// Or better: accept &[SmolStr]
fn tokenize(&self, text: &[String]) -> Vec<String> {
    // Use iterators and SmolStr for intermediate lookups
    // Return owned Vec<String> only at the end
}
```

Wait — PyO3 auto-converts `Vec<String>` from Python list regardless. The real fix is to process the Python list in-place:

```rust
fn tokenize(&self, py: Python<'_>, text: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let n = text.len();
    let mut result = Vec::with_capacity(n);
    let mut i = 0;
    // ... trie walk using text.get_item(i)?.extract::<String>()? lazily
}
```

**Expected improvement:** 3-8x. Eliminates O(n) clone on input boundary + O(n) clone inside loop.

---

### 3. DefaultTagger — 1.3x

```
NLTK: 2.84ms   fastNLTK: 2.24ms
```

**Root cause:** `self.tag.clone()` per token inside `map()` closure. Tag is same for all tokens — clone it once outside the iterator.

**Fix:**

```rust
fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
    let tag = &self.tag;
    tokens.into_iter().map(|w| (w, tag.clone())).collect()
    //                           ^ already cloning each time
}
```

Wait, this is already the pattern. The real issue is `clone()` per token. Better: use `Arc<str>` or `Cow<'static, str>` for the tag, or accept that each output token needs its own tag string:

```rust
fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
    let tag = self.tag.clone();  // clone ONCE
    tokens.into_iter().map(|w| (w, tag.clone())).collect()
    //                              ^ still clones per token
}
```

Better fix: use `Rc<str>` for cheap shared clones, or return `Vec<(String, &str)>` by keeping tag in self. But PyO3 requires owned types.

**Real fix:** PyO3 converts output `Vec<(String, String)>` back to Python anyway. Minimize work on Rust side:

```rust
fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
    let n = tokens.len();
    let mut out = Vec::with_capacity(n);
    let tag = self.tag.clone();
    for w in tokens {
        out.push((w, tag.clone()));
    }
    out
}
```

Actually, the `.clone()` of the tag is just cloning a short `String` — this is negligible. The real issue is likely PyO3 input conversion overhead (same as MWETokenizer).

**Expected improvement:** 2-3x if we reduce PyO3 boundary conversion. The tag string clone is trivial (~3 bytes). Disproportionate time is Python → Rust → Python conversion.

---

### 4. RegexpTokenizer — 1.7x

```
NLTK: 1.96ms   fastNLTK: 1.14ms
```

**Root cause:** `re.find_iter(text).map(|m| m.as_str().to_string()).collect()` — `to_string()` allocates per match, same issue as SpaceTokenizer.

**Fix:** Same approach — use `SmolStr`:

```rust
fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
    let re = regex_cache::get_or_compile(&self.pattern, self.flags)?;
    Ok(if self.gaps {
        re.split(text).filter(|s| !s.is_empty()).map(|s| SmolStr::new(s).to_string()).collect()
    } else {
        let matches: Vec<_> = re.find_iter(text).collect();
        let mut out = Vec::with_capacity(matches.len());
        for m in &matches {
            out.push(SmolStr::new(m.as_str()).to_string());
        }
        out
    })
}
```

**Expected improvement:** 2-3x.

---

### 5. NgramTaggers (1.6x-2.2x) — UnigramTagger, BigramTagger, TrigramTagger, AffixTagger

```
UnigramTagger:  3.78 → 2.42ms (1.6x)
BigramTagger:   7.98 → 4.43ms (1.8x) 
TrigramTagger:  7.74 → 3.54ms (2.2x)
AffixTagger:    6.56 → 3.47ms (1.9x)
```

**Root causes:**

a) **PyO3 boundary cost**: All taggers accept `Vec<String>` input and return `Vec<(String, String)>` output. Conversion dominates for small workloads. Benchmark at only ~1000 tokens — boundary cost is disproportionate.

b) **BigramTagger/TrigramTagger use `std::collections::HashMap`** instead of `hashbrown::HashMap` (`FastMap`). Slower.

c) **BigramTagger/TrigramTagger use `String`** not `SmolStr` for keys/values. Heap allocation per lookup key.

d) **Excess cloning**: `prev.clone()`, `w.clone()`, key cloning each iteration.

**Fixes:**

For BigramTagger and TrigramTagger — align with UnigramTagger pattern:

```rust
// Convert to FastMap + SmolStr (like UnigramTagger)
pub struct BigramTagger {
    bigram_map: FastMap<(SmolStr, SmolStr), SmolStr>,
    default_tag: Option<SmolStr>,
}

fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
    let n = tokens.len();
    let mut out = Vec::with_capacity(n);
    let mut prev = SmolStr::new_inline("START");
    let default = self.default_tag.clone().unwrap_or_default();
    for w in tokens {
        let w_smol = SmolStr::new(&w);
        let key = (prev.clone(), w_smol.clone());
        let tag = self.bigram_map.get(&key).cloned().unwrap_or_else(|| default.clone());
        out.push((w, tag.to_string()));
        prev = tag;
    }
    out
}
```

**Expected improvement:** 3-5x. `FastMap` is ~2x faster than `std::HashMap`. `SmolStr` eliminates key allocations. Pre-alloc `Vec`.

---

### 6. ToktokTokenizer — 3.9x (below 5x floor)

```
NLTK: 8.35ms   fastNLTK: 2.15ms
```

**Root cause:** `build_subs()` creates regex substitutions per call? No — it's called in `tokenize()` which runs the regex pipeline on each tokenize call. The regex + String pipeline is:

```rust
fn tokenize(&self, text: &str, return_str: bool) -> Vec<String> {
    let mut s = text.to_string();          // clone full text
    let subs = build_subs();               // build regex vec EVERY call
    for (re, replacement) in &subs {
        s = re.replace_all(&s, replacement.as_str()).to_string();  // clone full text per substitution
    }
    // ... split and collect
}
```

**Issues:**
- `build_subs()` called on every `tokenize()` — builds ~20 Regex objects each time
- `re.replace_all().to_string()` clones entire text after EACH of ~20 substitutions
- Final `split_whitespace().map(String::from)` allocates per token

**Fix:**

```rust
use once_cell::sync::Lazy;

static TOKTOK_SUBS: Lazy<Vec<(Regex, String)>> = Lazy::new(build_subs);
```

Then for the replace chain — use a single scan with a combined regex to avoid intermediate clones:

```rust
fn tokenize(&self, text: &str, return_str: bool) -> Vec<String> {
    let subs = &*TOKTOK_SUBS;
    let mut s: Cow<str> = Cow::Borrowed(text);
    for (re, replacement) in subs {
        if re.is_match(s.as_ref()) {
            let new_s = re.replace_all(s.as_ref(), replacement.as_str());
            s = Cow::Owned(new_s.into_owned());
        }
    }
    let t = s.trim();
    if return_str { return vec![t.to_string()]; }
    if t.is_empty() { return Vec::new(); }
    t.split_whitespace().map(|s| SmolStr::new(s).to_string()).collect()
}
```

**Expected improvement:** 2-3x (targeting 8-12x total speedup).

---

### 7. DefaultReasoner — 55ms (fastNLTK-only, suspiciously slow)

No NLTK comparison benchmark exists. 55ms for a single inference operation is unusually slow for Rust. Likely a bug or unoptimized path.

**Investigation needed:**
- Profile the extensions() call
- Check for accidentally quadratic loops or excess allocation
- Likely candidate: recursion depth or itertools combinator explosion

---

## Priority Implementation Order

| Priority | Item | Current Speedup | Target | Effort |
|----------|------|----------------|--------|--------|
| **P0** | SpaceTokenizer — SmolStr | 0.5x | 2x+ | 15 min |
| **P0** | MWETokenizer — PyList boundary | 1.1x | 5x+ | 1 hr |
| **P1** | ToktokTokenizer — Lazy static regexes | 3.9x | 8x+ | 1 hr |
| **P1** | BigramTagger/TrigramTagger → FastMap+SmolStr | 1.8x/2.2x | 5x+ | 2 hr |
| **P2** | RegexpTokenizer — SmolStr | 1.7x | 3x+ | 15 min |
| **P2** | DefaultTagger — boundary optimization | 1.3x | 3x+ | 30 min |
| **P2** | AffixTagger — SmolStr + pre-alloc | 1.9x | 4x+ | 30 min |
| **P3** | DefaultReasoner — profile + fix | N/A | <1ms | 2 hr |
| **FIXME** | TextTiling — install numpy | FAILED | 50x+ | 5 min |
| **FIXME** | KMeansClusterer — install numpy | FAILED | 5x+ | 5 min |
| **FIXME** | HMM Tagger — add NLTK comparison | false regr | 3x+ | 1 hr |
| **FIXME** | KneserNey — fix benchmark design | false regr | stable | 15 min |
| **FIXME** | Harness — noise floor for microbenchmarks | false regr | stable | 30 min |

---

---

### 8. TextTilingTokenizer — FAILED: `name 'numpy' is not defined`

**Root cause:** NLTK's `TextTilingTokenizer` internally imports and uses `numpy` for matrix operations (cosine similarity, depth scoring). numpy is not installed in the benchmark environment.

NLTK's `nltk.tokenize.texttiling` source does:
```python
import numpy as np
# ... uses np.zeros, np.array, np.sum, np.dot, etc.
```

The fastNLTK Rust implementation (`src/tokenize/texttiling.rs`) does NOT use numpy — it implements the algorithm in pure Rust. But the benchmark first creates an NLTK instance, which triggers the numpy import.

**Fix:** Install numpy in the environment:
```
pip install numpy
```

Then the benchmark can run both NLTK and fastNLTK sides. The Rust impl should be 50x+ faster than the numpy/Python NLTK version.

---

### 9. KMeansClusterer — FAILED: `No module named 'numpy'`

**Root cause:** Same as above — the benchmark explicitly imports `numpy` to create `np.array` vectors for NLTK's `KMeansClusterer`. numpy not installed.

**Fix:** Install numpy (`pip install numpy`). The Rust KMeansClusterer (`src/cluster.rs`) implements Lloyd's algorithm in pure Rust with fast convergence detection — expected 5-10x over NLTK's pure-Python + numpy version.

---

### 10. HiddenMarkovModelTagger — Regression false positive (0.25ms, fastNLTK-only)

**Symptom:** Triggers `--regression` failures when comparing baselines. The benchmark runs successfully (0.248ms in latest run), but the regression checker flags it because measurement noise on sub-millisecond measurements exceeds the 25% relative threshold.

**Root cause in benchmark harness:**
```python
# harness.py lines 133-136
cur_time = cur.fast_only_ms or cur.fast_ms
base_time = base.fast_only_ms or base.fast_ms
change = (cur_time - base_time) / base_time
if change > threshold:  # default 0.25 = 25%
    regressions.append(...)
```

For HMM at 0.25ms: 25% = 0.062ms noise floor. OS scheduling jitter, cache effects, and CPU frequency scaling can shift a 0.25ms measurement by ±0.05ms (20%). Any baseline shift > 0.06ms triggers false regression.

**Fix A — Harness (quick, handles all fastNLTK-only benchmarks):**
```python
# Add absolute noise floor to regression check
MIN_ABSOLUTE_REGRESSION_MS = 0.5  # Don't flag regressions below 0.5ms absolute change

cur_time = cur.fast_only_ms or cur.fast_ms
base_time = base.fast_only_ms or base.fast_ms
if cur_time < MIN_ABSOLUTE_REGRESSION_MS and base_time < MIN_ABSOLUTE_REGRESSION_MS:
    continue  # Skip microbenchmarks — noise dominates
```

**Fix B — Add NLTK comparison (better, makes benchmark meaningful):**
- NLTK has `nltk.tag.hmm.HiddenMarkovModelTagger` — add side-by-side comparison
- Removes HMM from `fast_only` category, gives it real speedup metric
- Expected speedup: 3-5x (rustling HMM vs Python HMM)

**Rust optimization opportunity:** In `hmm.rs` line 67: `model.predict(vec![tokens.clone()])` clones entire token vector. Can pass `&[Vec<String>]` instead. Also `model.state_labels().clone()` is unnecessary — labels are static after training.

---

### 11. KneserNeyInterpolated — Regression false positive (5.3µs, fastNLTK-only)

**Symptom:** Same as HMM — microsecond-scale measurement triggers false regression. 5.3µs baseline with 25% threshold = 1.3µs noise floor. System timer resolution alone is ~1µs (Windows `QueryPerformanceCounter`). Impossible to stay within threshold.

**Root cause in benchmark design:** The benchmark creates AND fits the model inside the timed lambda:
```python
def run():
    m = KneserNeyInterpolated(2, 0.75)
    m.fit([["the", "cat"], ["the", "dog"], ["a", "cat"], ["the", "mouse"]])
    return [m.score(w, ["the"]) for w in ["cat", "dog", "mouse", "rat"]]
```
Measuring create+fit+score as one unit. The `fit()` method clones every word (`word.clone()`, `token.clone()`) — O(n) overhead inside the timing loop.

**Fix A — Redesign benchmark (measure only scoring):**
```python
def bench_kneser_ney() -> BenchResult:
    # Pre-fit model outside timing loop
    m = KneserNeyInterpolated(2, 0.75)
    m.fit([["the", "cat"], ["the", "dog"], ["a", "cat"], ["the", "mouse"]])

    words_1k = ["cat", "dog", "mouse", "rat"] * 250
    f_ms = _median_time(lambda: [m.score(w, ["the"]) for w in words_1k], 100)
    return BenchResult(
        name="KneserNeyInterpolated.score", group="lm",
        params={"queries": len(words_1k)},
        fast_only_ms=f_ms, iterations=100,
    )
```
Then `fast_only_ms` will be ~0.5ms for 1000 queries instead of 5µs for 4 — noise becomes negligible.

**Fix B — Harness (same as HMM):** Add absolute noise floor of 0.5ms to regression check.

**Additional issues in KneserNey Rust:** The `score()` computation uses `self.counts.len()` twice and `self.total.max(1.0)` even though `self.total` is guaranteed positive after `fit()`. Precompute `vocab_size` and `inv_vocab_size` during `fit()` to avoid recomputation in hot path.

---

## Cross-Cutting Optimization: PyO3 Boundary Cost

The fastest functions (172x edit_distance, 117x windowdiff) work because they do heavy Rust computation on a small Python input. The slowest functions (DefaultTagger, MWETokenizer, SpaceTokenizer) spend disproportionate time converting Python ↔ Rust types.

**General pattern to eliminate boundary cost:**

```rust
// BEFORE — conversion dominates for small workloads
fn tokenize(&self, text: Vec<String>) -> Vec<String> { ... }

// AFTER — accept Python-native types, convert lazily
fn tokenize(&self, py: Python<'_>, text: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let n = text.len();
    let mut result = Vec::with_capacity(n);
    for item in text.iter() {
        let word: String = item.extract()?;  // convert only when needed
        // ... process
    }
    Ok(result)
}
```

Apply to: MWETokenizer, DefaultTagger, BigramTagger, TrigramTagger, AffixTagger, UnigramTagger.

---

## Verification

After each fix:
1. `cargo test` — ensure correctness
2. `python -m benchmarks.run --save` — measure improvement
3. Check speedup column ≥ target
