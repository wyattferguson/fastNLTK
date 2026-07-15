# Architectural Performance Plan

## Current Baselines (v0.4.0)

| Operation | Cold | Warm | Throughput |
|---|---|---|---|
| Tree.from_string (3 nodes) | 0.42µs | 0.42µs | 2.4M trees/s |
| CFG.from_string (tiny) | 0.62µs | 0.62µs | 1.6M grammars/s |
| pos_tag (100 words) | **80.6ms** | 0.01ms | 9M words/s |
| TreebankWordTokenizer (10K w) | 0.46ms | 0.46ms | 22M words/s |
| RegexpTokenizer (4 words) | 0.23µs | 0.23µs | 17M calls/s |

---

## 1. SIMD-Parsed Tree Format

**Research:** Tree.from_string and CFG.from_string parse bracket/arrow notation via recursive descent. At 0.42-0.62µs per call, they're already near the practical ceiling for general-purpose parsing. SIMD (AVX2/NEON) is designed for data-parallel workloads (same operation on many elements). Recursive descent parsing is control-flow-heavy: branching on every character, nested function calls, allocation.

The only thing SIMD could help with is:
- `memchr(b'(')` or `memchr(b'-')` to find the next structural token — but the input strings are already tiny (<100 bytes) so SIMD setup overhead outweighs benefit.
- Huffman/SIMD-decoded token stream — requires preprocessing the input format, not applicable to arbitrary user input.

**Verdict: No gain. SIMD cannot accelerate recursive descent parsing.**

| Claim | Reality |
|---|---|
| SIMD for token scanning | Already using memchr3 in tokenizers (done) |
| SIMD for tree parsing | Control-flow bound, not memory bound |
| Potential gain | **0%** — not applicable to this problem |

---

## 2. Pre-Compiled Model Binary Blobs

**Problem:** Cold pos_tag takes **80.6ms**, dominated by:
1. NLTK data search + pickle.open: ~5ms
2. Python pickle.load of 6MB: ~50ms (GIL-locked, single-threaded)
3. PyDict iteration + Rust FxHashMap insertion: ~25ms

**Fix:** Cache the Rust model state as bincode after first load.

Already have the building blocks:
- `bincode` crate in Cargo.toml ✅
- `bincode_cache_path()` in `src/data.rs` ✅
- `Serde` derives on model structs — need to add to PerceptronTagger

**Implementation plan (2 days):**

```rust
// After first successful pickle load, save bincode cache
fn save_model_cache(&self) {
    let path = bincode_cache_path("perceptron_tagger");
    let data = bincode::serialize(&self.weights).unwrap();
    std::fs::write(&path, data).ok();
}

// Before pickle load, check cache
fn try_load_cache() -> Option<Self> {
    let path = bincode_cache_path("perceptron_tagger");
    let data = std::fs::read(&path).ok()?;
    bincode::deserialize(&data).ok()
}
```

The trick: `PerceptronTagger.weights` is `FxHashMap<u64, FxHashMap<SmolStr, f64>>`. Serde derives already work if we add `#[derive(Serialize, Deserialize)]`.

**Estimated gain:**

| Load path | Time | Savings |
|---|---|---|
| Current (pickle + PyDict iteration) | 80.6ms | — |
| First load (pickle + save bincode) | 81ms | 0 (save cost added) |
| Second load+ (bincode deserialize) | **~5-10ms** | **-90%** |
| Tagdict/classes baked into blob | — | Removes separate Python→Rust weight transfer |

**Caveat:** Only helps cold start. Warm inference is already 0.01ms.

---

## 3. Thread-Level Parallelism via Rayon

**Current state:** Zero rayon usage in `src/` (despite `features = ["parallel"]` existing in Cargo.toml). The `parallel` feature only enables `rustling/parallel`.

**Opportunity:** Several fastNLTK operations are embarassingly parallel — each item is independent.

### Candidate operations

| Operation | Current (1 core) | Scalable? | Est. gain (8 cores) |
|---|---|---|---|
| `pos_tag_sents` (batch of 100 sents) | Sequential per-sentence | ✅ Embarrassingly parallel | **4-6×** |
| `PorterStemmer.stem` (list of words) | Sequential per-word | ✅ Yes | **4-6×** |
| `FreqDist.update` (batch of docs) | Sequential per-doc | ✅ Yes | **3-5×** |
| `word_tokenize` on long text | Sequential chars | ❌ No (string state) | 0 |
| `Bleu.score` (batch) | Sequential per-hypothesis | ✅ Yes | **4-6×** |
| `edit_distance` (batch of pairs) | Sequential per-pair | ✅ Yes | **4-6×** |

### Implementation

**Pattern:** Add `#[cfg(feature = "parallel")]`-gated rayon code behind existing methods.

```rust
fn tag_sents(&self, sentences: Vec<Vec<String>>) -> Vec<Vec<(String, String)>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        return sentences.par_iter().map(|s| self.tag_sentence(s)).collect();
    }
    #[cfg(not(feature = "parallel"))]
    {
        sentences.iter().map(|s| self.tag_sentence(s)).collect()
    }
}
```

**Pitfall:** Overhead of `par_iter` for small batches. For <10 sentences the sequential version is faster. The decision should be:
- If `sentences.len() > 50`: use rayon (parallel overhead amortized)
- If `sentences.len() <= 50`: use sequential (no overhead)

**Estimated gain (batch operations, 8 cores):** 4-6×

| Module | Method | Current (1K items) | With rayon (8 cores) | Speedup |
|---|---|---|---|---|
| tag | pos_tag_sents (200 sents) | 1.1ms | **0.2ms** | 5× |
| stem | PorterStemmer (10K words) | 8.7ms | **1.5ms** | 5.5× |
| collocations | BigramCF.from_words (10 batches) | 2.0ms | **0.5ms** | 4× |
| probability | FreqDist.update (100 docs) | 3.0ms | **0.6ms** | 5× |

---

## Summary

| Item | Effort | Gain | Applies to | Do it? |
|---|---|---|---|---|
| SIMD tree format | 1 week | **0%** | Tree/CFG parsing | ❌ No benefit |
| Bincode model cache | 2 days | **-90% cold start** | pos_tag model load | ✅ **Yes** |
| Rayon parallel batch ops | 3 days | **4-6×** | tag/stem/collocations/prob | ✅ **Yes** |

**Recommendation: Skip SIMD tree format (no benefit). Implement bincode model cache (2 days, 80ms→8ms cold start) + rayon parallel batch ops (3 days, 4-6× on large inputs).**
