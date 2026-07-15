# Performance Optimization Plan — Low Performers

Based on benchmark audit + online research.

## Ranking: Worst → Best

| Rank | Module | vs NLTK | Absolute time | Root cause | Effort | Est. gain |
|---|---|---|---|---|---|---|
| 1 | **SpaceTokenizer** | **0.4×** | 0.26→0.68ms | PyO3 boundary cost dominates tiny op | 2h | 3-5× |
| 2 | **TnT tagger** | **0.8×** | 1.46→1.72ms | SmolStr alloc in Viterbi inner loop | 1d | 3-5× |
| 3 | **MWETokenizer** | 1.0× | 1.00→0.96ms | Python-level string matching? | 4h | 2-3× |
| 4 | **DefaultTagger** | 1.1× | 1.67→1.55ms | Tag clone per token | 1h | 2× |
| 5 | **RegexpTokenizer** (custom) | 1.4× | 2.15→1.57ms | NLTK uses C `re` module | 4h | 2× |
| 6 | **UnigramTagger** | 1.5× | 2.32→1.55ms | SmolStr clone for hash key | 2h | 2× |
| 7 | **BigramTagger** | 2.0× | 3.94→1.99ms | SmolStr tuple key clone | 4h | 2-3× |
| 8 | **TrigramTagger** | 2.0× | 4.15→2.04ms | SmolStr triple key clone | 4h | 2-3× |
| 9 | **WordPunctTokenizer** | 2.0× | 5.30→2.30ms | Regex + SmolStr overhead | 2h | 2× |
| 10 | **AffixTagger** | 2.0× | 3.47→1.92ms | SmolStr clone + regex | 2h | 2× |
| 11 | **CCG parser** | 2.0× | 1.26→0.77ms | Complex recursive parsing | 2d | 2× |
| 12 | **NaiveBayes.classify** | 3.0× | 9.66→2.99ms | HashMap with SmolStr keys | 4h | 2× |

---

## Detailed Analysis

### #1 — SpaceTokenizer (0.4×)

**Problem:** `text.split(' ').map(String::from).collect()` is dominated by PyO3 call overhead for tiny inputs. NLTK's Python `str.split(' ')` runs at C speed and avoids the boundary.

**Research:** CPython's `str.split` is hand-tuned C that reuses existing PyUnicode objects (zero copy). Our Rust function pays 0.7µs just to cross the PyO3 boundary, which is meaningful when the function itself takes 0.26ms.

**Fix:** Replace with the same SIMD `memchr3` scanner used by `RegexpTokenizer`. Since SpaceTokenizer splits on any whitespace (not just space), we use our fast `\S+` scanner path. This also avoids the split-iterator overhead.

````rust
fn tokenize_space(text: &str) -> Vec<String> {
    // Reuse the SIMD memchr3 scanner from regexp.rs
    tokenize_whitespace(text)
}
````

**Est. gain: 3-5×** (same as RegexpTokenizer improvement: 0.927ms→0.272ms)

---

### #2 — TnT Tagger (0.8×)

**Problem:** Three nested loops in Viterbi decoding with `SmolStr::new()` and tuple-key HashMap clones in each iteration.

**Code:** `src/tag/tnt.rs`
- `SmolStr::new(tag)` called for every `(word, tag)` pair in O(N×T²) loop
- `self.trans_prob_smol(&SmolStr::new(tag_k), &SmolStr::new(tag_j))` — creates two SmolStrs per inner iteration
- `self.bi_counts.get(&(t1.clone(), t2.clone()))` — clones both SmolStrs to create tuple key
- `trans_prob_smol` iterates ALL entries with `filter(|((a, _), _)| a == t1)` to recompute `total` — O(T) per lookup

**Research:** Standard Viterbi optimization techniques:
1. **Transposed transition matrix** — store in column-major for cache-friendly access (from `crfs-rs` commit d303f90)
2. **Integer tag IDs** — map `SmolStr` tags to `u16` indices once, then use `[[f64; T]; T]` arrays instead of HashMaps
3. **Pre-compute `total`** for each transition row — avoid O(T) scan per lookup
4. **Statically-sized arrays** for small tag sets (typical T=45) — eliminate all HashMap overhead

**Fix:**
````rust
// Step 1: Pre-compute tag→ID mapping (done once in train())
let tag_to_id: FxHashMap<SmolStr, u16> = ...;

// Step 2: Store transition counts as 2D arrays
// bi_counts[t1][t2] = count
let bi_counts: Vec<Vec<u64>> = vec![vec![0; T]; T];
let tri_counts: Vec<Vec<Vec<u64>>> = vec![vec![vec![0; T]; T]; T];

// Step 3: Pre-compute per-tag totals
let bi_totals: Vec<u64> = (0..T).map(|i| bi_counts[i].iter().sum()).collect();
let uni_totals: Vec<u64> = uni_counts.iter().sum();

// Step 4: Emission probs as 2D array: em_probs[tag_id][word_id] = prob
let mut em_probs: Vec<Vec<f64>> = vec![vec![0.0; V]; T];
````

This eliminates ALL HashMap lookups and SmolStr allocations from the Viterbi loop. With T≈45 tags, the inner loop goes from ~180 HashMap ops per word to ~180 array accesses.

**Est. gain: 3-5×** (matching the typical Rust-vs-Python gap for tight loops)

---

### #3 — MWETokenizer (1.0×)

**Problem:** Probably string matching with many intermediate allocations.

**Fix:** Analyze and apply integer IDs for phrase matching.

**Est. gain: 2-3×**

---

### #4-#10 — Sequential Taggers (1.1×-2.0×)

**Common problem:** All of these taggers use `SmolStr` tuple keys in HashMaps. Each `get()` requires cloning both SmolStrs to build the key. For bigram/trigram taggers, the tuple key construction allocates.

**Fix:** Same approach as TnT — integer tag IDs:

````rust
// Before:
FastMap<(SmolStr, SmolStr), SmolStr>  // BigramTagger

// After:
Vec<Vec<u16>>  // bigram_map[prev_tag_id][word_id] = tag_id
````

For taggers with small vocabularies, a `Vec<Vec<u16>>` is faster than any HashMap for lookups (dense access, no hashing).

**Est. gain: 2-3×** each.

---

### #9 — WordPunctTokenizer (2.0×)

**Problem:** Uses regex `\w+|[^\w\s]+` via the Rust `regex` crate. NLTK uses Python's `re` which is also C. The gap is small.

**Fix:** Add a fast-path char scanner for the `\w+|[^\w\s]+` pattern, similar to our `\S+` fast path. Scan for runs of word chars and non-word non-whitespace chars.

**Est. gain: 3-4×**

---

## Implementation Priority

| Priority | Item | Effort | Gain | Depends on |
|---|---|---|---|---|
| **P0** | SpaceTokenizer → SIMD scanner | 2h | 3-5× | — |
| **P1** | TnT → integer IDs + flat arrays | 1d | 3-5× | — |
| **P2** | Bigram/Trigram taggers → integer IDs | 4h | 2-3× | P1 (same technique) |
| **P3** | Unigram/Affix/Default taggers → integer IDs | 4h | 2× | P1 |
| **P4** | WordPunctTokenizer → char scanner | 2h | 2× | — |
| **P5** | MWETokenizer → analyze + fix | 4h | 2× | — |
| **P6** | CCG parser → profile + optimize | 2d | 2× | — |

**Recommended first step: P0 + P1** — SpaceTokenizer is the worst offender (SLOWER than NLTK) and takes 2 hours. TnT is the second-worst and the integer-ID optimization unlocks P2/P3. Together they're ~1.5 days and bring the bottom of the benchmark table from 0.4×-0.8× to 3-5×.
