# Performance Bottleneck Analysis & Optimization Plan

## Benchmark Baseline

All measurements on 10,000-word English text (~65K chars), release build, single core.

| Operation | Time | Throughput | vs str.split |
|---|---|---|---|
| **str.split** (theoretical max) | 0.20 ms | 50M w/s | 1.0x |
| **RegexpTokenizer \S+** | 0.92 ms | 11M w/s | 4.5x |
| **TreebankWordTokenizer** (19 passes) | 0.84 ms | 12M w/s | 4.2x |
| **ToktokTokenizer** | 0.80 ms | 12M w/s | 4.0x |
| **pos_tag** (500 words) | 4.1 ms | 122K w/s | 400x |
| **sent_tokenize** (Punkt) | 3.2 ms | - | - |
| **PorterStemmer** (5000 words) | 0.15 ms | 33M w/s | - |
| **MLE.score** (single call) | 0.08 µs | - | - |

## Bottleneck #1: Regex Engine Overhead (Tokenizers)

**Gap: 0.64ms per 10K words (4x slower than str.split)**

Root cause: The `regex` crate compiles Unicode-aware DFAs for every pattern, handles full UTF-8 decoding, and allocates capture groups. For `\S+` (our most common tokenizer pattern), this is massive overkill — we just want non-whitespace sequences.

**Option A: Manual char scanner for `\S+` (high impact, 2 days)**
Replace `RegexpTokenizer`'s regex with a hand-written `char_indices()` scanner:
```rust
fn tokenize_whitespace(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                tokens.push(text[s..i].to_string());
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        tokens.push(text[s..].to_string());
    }
    tokens
}
```
**Estimated speedup: 3-5x for `\S+` tokenization** (approaches str.split speed)

**Option B: SIMD whitespace scanner (high impact, 1 week)**
Use `memchr` crate to find whitespace byte-by-byte with SIMD (`memchr::memchr` on space byte). Avoids UTF-8 decoding entirely for ASCII-common text.
- `memchr` uses SSE2/AVX2 on x86, NEON on ARM
- Can process 16-32 bytes per cycle
- **Estimated speedup: 6-10x** (close to memchr's 0.03ms for 10K words)

**Option C: Pre-compute regex DFAs at compile time (low effort, 2 hours)**
Replace `LazyLock<Regex>` with `Regex::new(...).unwrap()` in `const` or `static` via the `regex_lite`/`once_cell` compile-time approach. Saves the one-time compilation cost (~100µs per regex) — negligible for production use.

## Bottleneck #2: N-Pass String Copying (Treebank)

**Cost: 19 sequential `re.replace_all` passes, each creating a new `String`**

The Treebank tokenizer does:
```
for each of 19 regex patterns:
    s = re.replace_all(&s, replacement).to_string()   // O(n) scan + allocation
```

At 10K words (65K chars), that's 20 × 65K = 1.3 MB of temporary string allocations per call.

**Fix: Single-pass semantic scanner (high impact, 3 days)**
Instead of sequential regex substitution, write a character-by-character scanner that handles contractions, punctuation splitting, and space collapsing in a single pass:

```rust
fn tokenize_treebank_fast(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::with_capacity(text.len() / 20);
    for (i, c) in text.char_indices() {
        match c {
            // Handle contractions inline
            '\'' if is_contraction_start(&text, i) => { ... }
            // Handle punctuation splitting
            '.' | ',' | '!' | '?' => { flush(&mut current, &mut tokens); push_punct(c, &mut tokens); }
            // Normal whitespace
            c if c.is_whitespace() => flush(&mut current, &mut tokens),
            _ => current.push(c),
        }
    }
    flush(&mut current, &mut tokens);
    tokens
}
```

**Estimated speedup: 5-10x** (one scan instead of 20, avoid all intermediate allocations)

## Bottleneck #3: span_tokenize O(n²) Scan (Treebank)

**Cost: Each call to `span_tokenize` scans the text once per token via `text.find()`**

```rust
for token in &tokens {
    if let Some(pos) = text[search_start..].find(token.as_str()) {
        // O(n) substring search per token => O(n²) total
    }
}
```

**Fix: Collect spans during tokenization (high impact, 1 day)**
Modify the tokenizer to return `Vec<(usize, usize, &str)>` or store byte offsets during the single-pass scan. This eliminates the post-hoc O(n²) search entirely.

## Bottleneck #4: pos_tag Model Loading

**Cost: 4.1ms for 500 words (40x slower than NLTK's own Python tagger)**

The `pos_tag` function:
1. Loads a ~5MB NLTK perceptron model from disk
2. Copies weights into the Rust tagger
3. Runs inference per-token

The slowest part is the Python↔Rust boundary crossing for each word. The `pos_tag_sents` function processes entire sentences at once, amortizing the boundary cost.

**Fix: Batch inference (low effort, explore)**
- Ensure `pos_tag_sents` is preferred over repeated `pos_tag` calls
- Profile the Rust inference loop to check for allocation hot spots
- Consider using a smaller model or pre-compiled binary format

## Prioritized Action Plan

| # | Optimization | Effort | Gain | Module |
|---|---|---|---|---|
| 1 | **Single-pass Treebank scanner** (merge 19 regex passes) | 3 days | 5-10x | `src/tokenize/treebank.rs` |
| 2 | **Manual char scanner** for `\S+` / `\w+` | 2 days | 3-5x | `src/tokenize/regexp.rs` |
| 3 | **Span capture during tokenization** (O(n) not O(n²)) | 1 day | 10-100x | `src/tokenize/regexp.rs` |
| 4 | **SIMD whitespace via memchr** | 1 week | 6-10x | `src/tokenize/regexp.rs` |
| 5 | **pos_tag batch profiling** | 1 day | 2x? | `src/tag/perceptron.rs` |

**Recommended first step: Items 1 + 2 + 3 together (6 days).** These are independent and target the most commonly used code path (tokenization). The theoretical ceiling is str.split (0.2ms/10K words), so we have 4x headroom to capture.

## After-Optimization Target

| Operation | Current | Target | Gain |
|---|---|---|---|
| RegexpTokenizer `\S+` (10K w) | 0.92 ms | 0.10-0.20 ms | 5-9x |
| TreebankWordTokenizer (10K w) | 0.84 ms | 0.15-0.30 ms | 3-6x |
| span_tokenize (10K w) | 1.2 ms | 0.15-0.30 ms | 4-8x |
| pos_tag (1000 w) | 8.2 ms | 4-6 ms | 1.5-2x |
