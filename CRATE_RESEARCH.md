# Existing Rust Crate Analysis for fastNLTK

Deep evaluation of every Rust crate that can reduce fastNLTK workload.
Includes license, maturity, API surface, integration cost, and whether we reuse or just reference.

---

## Tier 1: Direct Reuse (dep via Cargo.toml, zero porting)

These crates are mature, correct, and map directly onto NLTK features.
Add them as dependencies and wrap with PyO3.

### 1.1 `rust-stemmers` — Snowball Stemmer

| Field | Value |
|---|---|
| **Crate** | `rust-stemmers` v1.0 |
| **License** | MIT |
| **Downloads** | ~500K+ |
| **Maturity** | Stable, used by dozens of crates. Port of Snowball libstemmer in pure Rust. |
| **Languages** | Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish, Arabic |
| **API** | `Stemmer::create(algo) → stem(&self, &str) → Cow<str>` |
| **Mapping to NLTK** | Direct. NLTK's `SnowballStemmer` uses the same Snowball algorithms. |
| **Integration** | Add to `Cargo.toml`. Write a 50-line PyO3 wrapper. |
| **LoC saved vs rewrite** | ~5,900 LoC (NLTK's `snowball.py` + exception tables) |
| **Speedup vs Python** | 15-20x (proven by vtext) |
| **Verdict** | ✅ **USE. No-brainer.** Covers all 16 Snowball languages NLTK supports. Porting the exception lists (English `running→run`, etc.) is the only work — those are data tables, not algorithm. |

### 1.2 `regex` — Regex Engine

| Field | Value |
|---|---|
| **Crate** | `regex` v1.11 |
| **License** | MIT/Apache-2.0 |
| **Maturity** | The most-used Rust crate. Guaranteed linear time. No ReDoS. |
| **API** | `Regex::new(pattern) → Regex`, `find_iter()`, `split()`, `replace()` |
| **Mapping to NLTK** | NLTK's `RegexpTokenizer`, `WordPunctTokenizer`, `BlanklineTokenizer`, `TreebankWordTokenizer`, `ToktokTokenizer`, `TweetTokenizer`, `RegexpStemmer`, `RegexpChunkParser` all use Python's `re` module. |
| **Integration** | Already a dependency. Write Rust tokenizer classes around it. |
| **LoC saved vs rewrite** | Impossible to quantify — regex engine itself is ~50K LoC C. Using the crate saves writing our own. |
| **Verdict** | ✅ **REQUIRED.** Already in Cargo.toml. This is the foundation of every tokenizer. |

### 1.3 `unicode-segmentation` — Unicode Word/Sentence Boundaries

| Field | Value |
|---|---|
| **Crate** | `unicode-segmentation` v1.12 |
| **License** | MIT/Apache-2.0 |
| **Maturity** | Very mature. Unicode TR#29 compliant. |
| **API** | `UnicodeSegmentation::graphemes()`, `split_word_bounds()`, `sentences()` |
| **Mapping to NLTK** | Used by NLTK indirectly. Directly used by vtext's tokenizers. NLTK's Punkt tokenizer implements its own heuristics instead of Unicode segmentation, but the Simple tokenizers could benefit. |
| **Integration** | Add to `Cargo.toml`. Use `split_word_bounds()` for unicode tokenizer, `sentences()` for rough sentence splitting. |
| **LoC saved** | ~500 LoC (implementing Unicode segmentation rules is complex) |
| **Verdict** | ✅ **USE.** Covers UnicodeWordTokenizer, UnicodeSentenceTokenizer, CharacterTokenizer. |

### 1.4 `hashbrown` — Faster HashMaps

| Field | Value |
|---|---|
| **Crate** | `hashbrown` v0.15 |
| **License** | MIT/Apache-2.0 |
| **Maturity** | Production. Used internally by Rust's std HashMap since 1.36. |
| **API** | Drop-in `HashMap<K, V>` replacement |
| **Mapping** | All FreqDist, tagger weight dicts, ngram counters, vocabulary lookups. |
| **Integration** | `use hashbrown::HashMap` instead of `std::collections::HashMap` |
| **Verdict** | ✅ **USE.** Small change, measurable perf win in hot loops. |

### 1.5 `rustc-hash` — FxHashMap for Small Keys

| Field | Value |
|---|---|
| **Crate** | `rustc-hash` v2.1 |
| **License** | Apache-2.0/MIT |
| **Maturity** | Used by rustc compiler itself. |
| **API** | `FxHashMap<K, V>`, `FxHashSet<K>` — FNV-based hasher |
| **Verdict** | ✅ **USE.** Proven by rustling. Use for tagger feature weights, class sets, string→ID maps. |

---

## Tier 2: Partial Reuse (install as dep, wrap subset of API)

These crates implement functionality we need, but their API doesn't match NLTK exactly.
We wrap them with adapter code.

### 2.1 `vtext` crate (Rust core, not Python package)

| Field | Value |
|---|---|
| **Crate** | `vtext` v0.2.0 |
| **License** | Apache-2.0 |
| **Maturity** | Low version (0.2), last updated 2021. But the tokenization code is solid. |
| **What it provides** | `RegexpTokenizer`, `UnicodeWordTokenizer`, `VTextTokenizer` (en/fr), `UnicodeSentenceTokenizer`, `PunctuationTokenizer`, `CharacterTokenizer`, `edit_distance`, `dice_similarity`, `jaro_similarity`, `jaro_winkler_similarity`, CountVectorizer, HashingVectorizer, Snowball stemmer |
| **Can we depend on it?** | **Bad idea.** v0.2.0, last commit 2021. PyO3 v0.10 (very old, incompatible with modern 0.23). Heavy deps (ndarray, sprs, itertools). |
| **Better approach** | **Copy/port the relevant source files.** The tokenization code is 400 LoC of pure Rust with no vtext-specific types. The metrics code is 300 LoC. Both are Apache-2.0 licensed. We can port the algorithms without depending on the crate. |
| **What to port** | `src/tokenize/mod.rs` (VTextTokenizer English contraction rules), `src/tokenize_sentence/mod.rs` (PunctuationTokenizer), `src/metrics/string.rs` (edit_distance, jaro, jaro_winkler, dice) |
| **LoC saved vs from-scratch** | ~700 LoC (tokenizer rules + 4 string metrics) |
| **Verdict** | 🔄 **PORT ALGORITHMS.** Don't depend on the crate — copy the Apache-2.0 code and adapt to our types. The contraction rules for English are correct and tested against UD treebanks. |

### 2.2 `rustling` crate (Rust core)

| Field | Value |
|---|---|
| **Crate** | `rustling` v0.8.0 |
| **License** | MIT |
| **Maturity** | Active (Feb 2026). Well-designed architecture. Uses FlatBuffers for model serialization. |
| **What it provides** | `AveragedPerceptron` (POS tagger with training+prediction), `MLE`/`Lidstone`/`Laplace` language models, `HiddenMarkovModel`, word segmentation (LongestStringMatching, HMM), Ngram counting via `CountTrie` |
| **Feature detection** | `SeqFeatureConfig`, `SeqFeatureTemplate`, `default_tagger_ap_features()` — extensible feature templates matching NLTK's perceptron tagger |
| **Model serialization** | FlatBuffers + gzip-compressed JSON. Both save/load. |
| **Dependency weight** | Light. Deps: `rand`, `rayon` (optional), `regex`, `fancy-regex`, `rustc-hash`, `flatbuffers`, `serde`, `serde_json`, `uuid`, `walkdir`, `zip`, `quick-xml`, `zstd`. Most are for CHAT/ELAN/TextGrid format conversion — **we only need the NLP subset**. |
| **Can we depend on it?** | **Better option: depend on it for the core NLP modules.** The crate is well-structured into modules (`perceptron_pos_tagger`, `lm`, `ngram`, `hmm`, `seq_feature`). The perceptron tagger is a direct implementation of the same algorithm NLTK uses (averaged perceptron from textblob-aptagger / Matthew Honnibal). We can depend on `rustling` and expose only the modules we need. |
| **Integration cost** | Medium. Need to write adapter that loads NLTK's pickle weights into rustling's model format. Or train from scratch using NLTK data. |
| **LoC saved vs rewrite** | ~4,000 LoC (perceptron tagger + LM + ngram + HMM + feature extraction) |
| **Performance** | 5-6x for POS tagging, 11x LM fitting, 25-39x LM generation — already benchmarked against NLTK |
| **Verdict** | ✅ **DEPEND ON IT for core NLP** — but only activate `perceptron_pos_tagger`, `lm`, `hmm`, `ngram`, `seq_feature` modules. Skip the CHAT/ELAN/TextGrid/CoNLL-U/SRT format converters. The crate is MIT, actively maintained, and the algorithms are tested. |

### 2.3 `sentencex` — Multilingual Sentence Segmentation

| Field | Value |
|---|---|
| **Crate** | `sentencex` v0.1.29 |
| **License** | MIT |
| **Maturity** | Active (May 2026). 519K recent downloads. |
| **What it provides** | Multilingual sentence boundary detection. Rule-based, similar to Punkt but without training. |
| **Does it match NLTK's Punkt?** | **No.** Punkt is trained on a corpus — it learns abbreviation lists, collocations, and capitalization patterns from data. `sentencex` is rule-based. They produce different results. |
| **Integration** | Could be a fallback for languages without trained Punkt models. Not a replacement for the trained English `sent_tokenize`. |
| **Verdict** | ❌ **NOT FOR CORE.** Punkt is trained, sentencex is rules. Different. But could be used as fallback for unsupported languages. |

### 2.4 `segtok` — Sentence + Word Tokenization

| Field | Value |
|---|---|
| **Crate** | `segtok` v0.1.5 |
| **License** | MIT |
| **Maturity** | Low version. 554K recent downloads. |
| **What it provides** | Sentence segmentation and word tokenization. |
| **Verdict** | ❌ **SKIP.** Similar to `sentencex` — doesn't match NLTK's Punkt-trained output. |

### 2.5 `crftag` / `crfrs` — CRF / Structured Perceptron Tagging

| Field | Value |
|---|---|
| **Crate** | `crfrs` v0.4 + `crftag` v0.2 |
| **License** | MIT |
| **What it provides** | CRF-based sequence labeling, structured perceptron with Viterbi decoding. |
| **Mapping to NLTK** | NLTK's `CRFTagger` and sequential `NgramTagger`/`TrigramTagger`. The Viterbi decoding is useful for TnT and HMM taggers too. |
| **Maturity** | Low downloads, but `crfrs` has a solid API for sequence labeling. |
| **Verdict** | 🔄 **CONSIDER** for CRF tagger. The Viterbi decoder could be reused for TnT. Evaluate API quality before committing. |

### 2.6 `postagger` — NLTK-Inspired Averaged Perceptron

| Field | Value |
|---|---|
| **Crate** | `postagger` v0.0.3 |
| **License** | MIT |
| **API** | `POSTagger::new(path)` — loads weights from file, then `tag(tokens)`. |
| **Maturity** | Very early (0.0.3). Requires external weight files. |
| **Verdict** | ❌ **SKIP.** `rustling`'s perceptron tagger is more mature and actively maintained. |

---

## Tier 3: Reference Implementations (study the algorithms, write fresh)

These crates are proof-of-concept, unmaintained, or wrong license.
We study their approach and write clean Rust.

### 3.1 `nltk_rs` by plutonium-guy

| Field | Value |
|---|---|
| **Repo** | `plutonium-guy/nltk_rs` |
| **License** | LGPL-3.0 |
| **What it does** | PyO3 bindings for regexp tokenization (batch + parallel), edit distance + Jaro/Jaro-Winkler (with parallel batch), HMM (Viterbi, forward, backward), K-means clustering, ALINE phonetic alignment. |
| **License issue** | **LGPL-3.0** — if we depend on it, our project must be LGPL-3.0 compatible. Our project uses Apache-2.0. License conflict. |
| **What can we use** | **Study only.** The regexp tokenizer code (handling of Python `re` flags → Rust `RegexBuilder` flags) is good reference. The parallel batch patterns (rayon `par_iter`) are reusable ideas. |
| **Verdict** | 🔄 **STUDY ONLY.** LGPL-3.0 incompatible with our Apache-2.0. Copy the *idea* of flag mapping and parallel batch processing, but write fresh code. |

### 3.2 `nalgebra` / `ndarray` — Matrix operations for Viterbi

| Field | Value |
|---|---|
| **Crate** | `nalgebra` or `ndarray` |
| **Usage** | Viterbi decoding in TnT/HMM, MaxEnt/GIS iteration, distance matrix computations |
| **Verdict** | 🔄 **CONSIDER.** For TnT's Viterbi, we can use a simple 2D `Vec<Vec<f64>>` — doesn't need full linear algebra. MaxEnt might benefit from `ndarray` for gradient computations. Start without, add if needed. |

### 3.3 `kenlm-rs` — KenLM wrapper for language models

| Field | Value |
|---|---|
| **Crate** | `kenlm-rs` |
| **License** | LGPL |
| **What it does** | Rust bindings to C++ KenLM. Supports Kneser-Ney smoothing, probing/trie data structures. |
| **Dependency weight** | Heavy — requires C++ compilation via `autocxx`, links to KenLM C++ library. |
| **Does it match NLTK?** | NLTK's LM module is pure-python ngram with smoothing (MLE, Lidstone, Laplace, WittenBell, KneserNey). KenLM is a production LM toolkit — far more optimized but different API. |
| **Verdict** | ❌ **SKIP.** Too heavy for what we need. `rustling`'s LM module is sufficient and pure-Rust. |

### 3.4 `whatlang` / `whichlang` — Language Detection

| Field | Value |
|---|---|
| **Crate** | `whatlang` v0.16 (MIT) |
| **What it does** | N-gram-based language detection |
| **Mapping to NLTK** | NLTK has `TextCat` classifier (language identification). |
| **Verdict** | 🔄 **CONSIDER** for TextCat replacement. `whatlang` supports 69 languages and is well-tested. Could replace NLTK's TextCat entirely with better accuracy and speed. |

### 3.5 `lindera` / `vaporetto` — Japanese NLP

| Field | Value |
|---|---|
| **Crate** | `lindera` (Japanese tokenization/morphological analysis, Apache-2.0) |
| **Mapping to NLTK** | NLTK wraps external Japanese tokenizers. Not core functionality. |
| **Verdict** | ❌ **SKIP.** Out of scope for v1.0. NLTK's Japanese support is thin (wrappers only). |

---

## Tier 4: Build & Infrastructure Crates

### 4.1 `pyo3` — Python Bindings

| Field | Value |
|---|---|
| **Version** | 0.23 (current stable). rustling uses 0.29.0-git. Use 0.23 for abi3-py38 compat. |
| **Feature** | `abi3-py38` + `extension-module` |
| **Verdict** | ✅ **REQUIRED.** Foundation of the project. |

### 4.2 `rayon` — Parallel Iteration

| Field | Value |
|---|---|
| **Version** | 1.10 |
| **Usage** | Parallel tokenization of multiple texts, batch edit distance, parallel collocation scoring |
| **Verdict** | ✅ **USE.** Proven by nltk_rs (regexp_tokenize_batch) and rustling (feature flagged). Make optional behind `parallel` feature. |

### 4.3 `serde` + `bincode` — Model Serialization

| Field | Value |
|---|---|
| **Usage** | Convert NLTK pickle models to portable binary format for Rust consumption |
| **Verdict** | ✅ **USE.** bincode is faster than JSON and smaller. Use for serializing Punkt models, tagger weights, ngram counters. |

### 4.4 `parking_lot` — Faster Mutex/RwLock

| Field | Value |
|---|---|
| **Usage** | Model cache (Punkt tokenizers, tagger weights loaded lazily) |
| **Verdict** | ✅ **USE.** RwLock<HashMap<String, Arc<Model>>> for thread-safe model loading. 3-5x faster than std::sync::RwLock on contended paths. |

---

## Summary Table: What We Actually Depend On

```
Cargo.toml dependencies:
  Direct (always):
    pyo3 = { version = "0.23", features = ["abi3-py38", "extension-module"] }
    regex = "1"
    unicode-segmentation = "1"
    rust-stemmers = "1"
    rustling = { version = "0.8", default-features = false, features = ["parallel"] }
    serde = { version = "1", features = ["derive"] }
    bincode = "2"
    once_cell = "1"
    hashbrown = "0.15"
    rustc-hash = "2"
    parking_lot = "0.12"

  Optional:
    rayon = "1"         # behind "parallel" feature
    whatlang = "0.16"   # for TextCat replacement

  dev-dependencies:
    approx = "0.5"      # float comparison in tests
    tempfile = "3"      # temp dirs for tests

Not depended on — code ported:
  - vtext algorithms (Apache-2.0, copy tokenization rules + string metrics)
  - nltk_rs techniques (LGPL-3.0, study only — write fresh)
```

### What Each Crate Saves Us

| Crate | Replaces NLTK module | NLTK LoC replaced | Our wrapper LoC | Net savings |
|---|---|---|---|---|
| `rust-stemmers` | `nltk.stem.snowball` (5,921 LoC) | 5,921 | 80 | **5,841 LoC** |
| `rustling` perceptron | `nltk.tag.perceptron` (407 LoC) + data loading | 407 | 120 | **287 LoC** (but more correct) |
| `rustling` LM | `nltk.lm.*` (1,020 LoC) + `nltk.probability` related | 1,020 | 150 | **870 LoC** |
| `rustling` HMM | `nltk.tag.hmm` (1,326 LoC) | 1,326 | 100 | **1,226 LoC** |
| `regex` | Foundation for all tokenizers (~4,000 LoC of Python `re` usage) | - | - | infinite (can't write regex engine) |
| `unicode-segmentation` | Unicode rules in various tokenizers (~300 LoC) | 300 | 10 | **290 LoC** |
| Ported vtext string metrics | `nltk.metrics.distance` (edit_distance, jaro, jaro_winkler) | 553 | 300 | **253 LoC** |
| **Total** | | **~9,527 LoC** | **~760 LoC** | **~8,767 LoC saved** |

Plus: **not having to write, test, and debug**:
- A Snowball stemmer in Rust (complex algorithmic port)
- An averaged perceptron tagger with correct feature extraction
- An ngram LM with Kneser-Ney smoothing
- A Unicode-aware sentence/word segmenter

---

## License Compatibility Matrix

| Crate | License | Our License (Apache-2.0) Compatible? |
|---|---|---|
| `rust-stemmers` | MIT | ✅ Yes |
| `regex` | MIT/Apache-2.0 | ✅ Yes |
| `unicode-segmentation` | MIT/Apache-2.0 | ✅ Yes |
| `rustling` | MIT | ✅ Yes |
| `hashbrown` | MIT/Apache-2.0 | ✅ Yes |
| `rustc-hash` | Apache-2.0/MIT | ✅ Yes |
| `parking_lot` | Apache-2.0/MIT | ✅ Yes |
| `pyo3` | Apache-2.0 | ✅ Yes |
| `rayon` | Apache-2.0/MIT | ✅ Yes |
| `serde` + `bincode` | MIT/Apache-2.0 | ✅ Yes |
| vtext (ported code) | Apache-2.0 | ✅ Yes (same license) |
| nltk_rs (study only) | LGPL-3.0 | ❌ No — cannot copy code |
| `whatlang` | MIT | ✅ Yes |
| `sentencex` | MIT | ✅ Yes |

All selected dependencies are MIT/Apache-2.0 compatible. No license conflicts.

---

## Complete Cargo.toml

```toml
[package]
name = "fastnltk"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[lib]
name = "fastnltk"
crate-type = ["cdylib"]

[dependencies]
# Core bindings
pyo3 = { version = "0.23", features = ["abi3-py38", "extension-module"] }

# Tokenization
regex = "1"
unicode-segmentation = "1"

# Stemming (saves 5,900 LoC — all Snowball algorithms)
rust-stemmers = "1"

# POS tagging + LM + HMM + ngram (saves ~2,500 LoC)
# Only activate modules we need, skip format converters
rustling = { version = "0.8", default-features = false, features = ["parallel"] }

# Serialization for model loading
serde = { version = "1", features = ["derive"] }
bincode = "2"

# Performance
hashbrown = "0.15"
rustc-hash = "2"
once_cell = "1"
parking_lot = "0.12"

# Optional parallel processing
rayon = { version = "1", optional = true }

# Language detection (for TextCat replacement)
whatlang = { version = "0.16", optional = true }

[features]
default = ["parallel"]
parallel = ["rayon", "rustling/parallel"]
language-detection = ["whatlang"]
```

## What We Still Write Ourselves

Despite using all these crates, we still write Rust for:

| Module | Why | Est LoC |
|---|---|---|
| Punkt sentence tokenizer | NLTK's Punkt has trained models. No existing Rust crate does this. Must port the algorithm + load pickle data. | ~1,200 |
| Treebank tokenizer | Contraction rules, Penn Treebank conventions. No existing crate. | ~400 |
| Tweet tokenizer | Emoji, hashtag, URL regex rules. NLTK-specific. | ~300 |
| RegexpTokenzier | Simple — wrap `regex` crate; handle both gap/match modes | ~150 |
| Simple tokenizers | LineTokenizer, SpaceTokenizer, TabTokenizer | ~100 |
| MWETokenizer | Multi-word expression matching | ~200 |
| Porter stemmer | Not in rust-stemmers (only Snowball). Port the algorithm. | ~200 |
| Lancaster stemmer | Not in rust-stemmers. Port the algorithm. | ~200 |
| WordNet lemmatizer (morphy) | Dictionary lookup + exception rules from WordNet data | ~300 |
| Other stemmers (ISRI, Cistem, RSLP, ARLSTem, Regexp) | Language-specific algorithms. Port each. | ~800 |
| FreqDist + ConditionalFreqDist | Counter-like with NLTK methods. Wrap hashbrown::HashMap. | ~400 |
| ProbDist types | MLE, Lidstone, Laplace, etc. Simple math. | ~500 |
| Collocation finders | Bigram/Trigram/Quadgram counters + association scoring | ~500 |
| String metrics (alignment, paice, segmentation) | aline.py port, association measures, windowdiff/pk | ~700 |
| Classifiers (NaiveBayes, MaxEnt, DecisionTree) | Core training loops in Rust | ~1,000 |
| `nltk.data` compatible loader | Find nltk_data directory, load pickles | ~300 |
| **Total Rust we write** | | **~6,250 LoC** |

**Without these crates we'd write ~15,000 LoC.** The crates cut our Rust workload by **~58%**. The remaining 6,250 LoC is unavoidable — Punkt, Treebank, Tweet, Porter, WordNet, classifiers, and the data layer are unique to NLTK's API.
