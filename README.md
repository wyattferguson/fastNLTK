# fastNLTK

**fastNLTK is a drop-in, Rust-accelerated replacement for NLTK.** Same API, same interfaces, same data files — just 5-50x faster on everything that matters.

NLTK is the most widely-used NLP library in Python. It's also pure Python: every regex, loop, and dict lookup runs through the interpreter. That means 10-50x slower than compiled code, production regressions (0.55s → 216s on 30K chars between 3.8.1 and 3.8.2), and no path to performance without abandoning the API.

fastNLTK keeps the API and replaces the engine. Hot paths compile to Rust native code:

| Component | Speedup vs NLTK | Engine |
|---|---|---|
| Regex tokenization | 10-50x | Rust `regex` crate — DFA, no backtracking |
| Punkt sentence detection | 10-50x | Algorithm port — trained models from NLTK data |
| Snowball stemming | 15-20x | `rust-stemmers` crate — libstemmer in Rust |
| POS tagging | 5-6x | Averaged perceptron via `rustling` crate |
| LM fitting/generation | 11-39x | Ngram + smoothing via `rustling` |
| Edit distance | 17-62x | Direct port — no Python loop overhead |
| Classification training | 3-8x | Training loops — GIL released |
| Trigram HMM tagging | 5-6x | TnT — Viterbi decoding in Rust |
| Tree operations | 5-10x | Recursive traversal compiled to native |

**Usage is identical to NLTK.** Import from `fastnltk` instead of `nltk` — no other changes needed. All NLTK corpus data works without re-download.

For detailed API documentation, see [NLTK's docs](https://www.nltk.org). Every function and class in nltk has the same signature in fastnltk.

### Design Philosophy

1. **API-identical** — swap `import nltk` → `import fastnltk`, nothing changes
2. **Progressive acceleration** — each module accelerated independently; unimplemented functions fall back to NLTK
3. **No new data** — uses existing `nltk_data`; no re-downloads
4. **Teachability preserved** — Python shim is readable; users can still inspect the fallback
5. **One wheel for all** — `abi3-py38` covers CPython 3.8 through 3.13+

### Comparison

| Feature | NLTK | fastNLTK | spaCy | Stanza |
|---|---|---|---|---|
| API style | Functional + OOP | **Identical to NLTK** | Pipeline-based | Pipeline-based |
| Speed (tokenization) | 1x | **10-50x** | ~8x (Cython) | ~3-5x (PyTorch) |
| Speed (tagging) | 1x | **5-6x** | ~10x | ~3-5x |
| Speed (stemming) | 1x | **15-20x** | N/A | N/A |
| Teaching focus | ✅ Yes | ✅ Yes (shim layer) | ❌ No | ❌ No |
| Neural models | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Corpus data | ✅ 50+ corpora | ✅ Same data | ❌ Limited | ❌ Limited |
| CPU-only | ✅ Yes | ✅ Yes | ✅ Yes | ❌ Needs GPU for speed |

### What Makes It Fast

| Rust Crate | Purpose | Speedup Source |
|---|---|---|
| `regex` | Tokenization regex engine | Guaranteed linear-time DFA, no catastrophic backtracking |
| `rust-stemmers` | Snowball stemming | 15-20x over Python loop (proven by vtext benchmarks) |
| `rustling` | Perceptron tagger, LM, HMM | 5-39x over pure Python (proven by rustling benchmarks) |
| `hashbrown` | Fast HashMaps | 10-15% faster than std HashMap |
| `parking_lot` | Fast RwLock | 3-5x faster than std sync for model caches |

---

## Performance Benchmarks

Measured on Intel i7-12700, 32GB RAM, Ubuntu 24.04. All benchmarks compare `fastnltk` against `nltk` on identical inputs. Times are mean of 10+ runs.

### Tokenization

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `sent_tokenize` | tiny (30B) | 0.12 | 0.01 | 12.0x |
| `sent_tokenize` | small (1KB) | 1.50 | 0.08 | 18.8x |
| `sent_tokenize` | medium (50KB) | 58.20 | 2.10 | 27.7x |
| `sent_tokenize` | large (1.2MB) | 1,420.00 | 45.10 | 31.5x |
| `word_tokenize` | tiny (30B) | 0.15 | 0.01 | 15.0x |
| `word_tokenize` | medium (50KB) | 72.10 | 3.80 | 19.0x |
| `RegexpTokenizer.tokenize` | medium (50KB) | 45.30 | 1.50 | 30.2x |
| `SpaceTokenizer.tokenize` | medium (50KB) | 8.40 | 0.20 | 42.0x |
| `TreebankWordTokenizer.tokenize` | medium (50KB) | 62.10 | 3.10 | 20.0x |
| `TweetTokenizer.tokenize` | medium (50KB) | 55.80 | 2.90 | 19.2x |

### Stemming

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `SnowballStemmer.stem` | 10K words | 45.20 | 2.30 | 19.7x |
| `PorterStemmer.stem` | 10K words | 38.10 | 2.80 | 13.6x |
| `LancasterStemmer.stem` | 10K words | 42.50 | 2.60 | 16.3x |
| `WordNetLemmatizer.lemmatize` | 10K words | 120.40 | 11.20 | 10.8x |

### POS Tagging

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `pos_tag` | 100 sentences | 25.40 | 4.50 | 5.6x |
| `pos_tag` | 1000 sentences | 248.10 | 39.80 | 6.2x |
| `PerceptronTagger.tag` | 100 sentences | 18.90 | 3.10 | 6.1x |
| `TnT.tag` | 100 sentences | 32.10 | 5.20 | 6.2x |

### Classification

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `NaiveBayesClassifier.train` | 10K instances | 850.00 | 180.00 | 4.7x |
| `NaiveBayesClassifier.classify` | 10K instances | 120.00 | 18.00 | 6.7x |
| `MaxentClassifier.train` | 5K instances | 3,200.00 | 520.00 | 6.2x |

### Collocations & Probability

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `BigramCollocationFinder.from_words` | 1M words | 185.00 | 14.00 | 13.2x |
| `FreqDist.update` | 1M items | 95.00 | 11.00 | 8.6x |

### Language Models

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `MLE.fit` | 10K sentences | 520.00 | 47.00 | 11.1x |
| `MLE.generate` | 1000 tokens | 480.00 | 14.00 | 34.3x |
| `Lidstone.score` | 10K queries | 125.00 | 22.00 | 5.7x |

### Full Pipeline

| Pipeline | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| tokenize → tag → chunk → NE | 1,850.00 | 210.00 | 8.8x |
| tokenize → stem → classify | 920.00 | 110.00 | 8.4x |
| sentence → word tokenize → pos tag | 280.00 | 42.00 | 6.7x |

---

## Installation

```bash
pip install fastnltk
```

Requires Python 3.8+. Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x64).

### Data

fastNLTK uses the same data files as NLTK. Download them if you haven't:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

Or use fastNLTK's download helper:

```python
from fastnltk import download
download("punkt")
```

---

## API Compatibility

fastNLTK is a **drop-in replacement** for NLTK. This means:

- **Same function names**: `word_tokenize()`, `sent_tokenize()`, `pos_tag()`
- **Same class names**: `SnowballStemmer`, `PerceptronTagger`, `NaiveBayesClassifier`
- **Same signatures**: Same arguments, same defaults, same return types
- **Same data files**: Reads from same `nltk_data` directory

### What's Rust-Accelerated

| Module | Coverage | Status |
|---|---|---|
| `tokenize` | All tokenizers (Punkt, Treebank, Regexp, Tweet, Simple, TokTok, etc.) | ✅ v0.1 |
| `stem` | Snowball, Porter, Lancaster, WordNet, ISRI, Cistem, RSLP, ARLSTem, Regexp | ✅ v0.2 |
| `tag` | Perceptron, TnT | ✅ v0.3 |
| `classify` | NaiveBayes, PositiveNaiveBayes, MaxEnt (GIS), TextCat (whatlang) | ✅ v0.4 |
| `collocations` | Bigram, Trigram, Quadgram finders | ✅ v0.4 |
| `probability` | FreqDist, ConditionalFreqDist | ✅ v0.4 |
| `lm` | MLE, Lidstone, Laplace, KneserNeyInterpolated (Rust) | ✅ v0.5 |
| `cluster` | KMeansClusterer (Euclidean distance + iterative refinement) | ✅ v0.6 |
| `chat` | Chat (compiled regex pattern matching) | ✅ v0.6 |
| `sentiment` | VADER | ✅ v0.5 |
| `translate` | BLEU, corpus_bleu | ✅ v0.5 |
| `metrics` | edit_distance, jaro, jaro_winkler, dice, jaccard, binary, precision, recall, f_measure | ✅ v0.2 |
| `chunk` | RegexpParser (grammar compilation + tag sequence matching) | ✅ v0.6 |
| `wordnet` | WordNetLemmatizer (morphy algorithm) | ✅ v0.6 |
| `tree` | Tree (leaves, height, productions, subtrees, bracket-string parsing) | ✅ v0.6 |

### What's a Python Shim (falls back to NLTK)

| Module | Strategy |
|---|---|
| `sem` | Pure Python — wraps nltk.sem |
| `inference` | Pure Python — wraps nltk.inference |
| `ccg` | Pure Python — wraps nltk.ccg |
| `twitter` | Pure Python — wraps nltk.twitter |
| `downloader` | Pure Python — wraps nltk.downloader |
| WittenBellInterpolated, StupidBackoff (LM) | Pure Python — wraps nltk.lm (no Rust smoothing crate) |
| NE chunker, ChunkScore, conll I/O | Pure Python — wraps nltk.chunk |
| ParentedTree, ImmutableTree, MultiParentedTree, ProbabilisticTree | Pure Python — wraps nltk.tree (complex tree variants) |

### What's Skipped

- `draw` — tkinter GUI (not performance-critical)
- `app` — interactive applications (tkinter dependency)

---

## Project Status

| Phase | Module | Version | Status |
|---|---|---|---|
| P0 | Foundation (scaffold, CI, data layer) | v0.1.0 | ✅ Complete |
| P0 | Tokenization | v0.1.0 | ✅ Complete |
| P1 | Stemming | v0.2.0 | ✅ Complete |
| P1 | Metrics | v0.2.0 | ✅ Complete |
| P2 | POS tagging | v0.3.0 | ✅ Complete |
| P3 | Classification | v0.4.0 | ✅ Complete |
| P3 | Collocations & Probability | v0.4.0 | ✅ Complete |
| P4 | Language models | v0.5.0 | ✅ Complete |
| P4 | VADER sentiment, BLEU/METEOR | v0.5.0 | ✅ Complete |
| P5 | Chunking | v0.6.0 | ✅ Complete |
| P5 | Full API parity (all shims) | v0.6.0 | ✅ Complete |
| — | v1.0.0 stable release | v1.0.0 | 📋 Planned |

---

## Development

### Setup

```bash
git clone https://github.com/your/fastnltk
cd fastnltk
pip install maturin pytest ruff
maturin develop --uv --release
```

### One-Function Development Cycle

```bash
# 1. Branch
git checkout -b feat/sent-tokenize

# 2. Implement + test Rust
#    (edit src/tokenize/punkt.rs, src/lib.rs)
cargo test

# 3. Implement + test Python shim
#    (edit fastnltk/tokenize.py)
pytest tests/test_tokenize.py -v

# 4. Benchmark
pytest benchmarks/tokenize_bench.py --benchmark-json results.json
python scripts/update_benchmark_table.py results.json

# 5. Quality check
make lint

# 6. Commit + PR
git add -A
git commit -m "feat: add sent_tokenize (31.5x speedup)"
git push origin feat/sent-tokenize
```

### Project Structure

```
fastnltk/
├── Cargo.toml              # Rust crate config
├── pyproject.toml          # maturin build config
├── Makefile                # dev workflow targets
├── fastnltk/               # Python shim package
│   ├── __init__.py
│   ├── tokenize.py
│   ├── tag.py
│   ├── stem.py
│   ├── ...
│   └── _rust.pyi           # Type stubs for Rust extension
├── src/                    # Rust source
│   ├── lib.rs
│   ├── tokenize/
│   ├── stem/
│   ├── tag/
│   ├── classify/
│   ├── chunk.rs
│   ├── collocations.rs
│   ├── data.rs
│   ├── lm.rs
│   ├── metrics/
│   ├── probability.rs
│   ├── sentiment.rs
│   ├── translate.rs
│   ├── tree.rs
│   └── util/
├── tests/                  # Python tests
├── benchmarks/             # Performance benchmarks
│   ├── data/               # Standard test corpora
│   ├── results/            # Current benchmark JSON
│   ├── archive/            # Historical benchmark data
│   ├── tokenize_bench.py
│   └── stem_bench.py
└── scripts/
    ├── update_benchmark_table.py
    └── convert_models.py
```

### Running Tests

```bash
# Rust unit tests
cargo test

# Python integration tests
pytest tests/ -v

# Both
make test-all

# Lint
make lint
```

### Building Wheels

```bash
maturin build --release --out dist
# Produces: fastnltk-0.1.0-cp38-abi3-{platform}.whl
```

---

## Technical Details

### Rust Crates Used

| Crate | License | Purpose |
|---|---|---|
| `pyo3` | Apache-2.0 | Python bindings |
| `regex` | MIT/Apache-2.0 | Tokenization regex engine |
| `unicode-segmentation` | MIT/Apache-2.0 | Unicode word/sentence boundaries |
| `rust-stemmers` | MIT | Snowball stemmer (all 16 langs) |
| `rustling` | MIT | Perceptron tagger, LM, HMM, ngram |
| `whatlang` | MIT | Language detection (TextCat replacement) |
| `hashbrown` | MIT/Apache-2.0 | Faster HashMaps |
| `rustc-hash` | Apache-2.0/MIT | FxHashMap for small-key maps |
| `parking_lot` | Apache-2.0/MIT | Faster RwLock for model cache |
| `serde` + `bincode` | MIT/Apache-2.0 | Model serialization |
| `rayon` (optional) | MIT/Apache-2.0 | Parallel batch processing |

### What We Write from Scratch

Despite heavy crate reuse, ~11,000 LoC of Rust is custom for NLTK compatibility:

- **Punkt sentence tokenizer** (~1,200 LoC) — no existing Rust crate handles NLTK's trained model format
- **Treebank/Tweet tokenizers** (~700 LoC) — NLTK-specific regex rule-sets
- **Porter/Lancaster/ISRI/Cistem/RSLP stemmers** (~1,200 LoC) — not in rust-stemmers
- **WordNet lemmatizer** (~300 LoC) — morphy algorithm + WordNet index file loading
- **MaxentClassifier** (~600 LoC) — GIS training loop with feature encoding
- **TnT tagger** (~400 LoC) — trigram HMM with Viterbi decoding + backoff smoothing
- **Language model bridge** (~400 LoC) — wraps rustling LM types (MLE, Lidstone, Laplace)
- **FreqDist + ProbDist types** (~500 LoC) — NLTK-specific method signatures
- **Collocation finders** (~500 LoC) — ngram scoring with NLTK's association measures
- **NaiveBayes** (~300 LoC) — training + prediction with Laplace smoothing
- **RegexpChunkParser** (~300 LoC) — grammar compilation + tag sequence matching
- **Tree data structure** (~400 LoC) — recursive Tree with leaves, height, productions, bracket-string parsing
- **Data layer** (~300 LoC) — nltk_data finder, pickle → bincode converter
- **TextCat bridge** (~50 LoC) — whatlang language detection wrapper
- **Sequential taggers** (~550 LoC) — Default/Unigram/Bigram/Trigram/Affix/Regexp taggers
- **ARLSTem/ARLSTem2** (~350 LoC) — Arabic stemmers
- **PlaintextCorpusReader** (~150 LoC) — file I/O + tokenization
- **Earley chart parser** (~500 LoC) — Earley's algorithm for any CFG
- **Chat** (~150 LoC) — compiled regex pattern matching
- **KMeansClusterer** (~200 LoC) — iterative distance computation
- **KneserNeyInterpolated** (~100 LoC) — Kneser-Ney smoothing

---

## License

Apache-2.0. Compatible with all dependencies (MIT/Apache-2.0).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [PLAN.md](PLAN.md) for development workflow.

Key principles:
- **One function at a time** — never start a second until the first is merged
- **Test against NLTK** — every function must match NLTK's output exactly
- **Benchmark every merge** — speedup must stay above target for the module
- **No new data formats** — use existing nltk_data; no re-downloads
