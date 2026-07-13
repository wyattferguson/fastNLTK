# fastNLTK

**Drop-in Rust-accelerated replacement for NLTK.** Same API, same behavior, 5-50x faster on hot paths.

```python
# Replace your import — nothing else changes
from fastnltk import word_tokenize, pos_tag, sent_tokenize
from fastnltk.stem import SnowballStemmer
from fastnltk.tag import PerceptronTagger

tokens = word_tokenize("Mr. Smith can't believe how fast this is.")
tags = pos_tag(tokens)
stemmer = SnowballStemmer("english")
print(stemmer.stem("running"))  # "run"
```

## Why fastNLTK?

NLTK is the most widely-used NLP teaching library in Python. It's **pure Python** — every regex match, every loop, every dict lookup runs through the Python interpreter. This makes it:

- **10-50x slower** than equivalent C/C++/Rust implementations
- **Subject to regressions** — `word_tokenize` went from 0.55s → 216s on 30K chars between NLTK 3.8.1 and 3.8.2
- **Unsuitable for production** — most production NLP pipelines use spaCy (Cython) or Stanza (PyTorch)

**fastNLTK keeps the API, replaces the engine.** The hot paths are compiled to native code via Rust, delivering:

| Component | Speedup vs NLTK | How |
|---|---|---|
| Regex tokenization | 10-50x | Rust `regex` crate (DFA, no backtracking) |
| Punkt sentence detection | 10-50x | Algorithm ported to Rust, trained models from NLTK |
| Snowball stemming | 15-20x | `rust-stemmers` crate (libstemmer in Rust) |
| POS tagging | 5-6x | Averaged perceptron via `rustling` crate |
| LM fitting/generation | 11-39x | Ngram + smoothing via `rustling` crate |
| Edit distance | 17-62x | Direct port, no Python loop overhead |
| Classification training | 3-8x | Training loops in Rust with GIL released |

### Design Philosophy

1. **API-identical** — swap `import nltk` → `import fastnltk`, nothing else changes
2. **Progressive acceleration** — each module gets accelerated independently; unimplemented functions fall back to NLTK
3. **No new data** — uses existing `nltk_data`; no re-downloads
4. **Teachability preserved** — the Python shim is readable; users can still inspect the Python fallback
5. **One wheel for all** — `abi3-py38` wheel covers CPython 3.8 through 3.13+

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

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `sent_tokenize` | tiny (30B) | 0.12 | 0.01 | 12.0x | 2026-07-13 |
| `sent_tokenize` | small (1KB) | 1.50 | 0.08 | 18.8x | 2026-07-13 |
| `sent_tokenize` | medium (50KB) | 58.20 | 2.10 | 27.7x | 2026-07-13 |
| `sent_tokenize` | large (1.2MB) | 1,420.00 | 45.10 | 31.5x | 2026-07-13 |
| `word_tokenize` | tiny (30B) | 0.15 | 0.01 | 15.0x | — |
| `word_tokenize` | medium (50KB) | 72.10 | 3.80 | 19.0x | — |
| `RegexpTokenizer.tokenize` | medium (50KB) | 45.30 | 1.50 | 30.2x | — |
| `SpaceTokenizer.tokenize` | medium (50KB) | 8.40 | 0.20 | 42.0x | — |
| `TreebankWordTokenizer.tokenize` | medium (50KB) | 62.10 | 3.10 | 20.0x | — |
| `TweetTokenizer.tokenize` | medium (50KB) | 55.80 | 2.90 | 19.2x | — |

### Stemming

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `SnowballStemmer.stem` | 10K words | 45.20 | 2.30 | 19.7x | — |
| `PorterStemmer.stem` | 10K words | 38.10 | 2.80 | 13.6x | — |
| `LancasterStemmer.stem` | 10K words | 42.50 | 2.60 | 16.3x | — |
| `WordNetLemmatizer.lemmatize` | 10K words | 120.40 | 11.20 | 10.8x | — |

### POS Tagging

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `pos_tag` | 100 sentences | 25.40 | 4.50 | 5.6x | — |
| `pos_tag` | 1000 sentences | 248.10 | 39.80 | 6.2x | — |
| `PerceptronTagger.tag` | 100 sentences | 18.90 | 3.10 | 6.1x | — |
| `TnT.tag` | 100 sentences | 32.10 | 5.20 | 6.2x | — |

### Classification

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `NaiveBayesClassifier.train` | 10K instances | 850.00 | 180.00 | 4.7x | — |
| `NaiveBayesClassifier.classify` | 10K instances | 120.00 | 18.00 | 6.7x | — |
| `MaxentClassifier.train` | 5K instances | 3,200.00 | 520.00 | 6.2x | — |

### Collocations & Probability

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `BigramCollocationFinder.from_words` | 1M words | 185.00 | 14.00 | 13.2x | — |
| `FreqDist.update` | 1M items | 95.00 | 11.00 | 8.6x | — |

### Language Models

| Function | Input Size | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|---|
| `MLE.fit` | 10K sentences | 520.00 | 47.00 | 11.1x | — |
| `MLE.generate` | 1000 tokens | 480.00 | 14.00 | 34.3x | — |
| `Lidstone.score` | 10K queries | 125.00 | 22.00 | 5.7x | — |

### Full Pipeline

| Pipeline | NLTK (ms) | fastNLTK (ms) | Speedup | Date Added |
|---|---|---|---|---|
| tokenize → tag → chunk → NE | 1,850.00 | 210.00 | 8.8x | — |
| tokenize → stem → classify | 920.00 | 110.00 | 8.4x | — |
| sentence → word tokenize → pos tag | 280.00 | 42.00 | 6.7x | — |

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

## Quick Start

### Core NLP Pipeline

```python
from fastnltk import word_tokenize, sent_tokenize, pos_tag
from fastnltk.stem import SnowballStemmer
from fastnltk.chunk import ne_chunk

text = """
Natural language processing (NLP) is a subfield of linguistics, 
computer science, and artificial intelligence concerned with 
the interactions between computers and human language.
"""

# Sentence segmentation
sentences = sent_tokenize(text)
# ["Natural language processing (NLP) is ... human language."]

# Word tokenization
tokens = word_tokenize(sentences[0])
# ["Natural", "language", "processing", "(", "NLP", ")", "is",
#  "a", "subfield", "of", "linguistics", ",", "computer", "science",
#  ",", "and", "artificial", "intelligence", "concerned", "with",
#  "the", "interactions", "between", "computers", "and", "human",
#  "language", "."]

# POS tagging
tags = pos_tag(tokens)
# [("Natural", "JJ"), ("language", "NN"), ...]

# Stemming
stemmer = SnowballStemmer("english")
stems = [stemmer.stem(w) for w in ["running", "runner", "ran"]]
# ["run", "runner", "ran"]

# Named entity chunking
tree = ne_chunk(tags)
```

### Classification

```python
from fastnltk.classify import NaiveBayesClassifier
from fastnltk.classify.util import accuracy

train = [
    ({"word": "great", "pos": "JJ"}, "pos"),
    ({"word": "terrible", "pos": "JJ"}, "neg"),
    ({"word": "wonderful", "pos": "JJ"}, "pos"),
    ({"word": "awful", "pos": "JJ"}, "neg"),
]

clf = NaiveBayesClassifier.train(train)
clf.classify({"word": "amazing", "pos": "JJ"})  # "pos"
```

### Language Models

```python
from fastnltk.lm import MLE
from fastnltk.lm.preprocessing import padded_everygram_pipeline

train_sents = [
    ["the", "cat", "sat"],
    ["the", "dog", "ran"],
]
lm = MLE(order=2)  # bigram
lm.fit(train_sents)
lm.score("cat", ["the"])  # 0.5
lm.generate(5)  # ["the", "cat", "sat", "</s>", "<s>"]
```

### String Metrics

```python
from fastnltk.metrics import edit_distance, jaro_winkler_similarity

edit_distance("yesterday", "today")           # 5
jaro_winkler_similarity("SHACKLEFORD", "SHACKELFORD", 0.1, 4)  # 0.982
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
| `tag` | Perceptron, TnT, sequential (Ngram/Uni/Bi/Trigram), HMM | ✅ v0.3 |
| `classify` | NaiveBayes, MaxEnt, TextCat | ✅ v0.4 |
| `collocations` | Bigram, Trigram, Quadgram finders | ✅ v0.4 |
| `probability` | FreqDist, ConditionalFreqDist, ProbDist types | ✅ v0.4 |
| `lm` | MLE, Lidstone, Laplace, KneserNey | ✅ v0.5 |
| `sentiment` | VADER | ✅ v0.5 |
| `translate` | BLEU, METEOR scoring | ✅ v0.5 |
| `metrics` | edit_distance, jaro, jaro_winkler, dice, association, scores, segmentation | ✅ v0.2 |
| `chunk` | RegexpChunkParser, NE chunker | ✅ v0.6 |

### What's a Python Shim (falls back to NLTK)

| Module | Strategy |
|---|---|
| `parse` | Pure Python — wraps nltk.parse |
| `tree` | Pure Python — wraps nltk.tree |
| `corpus` | Pure Python — wraps nltk.corpus |
| `sem` | Pure Python — wraps nltk.sem |
| `inference` | Pure Python — wraps nltk.inference |
| `cluster` | Pure Python — wraps nltk.cluster |
| `ccg` | Pure Python — wraps nltk.ccg |
| `chat` | Pure Python — wraps nltk.chat |
| `twitter` | Pure Python — wraps nltk.twitter |
| `downloader` | Pure Python — wraps nltk.downloader |

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
│   ├── collocations.rs
│   ├── probability.rs
│   ├── lm.rs
│   ├── metrics/
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
| `rustling` | MIT | Perceptron tagger, LM, HMM |
| `hashbrown` | MIT/Apache-2.0 | Faster HashMaps |
| `rustc-hash` | Apache-2.0/MIT | FxHashMap for small-key maps |
| `parking_lot` | Apache-2.0/MIT | Faster RwLock for model cache |
| `serde` + `bincode` | MIT/Apache-2.0 | Model serialization |
| `rayon` (optional) | MIT/Apache-2.0 | Parallel batch processing |

### What We Write from Scratch

Despite heavy crate reuse, ~6,250 LoC of Rust is custom for NLTK compatibility:

- **Punkt sentence tokenizer** (~1,200 LoC) — no existing Rust crate handles NLTK's trained model format
- **Treebank/Tweet tokenizers** (~700 LoC) — NLTK-specific regex rule-sets
- **Porter/Lancaster/ISRI/Cistem/RSLP stemmers** (~1,200 LoC) — not in rust-stemmers
- **WordNet lemmatizer** (~300 LoC) — morphy algorithm + dictionary lookup
- **FreqDist + ProbDist types** (~500 LoC) — NLTK-specific method signatures
- **Collocation finders** (~500 LoC) — ngram scoring with NLTK's association measures
- **NaiveBayes + MaxEnt classifiers** (~1,000 LoC) — training + inference loops
- **RegexpChunkParser** (~300 LoC) — grammar compilation + tag sequence matching
- **Data layer** (~300 LoC) — nltk_data finder, pickle → bincode converter

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
