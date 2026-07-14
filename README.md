<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>Drop-in Rust-accelerated replacement for NLTK.</strong><br>
   <em>Same API. Same data. 5–50× faster.</em></p>

  [![CI](https://github.com/your/fastnltk/actions/workflows/quality.yml/badge.svg)](https://github.com/your/fastnltk/actions)
  [![PyPI version](https://img.shields.io/pypi/v/fastnltk.svg)](https://pypi.org/project/fastnltk/)
  [![Python](https://img.shields.io/pypi/pyversions/fastnltk.svg)](https://pypi.org/project/fastnltk/)
  [![Rust](https://img.shields.io/badge/rust-1.80%2B-blue)](https://www.rust-lang.org/)
  [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
  [![codecov](https://img.shields.io/codecov/c/github/your/fastnltk)](https://codecov.io/gh/your/fastnltk)
</div>

---

## Overview

fastNLTK is a **drop-in replacement** for the [Natural Language Toolkit](https://www.nltk.org)
that keeps the exact same Python API while replacing hot paths with native Rust code.
Import from `fastnltk` instead of `nltk` — no other changes needed. All NLTK corpus data works
without re-downloading.

NLTK is the most widely-used NLP teaching library in the world, but its pure-Python implementation
means regex, loops, and dict lookups all run through the interpreter. That's 10–50× slower than
compiled code — and production regressions have made NLTK's performance unpredictable
(e.g. 0.55s → 216s on 30K chars between versions).

fastNLTK preserves the exact API while replacing the engine.

```python
# Before
import nltk
nltk.download("punkt")
tokens = nltk.word_tokenize("Hello, world!")

# After — same code, different import
import fastnltk as nltk   # or: from fastnltk import word_tokenize
tokens = nltk.word_tokenize("Hello, world!")
```

## Performance

| Component | Speedup vs NLTK | Engine |
|---|---|---|
| Regex tokenization | 10–50× | `regex` crate — DFA, no backtracking |
| Punkt sentence detection | 10–50× | Direct algorithm port |
| Snowball stemming | 15–20× | `rust-stemmers` (libstemmer in Rust) |
| POS tagging | 5–6× | Averaged perceptron via `rustling` |
| Language model scoring | 5–39× | Ngram + smoothing via `rustling` |
| Segmentation (windowdiff / pk) | **48–118×** | Direct algorithm port, zero Python overhead |
| Edit distance | 17–62× | DP in native code |
| Classification training | 3–8× | GIL-released training loops |
| HMM tagging | 5–6× | Viterbi decoding in Rust |
| Tree operations | 5–20× | Recursive traversal compiled to native |
| Average (55 benchmarks) | **23.9×** | All modules combined |

[Full benchmark details →](BENCHMARKS.md)

## API Coverage

| Module | Rust‑accelerated | Python shim | Status |
|---|---|---|---|
| `tokenize` | Treebank, Toktok, Tweet, Regexp, Space, MWE, TextTiling, Punkt, SExpr, Logos DFA | Fallback to NLTK | ✅ |
| `stem` | Porter, Lancaster, Snowball, Regexp, WordNet, ARLSTem, Cistem, ISRI, RSLP | — | ✅ |
| `tag` | PerceptronTagger, TnT, HMM, DefaultTagger, Unigram/Bigram/TrigramTagger, RegexpTagger, AffixTagger | — | ✅ |
| `classify` | NaiveBayes, Maxent, TextCat | — | ✅ |
| `probability` | FreqDist, ConditionalFreqDist, MLEProbDist, LaplaceProbDist | — | ✅ |
| `lm` | MLE, Lidstone, Laplace, KneserNey, WittenBell, StupidBackoff | Fallback to NLTK | ✅ |
| `collocations` | Bigram/Trigram/Quadgram finders | — | ✅ |
| `ccg` | Chart parser, lexicon, combinators | — | ✅ |
| `inference` | Tableau prover, Resolution prover, Discourse QA | — | ✅ |
| `drt` | DRS parsing, FOL conversion | — | ✅ |
| `sem` | Expression parser, model evaluation | — | ✅ |
| `metrics` | Association, agreement, segmentation, distance, Jaccard, Spearman | — | ✅ |
| `chunk` | Regexp chunker (NP/VP extraction) | — | ✅ |
| `cluster` | K-means clustering | — | ✅ |
| `sentiment` | VADER sentiment analysis | — | ✅ |
| `parse` | CFG, Earley chart parser | — | ✅ |
| `tree` | Tree data structure (bracket parse, subtrees, productions) | — | ✅ |
| `corpus` | NLTK corpus reader wrappers | Reading API | ✅ |
| `chat` | Eliza-style chatbot | — | ✅ |
| `translate` | BLEU score | — | ✅ |
| `data` | Resource finder, bincode cache | — | ✅ |

## Quick Start

```bash
pip install fastnltk
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

```python
from fastnltk import word_tokenize, pos_tag, sent_tokenize
from fastnltk.corpus import nltk_data

# Tokenization
tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
print(tokens)
# → ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# Sentence segmentation
sents = sent_tokenize("Dr. Smith went home. He ate dinner.")
print(sents)
# → ['Dr. Smith went home.', 'He ate dinner.']

# POS tagging
tagged = pos_tag(tokens)
print(tagged)
# → [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ...]

# Parsing
from fastnltk import Tree
tree = Tree.from_string("(S (NP The/DT cat/NN) (VP runs/VBZ))")
print(tree.leaves())      # → ['The/DT', 'cat/NN', 'runs/VBZ']
print(tree.productions()) # → ['S -> NP VP', 'NP -> The/DT cat/NN', ...]
```

## Why fastNLTK?

**NLTK is the standard for NLP education.** It's used in every major NLP course,
every textbook, and thousands of tutorials. But NLTK's performance makes it
unsuitable for production and painful even for large-scale experimentation.

| | NLTK | fastNLTK | spaCy | Stanza |
|---|---|---|---|---|
| API style | Functional + OOP | **Identical to NLTK** | Pipeline-based | Pipeline-based |
| Tokenization | 1× | **10–50×** | ~8× (Cython) | ~3–5× (PyTorch) |
| Tagging | 1× | **5–6×** | ~10× | ~3–5× |
| Stemming | 1× | **15–20×** | N/A | N/A |
| Teaching readability | ✅ Yes | ✅ Yes (shim layer readable) | ❌ No | ❌ No |
| Corpus data | ✅ 50+ corpora | ✅ Same NLTK data | ❌ Limited | ❌ Limited |
| CPU-only | ✅ Yes | ✅ Yes | ✅ Yes | ❌ Needs GPU |
| Import change | — | `import fastnltk` | Complete rewrite | Complete rewrite |
| Neural models | ❌ No | ❌ No | ✅ Yes | ✅ Yes |

## Project Status

**v0.2.0** — Production-ready for all NLP pipelines. 90+ Rust exports, 21 Python shims,
275+ Rust tests, 254 Python integration tests. CI gated on correctness + performance regressions.

| Metric | Value |
|---|---|
| Rust tests | **279** passing |
| Python tests | **254** passing |
| CI clippy | **0 errors** (`correctness` + `suspicious` + `perf` denied) |
| Unwraps in production | **0** (all `Result` / expect with context) |
| Unsafe code | **0 lines** (`unsafe_code` denied at crate level) |
| Benchmark regression detection | **25% threshold** in CI |
| LTO | `fat` in release builds |
| Allocator | `mimalloc` — up to 140% faster on CCG, MWE |
| Python wheel | `abi3-py38` — one wheel for CPython 3.8–3.13+ |

## Installation

```bash
pip install fastnltk
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x64).
Requires Python 3.8+ and an existing NLTK data installation.

### From source

```bash
git clone https://github.com/your/fastnltk
cd fastnltk
pip install maturin
maturin develop --release   # Development install
# or
maturin build --release     # Build wheel
```

### Data

fastNLTK uses NLTK's corpus data. If you have NLTK installed with data,
no additional downloads are needed:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

## Performance Benchmarks

Measured on Intel i7-12700, 32GB RAM. All benchmarks compare `fastnltk` against `nltk`
on identical inputs. Times are median of 30+ iterations.

### Tokenization

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `sent_tokenize` | 30B | 0.12 | 0.01 | 12.0× |
| `sent_tokenize` | 1KB | 1.50 | 0.08 | 18.8× |
| `sent_tokenize` | 50KB | 58.20 | 2.10 | 27.7× |
| `sent_tokenize` | 1.2MB | 1,420.00 | 45.10 | 31.5× |
| `word_tokenize` | 30B | 0.15 | 0.01 | 15.0× |
| `word_tokenize` | 50KB | 72.10 | 3.80 | 19.0× |
| `RegexpTokenizer.tokenize` | 50KB | 45.30 | 1.50 | 30.2× |
| `SpaceTokenizer.tokenize` | 50KB | 8.40 | 0.20 | 42.0× |
| `TreebankWordTokenizer.tokenize` | 50KB | 62.10 | 3.10 | 20.0× |
| `TweetTokenizer.tokenize` | 50KB | 55.80 | 2.90 | 19.2× |

### Stemming & Lemmatization

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `SnowballStemmer.stem` | 10K words | 45.20 | 2.30 | 19.7× |
| `PorterStemmer.stem` | 10K words | 38.10 | 2.80 | 13.6× |
| `LancasterStemmer.stem` | 10K words | 42.50 | 2.60 | 16.3× |
| `WordNetLemmatizer.lemmatize` | 10K words | 120.40 | 11.20 | 10.8× |

### POS Tagging

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `pos_tag` | 100 sents | 25.40 | 4.50 | 5.6× |
| `pos_tag` | 1K sents | 248.10 | 39.80 | 6.2× |
| `PerceptronTagger.tag` | 100 sents | 18.90 | 3.10 | 6.1× |
| `TnT.tag` | 100 sents | 32.10 | 5.20 | 6.2× |

### Classification

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `NaiveBayesClassifier.train` | 10K instances | 850.00 | 180.00 | 4.7× |
| `NaiveBayesClassifier.classify` | 10K instances | 120.00 | 18.00 | 6.7× |
| `MaxentClassifier.train` | 5K instances | 3,200.00 | 520.00 | 6.2× |

### Language Models

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `MLE.fit` | 10K sents | 520.00 | 47.00 | 11.1× |
| `MLE.generate` | 1K tokens | 480.00 | 14.00 | 34.3× |
| `Lidstone.score` | 10K queries | 125.00 | 22.00 | 5.7× |

### Segmentation

| Function | Input | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|---|
| `windowdiff` | 12K chars | 3.14 | 0.03 | **118.0×** |
| `pk` | 12K chars | 2.81 | 0.06 | **48.1×** |

### Full Pipeline

| Pipeline | NLTK (ms) | fastNLTK (ms) | Speedup |
|---|---|---|---|
| tokenize → tag → chunk → NE | 1,850.00 | 210.00 | 8.8× |
| tokenize → stem → classify | 920.00 | 110.00 | 8.4× |
| sentence → word tokenize → pos tag | 280.00 | 42.00 | 6.7× |
| tag → parse → sem evaluate | 320.00 | 45.00 | 7.1× |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python user code                          │
│               import fastnltk / from fastnltk                │
├─────────────────────────────────────────────────────────────┤
│                    Python shim layer                         │
│          fastnltk/*.py — 21 shims, 1:1 API match            │
├─────────────────────────────────────────────────────────────┤
│                    PyO3 FFI boundary                         │
│              fastnltk._rust — PyO3 extension module          │
├─────────────────────────────────────────────────────────────┤
│              90+ Rust exports, 20+ modules                  │
│  tokenize  stem  tag  classify  lm  sem  inference  ...     │
│  mimalloc · hashbrown · smol_str · phf · smallvec · LTO    │
└─────────────────────────────────────────────────────────────┘
```

Each Python module in `fastnltk/` delegates hot paths to the compiled
`_rust` extension. Unimplemented functions fall back transparently
to NLTK with `try: from fastnltk._rust import ...; except: ...`.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed setup, code style,
testing, and PR workflow.

```bash
# Clone and build
git clone https://github.com/your/fastnltk
cd fastnltk
pip install -e ".[dev]"
maturin develop --release

# Run tests
cargo test --all-targets          # 279 Rust tests
pytest tests/                     # 254 Python tests

# Quality checks
cargo fmt --all -- --check
cargo clippy --all-targets
ruff check fastnltk/ tests/

# Benchmarks
maturin develop --release
python -m benchmarks.run --save   # Run + save results
```

## License

fastNLTK is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.

fastNLTK is not affiliated with, endorsed by, or sponsored by NLTK or
its maintainers. NLTK is a trademark of the NLTK Project.
