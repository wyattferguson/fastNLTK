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

| Component                      | Speedup vs NLTK | Engine                                      |
| ------------------------------ | --------------- | ------------------------------------------- |
| Regex tokenization             | 10–50×          | `regex` crate — DFA, no backtracking        |
| Punkt sentence detection       | 10–50×          | Direct algorithm port                       |
| Snowball stemming              | 15–20×          | `rust-stemmers` (libstemmer in Rust)        |
| POS tagging                    | 5–6×            | Averaged perceptron via `rustling`          |
| Language model scoring         | 5–39×           | Ngram + smoothing via `rustling`            |
| Segmentation (windowdiff / pk) | **48–118×**     | Direct algorithm port, zero Python overhead |
| Edit distance                  | 17–62×          | DP in native code                           |
| Classification training        | 3–8×            | GIL-released training loops                 |
| HMM tagging                    | 5–6×            | Viterbi decoding in Rust                    |
| Tree operations                | 5–20×           | Recursive traversal compiled to native      |
| Average (55 benchmarks)        | **23.9×**       | All modules combined                        |

[Full benchmark details →](BENCHMARKS.md)

## API Coverage

| Module         | Rust‑accelerated                                                                                   | Python shim      | Status |
| -------------- | -------------------------------------------------------------------------------------------------- | ---------------- | ------ |
| `tokenize`     | Treebank, Toktok, Tweet, Regexp, Space, MWE, TextTiling, Punkt, SExpr, Logos DFA                   | Fallback to NLTK | ✅     |
| `stem`         | Porter, Lancaster, Snowball, Regexp, WordNet, ARLSTem, Cistem, ISRI, RSLP                          | —                | ✅     |
| `tag`          | PerceptronTagger, TnT, HMM, DefaultTagger, Unigram/Bigram/TrigramTagger, RegexpTagger, AffixTagger | —                | ✅     |
| `classify`     | NaiveBayes, Maxent, TextCat                                                                        | —                | ✅     |
| `probability`  | FreqDist, ConditionalFreqDist, MLEProbDist, LaplaceProbDist                                        | —                | ✅     |
| `lm`           | MLE, Lidstone, Laplace, KneserNey, WittenBell, StupidBackoff                                       | Fallback to NLTK | ✅     |
| `collocations` | Bigram/Trigram/Quadgram finders                                                                    | —                | ✅     |
| `ccg`          | Chart parser, lexicon, combinators                                                                 | —                | ✅     |
| `inference`    | Tableau prover, Resolution prover, Discourse QA                                                    | —                | ✅     |
| `drt`          | DRS parsing, FOL conversion                                                                        | —                | ✅     |
| `sem`          | Expression parser, model evaluation                                                                | —                | ✅     |
| `metrics`      | Association, agreement, segmentation, distance, Jaccard, Spearman                                  | —                | ✅     |
| `chunk`        | Regexp chunker (NP/VP extraction)                                                                  | —                | ✅     |
| `cluster`      | K-means clustering                                                                                 | —                | ✅     |
| `sentiment`    | VADER sentiment analysis                                                                           | —                | ✅     |
| `parse`        | CFG, Earley chart parser                                                                           | —                | ✅     |
| `tree`         | Tree data structure (bracket parse, subtrees, productions)                                         | —                | ✅     |
| `corpus`       | NLTK corpus reader wrappers                                                                        | Reading API      | ✅     |
| `chat`         | Eliza-style chatbot                                                                                | —                | ✅     |
| `translate`    | BLEU score                                                                                         | —                | ✅     |
| `data`         | Resource finder, bincode cache                                                                     | —                | ✅     |

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

## Project Status

**v0.2.0** — Production-ready for all NLP pipelines. 90+ Rust exports, 21 Python shims,
275+ Rust tests, 254 Python integration tests. CI gated on correctness + performance regressions.

| Metric                         | Value                                                       |
| ------------------------------ | ----------------------------------------------------------- |
| Rust tests                     | **279** passing                                             |
| Python tests                   | **254** passing                                             |
| CI clippy                      | **0 errors** (`correctness` + `suspicious` + `perf` denied) |
| Unwraps in production          | **0** (all `Result` / expect with context)                  |
| Unsafe code                    | **0 lines** (`unsafe_code` denied at crate level)           |
| Benchmark regression detection | **25% threshold** in CI                                     |
| LTO                            | `fat` in release builds                                     |
| Allocator                      | `mimalloc` — up to 140% faster on CCG, MWE                  |
| Python wheel                   | `abi3-py38` — one wheel for CPython 3.8–3.13+               |

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
