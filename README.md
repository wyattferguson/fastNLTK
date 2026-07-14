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

**68 automated benchmarks** across all 24 Rust modules. Average **28×** vs NLTK.
Every function below has an NLTK counterpart unless noted in [BENCHMARKS.md](BENCHMARKS.md).

| Module | Benchmarks | Best Speedup | Engine |
|---|---|---|---|
| [classify](BENCHMARKS.md) | 4 | **377×** | Maxent GIS training, NaiveBayes, TextCat |
| [metrics](BENCHMARKS.md) | 4 | **242×** | Pure algorithmic port, zero Python overhead |
| [tokenize](BENCHMARKS.md) | 16 | **145×** | Compiled regex via `regex` crate + logos |
| [tag](BENCHMARKS.md) | 9 | **98×** | rustling HMM, hashbrown FastMap lookups |
| [sentiment](BENCHMARKS.md) | 1 | **50×** | VADER in Rust, no regex re-compilation |
| [parse](BENCHMARKS.md) | 2 | **30×** | Earley parser in Rust, CFG grammar parsing |
| [collocations](BENCHMARKS.md) | 3 | **23×** | FastMap ngram frequency counting |
| [stem](BENCHMARKS.md) | 8 | **24×** | rust-stemmers (Snowball C), ISRI in Rust |
| [sem](BENCHMARKS.md) | 1 | **19×** | Expression parser in Rust |
| [translate](BENCHMARKS.md) | 1 | **10×** | BLEU in Rust |
| [tree](BENCHMARKS.md) | 1 | **10×** | Tree parser in Rust |
| [chunk](BENCHMARKS.md) | 1 | **7×** | Regexp chunk parser |
| [probability](BENCHMARKS.md) | 4 | **12×** | FreqDist, ConditionalFreqDist, prob dists |
| [cluster](BENCHMARKS.md) | 1 | **4×** | K-means Lloyd's algorithm |
| [lm](BENCHMARKS.md) | 6 | — | MLE, Lidstone, Laplace, StupidBackoff, KneserNey, WittenBell ¹ |
| [ccg](BENCHMARKS.md) | 1 | **2×** | CCG category parsing |
| [chat](BENCHMARKS.md) | 1 | **3×** | Eliza chatbot |
| [inference](BENCHMARKS.md) | 4 | — | Tableau, Resolution, Discourse, DefaultReasoner ¹ |

¹ fastNLTK-only — no NLTK equivalent or incompatible API.
| [sentiment](BENCHMARKS.md) | 1 | **46×** | VADER algorithm in Rust vs Python |
| [sem](BENCHMARKS.md) | 1 | **38×** | Recursive descent parser in native code |
| [parse](BENCHMARKS.md) | 1 | **27×** | Earley chart parsing in Rust |
| [stem](BENCHMARKS.md) | 4 | **22×** | `rust-stemmers` + algorithmic ports |
| [tree](BENCHMARKS.md) | 1 | **11×** | Bracket parser in Rust |
| [tag](BENCHMARKS.md) | 8 | **10×** | Hash lookups + compiled regex dispatch |
| [translate](BENCHMARKS.md) | 1 | **10×** | Tight DP loop in native code |
| [collocations](BENCHMARKS.md) | 1 | **9×** | HashMap counting in native code |
| [classify](BENCHMARKS.md) | 2 | **8×** | GIL-released training loops |
| [probability](BENCHMARKS.md) | 1 | **6×** | Hash table ops in native code |
| [chunk](BENCHMARKS.md) | 1 | **4×** | Compiled chunk grammar, no Python regex |
| [cluster](BENCHMARKS.md) | 1 | **4×** | K-means in native code |
| [chat](BENCHMARKS.md) | 1 | **4×** | Simple pattern match in Rust |
| [ccg](BENCHMARKS.md) | 1 | **2×** | Pure Rust string parsing |
| [lm](BENCHMARKS.md) | 2 | fastNLTK-only | Ngram + smoothing via `rustling` |
| [inference](BENCHMARKS.md) | 4 | fastNLTK-only | Recursive proof search in Rust |

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
cargo test                       # 279 Rust tests
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
