<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>Drop-in Rust-accelerated replacement for NLTK.</strong><br>
   <em>Same API. Same data. 5–50× faster.</em></p>

[![PyPI version](https://img.shields.io/pypi/v/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![Python](https://img.shields.io/pypi/pyversions/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

**68 automated benchmarks** across all 16 Rust modules. **Geometric mean 15.0×** vs NLTK
on core tokenize/tag/stem operations (v0.4.0 optimizations). [Detailed results →](BENCHMARKS.md)

| Operation | NLTK | fastNLTK | Speedup | Optimization |
|---|---|---|---|---|
| **word_tokenize** (10K w) | 55.0 ms | **0.80 ms** | **68.7×** | Single-pass char scanner |
| **sent_tokenize** (10K w) | 14.1 ms | **0.49 ms** | **28.9×** | Byte-level sentence scan |
| **pos_tag** (1000 w) | 34.3 ms | **1.22 ms** | **28.2×** | u64 feature IDs, zero alloc |
| **TreebankWordTokenizer** (50K w) | 87.2 ms | **4.95 ms** | **17.6×** | O(n) scan + SIMD memchr3 |
| **PorterStemmer** (2000 w) | 20.7 ms | **2.23 ms** | **9.3×** | Pure Rust Snowball |
| **RegexpTokenizer** (50K w) | 7.43 ms | **3.01 ms** | **2.5×** | SIMD whitespace via memchr3 |

### Module leaderboard

| Module | Best Speedup | Engine |
| -------------------------------------------------------------- | ------------ | -------------------------------------------------------------- |
| [classify](BENCHMARKS.md) | **339×** | Maxent GIS training in Rust |
| [metrics](BENCHMARKS.md) | **168×** | Pure algorithmic port, zero Python overhead |
| [tokenize](BENCHMARKS.md) | **94×** | SIMD memchr3 + single-pass char scanner |
| [tag](BENCHMARKS.md) | **73×** | u64 feature IDs, FxHashMap, rustling HMM |
| [sentiment](BENCHMARKS.md) | **38×** | VADER in Rust, no regex re-compilation |
| [sem](BENCHMARKS.md) | **28×** | Expression parser in Rust |
| [parse](BENCHMARKS.md) | **26×** | Earley + CFG parsing |
| [collocations](BENCHMARKS.md) | **23×** | FastMap ngram frequency counting |
| [translate](BENCHMARKS.md) | **9×** | BLEU in Rust |
| [tree](BENCHMARKS.md) | **9×** | Tree parser in Rust |
| [chunk](BENCHMARKS.md) | **7×** | Regexp chunk parser |
| [probability](BENCHMARKS.md) | **6×** | FreqDist, ConditionalFreqDist, prob dists |
| [cluster](BENCHMARKS.md) | **4×** | K-means Lloyd's algorithm |
| [chat](BENCHMARKS.md) | **3×** | Eliza chatbot |
| [ccg](BENCHMARKS.md) | **2×** | CCG category parsing |
| [lm](BENCHMARKS.md) | — | MLE, Lidstone, Laplace, KneserNey, WittenBell ¹ |
| [inference](BENCHMARKS.md) | — | Tableau, Resolution, Discourse ¹ |

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
git clone https://github.com/fastnltk/fastnltk
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
git clone https://github.com/fastnltk/fastnltk
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

## Contact + Support

Created by [Wyatt Ferguson](https://github.com/wyattferguson)

For any questions or comments heres how you can reach me:

**:octopus: Follow me on [Github @wyattferguson](https://github.com/wyattferguson)**

**:mailbox_with_mail: Email me at [wyattxdev@duck.com](wyattxdev@duck.com)**

**:tropical_drink: Follow on [BlueSky @wyattf](https://wyattf.bsky.social)**

If you find this useful and want to tip me a little coffee money:

**:coffee: [Buy Me A Coffee](https://www.buymeacoffee.com/wyattferguson)**
