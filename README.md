<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>Drop-in Rust-accelerated replacement for NLTK.</strong><br>
   <em>Same API. Same data. 5–50× faster.</em></p>

[![PyPI version](https://img.shields.io/pypi/v/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![Python](https://img.shields.io/pypi/pyversions/fastnltk.svg)](https://pypi.org/project/fastnltk/)
[![CI](https://github.com/fastnltk/fastnltk/actions/workflows/quality.yml/badge.svg)](https://github.com/fastnltk/fastnltk/actions/workflows/quality.yml)
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

## Platform Support

| Platform | Architecture | Wheel |
|----------|-------------|-------|
| Linux    | x86_64      | ✅    |
| Linux    | aarch64     | ✅    |
| macOS    | x86_64      | ✅    |
| macOS    | arm64       | ✅    |
| Windows  | x64         | ✅    |

Python 3.8–3.13, PyPy 3.9+. Rust 1.80+ for source builds.

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

**61 benchmarks** across all 17 modules. **Geometric mean 7.6×** vs NLTK (44 compared benchmarks).
[Full results →](BENCHMARKS.md)

| Operation | NLTK (ms) | fastNLTK (ms) | Speedup | Optimization |
|---|---|---|---|---|
| **windowdiff** | 2.47 | **0.01** | **168×** | Pure algorithmic port |
| **edit_distance** | 2.44 | **0.02** | **152×** | Damerau-Levenshtein in Rust |
| **pk** (segmentation) | 2.20 | **0.02** | **93×** | Segmentation metric in Rust |
| **TreebankWordDetokenizer** | 6.96 | **0.20** | **35×** | Single-pass undo |
| **PunktSentenceTokenizer** | 14.56 | **0.43** | **34×** | Byte-level sentence scan |
| **Expression.fromstring** | 17.45 | **0.58** | **30×** | FOL expression parser |
| **PerceptronTagger** | 17.64 | **0.66** | **27×** | u64 feature IDs, no String alloc |
| **CFG.from_string** | 0.05 | **0.002** | **26×** | Grammar parser in Rust |
| **TweetTokenizer** | 85.44 | **3.51** | **24×** | LazyLock regexes |
| **LancasterStemmer** | 33.86 | **2.01** | **17×** | Full 124-rule NLTK port |
| **TreebankWordTokenizer** | 42.56 | **2.54** | **17×** | O(n) scan + SIMD memchr3 |

### Module leaderboard

| Module | Geo Mean | Best Single | Engine |
|--------|----------|-------------|--------|
| [metrics](BENCHMARKS.md) | **133×** | 168× (windowdiff) | Pure algorithmic port |
| [parse](BENCHMARKS.md) | **24×** | 26× (CFG) | Earley + CFG in Rust |
| [sem](BENCHMARKS.md) | **30×** | 30× | FOL expression parser |
| [collocations](BENCHMARKS.md) | **13×** | 17× (Quadgram) | FastMap ngram counting |
| [tree](BENCHMARKS.md) | **10×** | 10× | Bracket parser in Rust |
| [translate](BENCHMARKS.md) | **8×** | 8× (BLEU) | BLEU in Rust |
| [stem](BENCHMARKS.md) | **8×** | 17× (Lancaster) | 124-rule NLTK port |
| [chunk](BENCHMARKS.md) | **8×** | 8× | Regexp chunk parser |
| [tokenize](BENCHMARKS.md) | **5×** | 35× (Detokenizer) | SIMD memchr3 + char scanner |
| [classify](BENCHMARKS.md) | **5×** | 8× (NaiveBayes) | Maxent GIS training |
| [tag](BENCHMARKS.md) | **4×** | 27× (Perceptron) | u64 IDs, integer Viterbi |
| [probability](BENCHMARKS.md) | **3×** | 4× (FreqDist) | FreqDist/ConditionalFreqDist |
| [chat](BENCHMARKS.md) | **3×** | 3× | Eliza chatbot |
| [ccg](BENCHMARKS.md) | **2×** | 2× | CCG category parsing |
| [lm](BENCHMARKS.md) | — | — | MLE, Lidstone, Laplace, KneserNey, WittenBell ¹ |
| [inference](BENCHMARKS.md) | — | — | Tableau, Resolution, Discourse ¹ |

¹ fastNLTK-only — NLTK has no equivalent benchmarks.

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
cargo test                       # 312 Rust tests
pytest tests/                     # 257 Python tests

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

## Citing

If you use fastNLTK in academic work, please cite:

```bibtex
@software{fastnltk2025,
  author       = {Wyatt Ferguson},
  title        = {fastNLTK: Drop-in Rust-accelerated replacement for NLTK},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/fastnltk/fastnltk}
}
```

## Contact

Created by [Wyatt Ferguson](https://github.com/wyattferguson)

- [GitHub @wyattferguson](https://github.com/wyattferguson)
- [BlueSky @wyattf](https://wyattf.bsky.social)
- [Email](mailto:wyattxdev@duck.com)
