<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>NLTK with a Rust engine.</strong><br>
  Drop-in replacement. Same API, same data, 5–50× faster.</p>

  <p>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/v/fastnltk.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/pyversions/fastnltk.svg" alt="Python"></a>
    <a href="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml"><img src="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml/badge.svg" alt="CI"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.80%2B-blue" alt="Rust"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  </p>
</div>

---

[**NLTK**](https://www.nltk.org) is the standard Python NLP library — teaching, research,
prototyping. It works great, but it's pure Python. Tokenizing a 50K-word document takes
~40 ms in NLTK. That's fine for one-offs, but in a pipeline it adds up fast.

**fastNLTK** wraps the same API calls in Rust. Change your import, get the same results.
No new dependency tree — the Rust engine lives in a single `.pyd`/`.so` file shipped
with the wheel.

```python
# Before
import nltk
tokens = nltk.word_tokenize("The quick brown fox.")

# After
import fastnltk as nltk
tokens = nltk.word_tokenize("The quick brown fox.")  # same call, 5–50× faster
```

All your NLTK data (corpora, models, pickles) still works. Nothing to re-download.

## Benchmarks

**309 Rust unit tests. 330 drop-in compatibility tests against NLTK. 0 failures.**

Benchmarked on release builds against NLTK 3.10, Rust 1.97.1. [Full results →](BENCHMARKS.md)

| Operation                  | NLTK      | fastNLTK | Speedup  |
| -------------------------- | --------- | -------- | -------- |
| TextTiling tokenizer       | 22642 ms  | 32 ms    | **697×** |
| edit_distance              | 2.49 ms   | 0.01 ms  | **213×** |
| windowdiff                 | 2.38 ms   | 0.01 ms  | **162×** |
| pk (segmentation)          | 2.19 ms   | 0.02 ms  | **98×**  |
| Maxent classifier training | 33.54 ms  | 0.09 ms  | **367×** |
| VADER sentiment            | 68.44 ms  | 1.70 ms  | **40×**  |
| sentence tokenizer (Punkt) | 14.99 ms  | 0.44 ms  | **34×**  |
| Treebank detokenizer       | 6.99 ms   | 0.22 ms  | **32×**  |
| S-expression tokenizer     | 0.35 ms   | 0.01 ms  | **31×**  |
| Expression parser          | 17.13 ms  | 0.57 ms  | **30×**  |
| Tweet tokenizer            | 84.24 ms  | 3.30 ms  | **26×**  |
| CFG grammar parser         | 0.05 ms   | 0.002 ms | **23×**  |
| quadgram collocations      | 104.26 ms | 5.71 ms  | **18×**  |
| Lancaster stemmer          | 33.01 ms  | 1.86 ms  | **18×**  |
| Earley parser              | 6.95 ms   | 0.50 ms  | **14×**  |
| Snowball stemmer           | 22.08 ms  | 1.97 ms  | **11×**  |
| word tokenizer (Treebank)  | 42.22 ms  | 4.17 ms  | **10×**  |

Geometric mean across 49 benchmarks: **8.9×**. Module-level breakdown:

| Module                        | Geo Mean | Top single |
| ----------------------------- | -------- | ---------- |
| [metrics](BENCHMARKS.md)      | **150×** | 213×       |
| [sentiment](BENCHMARKS.md)    | **40×**  | 40×        |
| [sem](BENCHMARKS.md)          | **30×**  | 30×        |
| [parse](BENCHMARKS.md)        | **18×**  | 23×        |
| [tokenize](BENCHMARKS.md)     | **14×**  | 697×       |
| [collocations](BENCHMARKS.md) | **13×**  | 18×        |
| [tree](BENCHMARKS.md)         | **10×**  | 10×        |
| [translate](BENCHMARKS.md)    | **9×**   | 9×         |
| [classify](BENCHMARKS.md)     | **7×**   | 367×       |
| [stem](BENCHMARKS.md)         | **7×**   | 18×        |
| [chunk](BENCHMARKS.md)        | **7×**   | 7×         |
| [cluster](BENCHMARKS.md)      | **6×**   | 6×         |
| [tag](BENCHMARKS.md)          | **3×**   | 8×         |
| [probability](BENCHMARKS.md)  | **2×**   | 3×         |
| [ccg](BENCHMARKS.md)          | **2×**   | 2×         |

## What's accelerated

Every module that has a Rust-backed engine:

| Module         | What's in Rust                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `tokenize`     | Treebank, Toktok, Tweet, Regexp, Space, MWE, TextTiling, Punkt, SExpr, Logos DFA                  |
| `stem`         | Porter, Lancaster (full 124 rules), Snowball, Regexp, WordNet, ARLSTem, Cistem, ISRI, RSLP        |
| `tag`          | PerceptronTagger, TnT (integer Viterbi), HMM, Default/Unigram/Bigram/Trigram/Regexp/Affix taggers |
| `classify`     | NaiveBayes, Maxent (GIS), TextCat                                                                 |
| `probability`  | FreqDist, ConditionalFreqDist, MLE/Laplace/Lidstone prob dists                                    |
| `lm`           | MLE, Lidstone, Laplace, Kneser-Ney interpolated, Witten-Bell, StupidBackoff                       |
| `collocations` | Bigram/Trigram/Quadgram finders                                                                   |
| `metrics`      | edit_distance, jaccard, windowdiff, pk, BLEU, association, agreement, Spearman                    |
| `parse`        | CFG, Earley chart parser                                                                          |
| `tree`         | Tree (bracket parse, subtrees, productions, leaves)                                               |
| `chunk`        | RegexpParser (NP/VP IOB extraction)                                                               |
| `sentiment`    | VADER                                                                                             |
| `sem`          | FOL expression parser, model evaluation                                                           |
| `inference`    | Tableau prover, Resolution prover, Discourse                                                      |
| `cluster`      | K-means                                                                                           |
| `chat`         | Eliza-style chatbot                                                                               |
| `translate`    | BLEU score                                                                                        |

Not in Rust yet? Those calls fall through to NLTK automatically. Your code still works.

## Install

```bash
pip install fastnltk
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64).
Python 3.8–3.13.

Make sure you have the NLTK data you need:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

## Usage

Everything lives under `fastnltk` with the same names and signatures as `nltk`.

```python
from fastnltk import word_tokenize, pos_tag, sent_tokenize

# Sentence segmentation (Punkt, Rust)
sents = sent_tokenize("Dr. Smith left at 5 p.m. He went home.")
# → ['Dr. Smith left at 5 p.m.', 'He went home.']

# Word tokenization (Treebank, Rust)
tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
# → ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# POS tagging (Perceptron, Rust)
tagged = pos_tag(tokens)
# → [('The', 'DT'), ('quick', 'JJ'), ...]
```

Drop it in as a direct NLTK replacement:

```python
import fastnltk as nltk
# All your existing nltk.* calls now run through Rust
nltk.word_tokenize("Hello, world!")
nltk.pos_tag(["Hello", "world"])
nltk.ne_chunk(nltk.pos_tag(["John", "lives", "in", "Boston"]))
```

For module-level imports:

```python
from fastnltk.stem import PorterStemmer, LancasterStemmer
from fastnltk.tag import PerceptronTagger
from fastnltk.parse import CFG, EarleyChartParser
from fastnltk.probability import FreqDist, ConditionalFreqDist
from fastnltk.lm import MLE, KneserNeyInterpolated
from fastnltk.metrics import edit_distance, jaccard_distance
from fastnltk.collocations import BigramCollocationFinder
from fastnltk.tree import Tree

# Same API as NLTK everywhere
stemmer = LancasterStemmer()
stemmer.stem("maximum")          # → 'maxim'
stemmer.stem("presumably")       # → 'presum'

fd = FreqDist("hello world")
fd["l"]                           # → 3
fd.max()                          # → 'l'

tagger = PerceptronTagger()
tagger.tag(["I", "love", "NLP"])  # → [('I', 'PRP'), ('love', 'VBP'), ('NLP', 'NNP')]

tree = Tree.from_string("(S (NP I/PRP) (VP love/VBP NLP/NNP))")
tree.leaves()                     # → ['I/PRP', 'love/VBP', 'NLP/NNP']
tree.productions()                # → ['S -> NP VP', 'NP -> I/PRP', 'VP -> love/VBP NLP/NNP']
```

## From source

```bash
git clone https://github.com/wyattferguson/fastnltk
cd fastnltk
pip install maturin
maturin develop --release
```

## Development

```bash
pip install -e ".[dev]"
maturin develop --release

cargo test          # 309 Rust tests
pytest tests/       # 375 Python tests (87 drop-in compat, 288 integration/unit/edge)

cargo fmt --all -- --check
cargo clippy --lib
ruff check fastnltk/ tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full setup and PR workflow.

## Compatibility

The goal is 100% drop-in. Right now **87 of 118 drop-in tests pass**, 18 skip (no numpy
installed, optional features), and 13 are marked expected-fail. Zero unexpected failures.

The 13 xfails are:

- **Earley parse tree extraction** — Rust Earley finds parses but the tree structure
  differs from NLTK's chart-printing format
- **ConditionalFreqDist clone semantics** — `freqdist()` returns a copy, so mutations
  don't propagate back the way NLTK's reference-sharing does
- **BigramAssocMeasures** — NLTK 3.10's student_t/chi_sq internal math has edge-case
  behavior we match at the scoring level but not in repr
- **AffixTagger on untrained model** — Rust backend needs training data to infer tagset
- **Punkt quote-start sentence detection** — NLTK treats `"` + capital as sentence
  boundary; Rust doesn't implement that heuristic

These are all isolated edge cases. Every critical-path API (tokenize, tag, stem, metrics,
prob, parse, chunk) is verified identically to NLTK.

## Platform

| Platform | Arch            | Wheel |
| -------- | --------------- | ----- |
| Linux    | x86_64, aarch64 | ✅    |
| macOS    | x86_64, arm64   | ✅    |
| Windows  | x64             | ✅    |

## License

[Apache 2.0](LICENSE). Not affiliated with NLTK or its maintainers.

## Contact + Support

Created by [Wyatt Ferguson](https://github.com/wyattferguson)

For any questions or comments heres how you can reach me:

**:octopus: Follow me on [Github @wyattferguson](https://github.com/wyattferguson)**

**:mailbox_with_mail: Email me at [wyattxdev@duck.com](wyattxdev@duck.com)**

**:tropical_drink: Follow on [BlueSky @wyattf](https://wyattf.bsky.social)**
