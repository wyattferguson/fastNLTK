<div align="center">
  <h1>fastNLTK</h1>
  <p><strong>NLTK with a Rust engine.</strong><br>
  Drop-in replacement. Same API, Same data, 12× faster.</p>

  <p>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/v/fastnltk.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/fastnltk/"><img src="https://img.shields.io/pypi/pyversions/fastnltk.svg" alt="Python"></a>
    <a href="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml"><img src="https://github.com/wyattferguson/fastnltk/actions/workflows/quality.yml/badge.svg" alt="CI"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.97%2B-blue" alt="Rust"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  </p>
</div>

---

[**NLTK**](https://www.nltk.org) is fantastic. The API is clean, and it does just about everything you'd want from an NLP library. The only drawback being it's pure Python, and on large inputs that starts to show. A single
call is fine. A million calls in a data pipeline is a different story.

**fastNLTK** is NLTK with the hot path rewritten in Rust. Same API, same
data, same results. Just faster — 5× to 700× depending on what you're doing.
No new dependencies, no YAML files, no config. Simply change your import and watch your code fly.

```python
# Before
import nltk
tokens = nltk.word_tokenize("The quick brown fox.")

# After
import fastnltk as nltk
tokens = nltk.word_tokenize("The quick brown fox.")  # same call, 5–50× faster
```

Your NLTK data (corpora, models, pickles, all of it) still works. Nothing to
re-download, nothing to migrate.

## Benchmarks

**368 Python drop-in compatibility tests against NLTK. 6 skipped (chat stdin). 1 expected failure.**

Benchmarked on release builds against NLTK 3.10. [Full results →](BENCHMARKS.md)

| Operation                  | NLTK      | fastNLTK | Speedup  |
| -------------------------- | --------- | -------- | -------- |
| HMM tagger                 | 16.06 ms  | 0.15 ms  | **104×** |
| TextTiling tokenizer       | 4.69 ms   | 0.06 ms  | **77×**  |
| Treebank detokenizer       | 11.25 ms  | 0.15 ms  | **73×**  |
| S-expression tokenizer     | 1.29 ms   | 0.02 ms  | **60×**  |
| Punkt sentence tokenizer   | 17.65 ms  | 0.17 ms  | **106×** |
| Tweet tokenizer            | 68.03 ms  | 1.56 ms  | **44×**  |
| Sentiment (VADER)          | 30.23 ms  | 0.96 ms  | **32×**  |
| Lancaster stemmer          | 54.98 ms  | 2.15 ms  | **26×**  |
| CFG grammar parser         | 0.11 ms   | 0.00 ms  | **28×**  |
| quadgram collocations      | 101.73 ms | 3.01 ms  | **34×**  |
| edit_distance              | 4.55 ms   | 0.03 ms  | **165×** |
| Trigram collocations       | 49.87 ms  | 2.43 ms  | **21×**  |
| Snowball stemmer           | 44.40 ms  | 2.89 ms  | **15×**  |
| Regexp tagger              | 19.59 ms  | 1.66 ms  | **12×**  |
| Tree from_string           | 6.46 ms   | 0.63 ms  | **10×**  |

Geometric mean across 48 benchmarks: **12.2×**. Module-level breakdown:

| Module                        | Geo Mean | Top single |
| ----------------------------- | -------- | ---------- |
| [metrics](BENCHMARKS.md)      | **137×** | 165×       |
| [tag](BENCHMARKS.md)          | **7×**   | 104×       |
| [sentiment](BENCHMARKS.md)    | **34×**  | 34×        |
| [sem](BENCHMARKS.md)          | **34×**  | 34×        |
| [parse](BENCHMARKS.md)        | **19×**  | 28×        |
| [tokenize](BENCHMARKS.md)     | **6×**   | 120×       |
| [collocations](BENCHMARKS.md) | **14×**  | 35×        |
| [tree](BENCHMARKS.md)         | **12×**  | 12×        |
| [translate](BENCHMARKS.md)    | **9×**   | 9×         |
| [stem](BENCHMARKS.md)         | **9×**   | 26×        |
| [chunk](BENCHMARKS.md)        | **8×**   | 8×         |
| [classify](BENCHMARKS.md)     | **5×**   | 562×       |
| [cluster](BENCHMARKS.md)      | **5×**   | 5×         |
| [chat](BENCHMARKS.md)         | **4×**   | 4×         |
| [ccg](BENCHMARKS.md)          | **3×**   | 3×         |
| [probability](BENCHMARKS.md)  | **3×**   | 6×         |

## What's accelerated

Every module that has a Rust-backed engine:

| Module         | What's in Rust                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `tokenize`     | Treebank, Toktok, Tweet, Regexp, Space, MWE, TextTiling, Punkt, SExpr, Logos DFA                  |
| `stem`         | Porter, Lancaster (full 124 rules), Snowball, Regexp, WordNet, ARLSTem, Cistem, ISRI, RSLP        |
| `tag`          | PerceptronTagger, HMM (integer Viterbi), TnT, Default/Unigram/Bigram/Trigram/Regexp/Affix |
| `classify`     | NaiveBayes, Maxent (GIS), TextCat                                                                 |
| `corpus`       | PlaintextCorpusReader, TaggedCorpusReader, CategorizedPlaintextCorpusReader                        |
| `probability`  | FreqDist, ConditionalFreqDist (shared references), MLE/Laplace/Lidstone prob dists                |
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
Python 3.10–3.13.

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

cargo test          # Rust unit tests (309 pass)
pytest tests/       # 375 Python tests (368 pass, 6 skip, 1 xfail)

cargo fmt --all -- --check
cargo clippy --lib
ruff check fastnltk/ tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full setup and PR workflow.

## Compatibility

The goal is 100% drop-in. Right now **368 of 375 tests pass** (6 skipped — chat bots
read stdin), with only **1 expected failure**:

- **CCG `fromstring`** — NLTK 3.10's `ccg.chart.fromstring` is broken (upstream bug)

Every critical-path API (tokenize, tag, stem, metrics, prob, parse, chunk, sentiment,
classify, collocations, tree, cluster, translate, chat) is verified byte-identical
to NLTK across all tested inputs.

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
